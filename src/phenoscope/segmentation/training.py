"""Training / fine-tuning loops for the U-Net and SAM mask decoder.

Both loops consume ``(image_cyx, labels_yx)`` pairs from a
:class:`phenoscope.benchmarks.datasets.InstanceDataset` and log the
instance-segmentation AP on a held-out split after each epoch.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phenoscope.config import get_settings, resolve_device
from phenoscope.evaluation.instance import average_precision
from phenoscope.preprocessing.normalize import percentile_normalize
from phenoscope.segmentation.unet import build_unet, labels_to_three_class, three_class_to_labels

log = logging.getLogger(__name__)

Pair = tuple[np.ndarray, np.ndarray]


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patch: int = 256
    seed: int = 0
    out: str = "models/unet_nuclei.pt"


def _random_patch(img: np.ndarray, lab: np.ndarray, size: int, rng: random.Random) -> Pair:
    _, h, w = img.shape
    if h < size or w < size:
        pad = ((0, 0), (0, max(0, size - h)), (0, max(0, size - w)))
        img = np.pad(img, pad, mode="reflect")
        lab = np.pad(lab, pad[1:], mode="constant")
        _, h, w = img.shape
    y, x = rng.randint(0, h - size), rng.randint(0, w - size)
    img, lab = img[:, y : y + size, x : x + size], lab[y : y + size, x : x + size]
    # Dihedral augmentation.
    k = rng.randint(0, 3)
    img, lab = np.rot90(img, k, axes=(1, 2)), np.rot90(lab, k)
    if rng.random() < 0.5:
        img, lab = img[:, :, ::-1], lab[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(lab)


def train_unet(
    train: Sequence[Pair], val: Sequence[Pair], cfg: TrainConfig | None = None
) -> dict[str, float]:
    import torch
    from monai.losses import DiceCELoss

    cfg = cfg or TrainConfig()
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(get_settings().device)
    in_ch = train[0][0].shape[0]
    net = build_unet(in_ch).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    steps = max(1, len(train) // cfg.batch_size)

    best = {"ap50": -1.0, "epoch": -1}
    for epoch in range(cfg.epochs):
        net.train()
        for _ in range(steps):
            batch = [_random_patch(*rng.choice(train), cfg.patch, rng) for _ in range(cfg.batch_size)]
            x = np.stack([percentile_normalize(b[0], axis=(1, 2)) for b in batch])
            y = np.stack([labels_to_three_class(b[1])[None] for b in batch])
            xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            loss = loss_fn(net(xt), yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        ap = evaluate_unet(net, val, device)
        log.info("epoch %d loss %.4f val AP50 %.3f", epoch, loss.item(), ap)
        if ap > best["ap50"]:
            best = {"ap50": ap, "epoch": epoch}
            Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), cfg.out)
    return best


def evaluate_unet(net, val: Sequence[Pair], device: str) -> float:
    import torch
    from monai.inferers import sliding_window_inference

    net.eval()
    scores = []
    with torch.no_grad():
        for img, gt in val:
            x = torch.from_numpy(percentile_normalize(img, axis=(1, 2)))[None].to(device)
            logits = sliding_window_inference(x, (256, 256), 4, net, overlap=0.25)
            probs = torch.softmax(logits, 1)[0].cpu().numpy()
            scores.append(average_precision(gt, three_class_to_labels(probs), thresholds=(0.5,))[0])
    return float(np.mean(scores))


def finetune_sam_decoder(
    train: Sequence[Pair],
    checkpoint: str,
    model_type: str = "vit_b",
    epochs: int = 20,
    lr: float = 1e-4,
    prompts_per_image: int = 32,
    out: str = "models/sam_decoder_ft.pt",
    seed: int = 0,
) -> str:
    """Fine-tune SAM's mask decoder with box prompts derived from ground-truth instances.

    The image encoder and prompt encoder stay frozen; image embeddings are cached
    once per image, which makes a full fine-tune feasible on a single GPU.
    """
    import torch
    import torch.nn.functional as F
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    from skimage.measure import regionprops

    from phenoscope.segmentation.sam_seg import SAMSegmenter

    rng = random.Random(seed)
    device = resolve_device(get_settings().device)
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    for p in [*sam.image_encoder.parameters(), *sam.prompt_encoder.parameters()]:
        p.requires_grad_(False)
    opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr)
    resize = ResizeLongestSide(sam.image_encoder.img_size)

    cache = []
    with torch.no_grad():
        for img, lab in train:
            rgb = SAMSegmenter.to_rgb(img)
            inp = resize.apply_image(rgb)
            t = torch.as_tensor(inp, device=device).permute(2, 0, 1)[None].float()
            emb = sam.image_encoder(sam.preprocess(t))
            props = [r for r in regionprops(lab) if r.area >= 20]
            cache.append((emb, lab, props, rgb.shape[:2]))

    for epoch in range(epochs):
        total = 0.0
        for emb, lab, props, shape in cache:
            if not props:
                continue
            sel = rng.sample(props, min(prompts_per_image, len(props)))
            boxes = np.array([[r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2]] for r in sel], float)
            boxes_t = torch.as_tensor(resize.apply_boxes(boxes, shape), device=device).float()
            gt = torch.as_tensor(np.stack([lab == r.label for r in sel]), device=device).float()

            sparse, dense = sam.prompt_encoder(points=None, boxes=boxes_t, masks=None)
            low_res, _ = sam.mask_decoder(
                image_embeddings=emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            masks = sam.postprocess_masks(low_res, resize.get_preprocess_shape(*shape, 1024), shape)
            logits = masks[:, 0]
            probs = torch.sigmoid(logits)
            dice = 1 - (2 * (probs * gt).sum((1, 2)) + 1) / (probs.sum((1, 2)) + gt.sum((1, 2)) + 1)
            loss = F.binary_cross_entropy_with_logits(logits, gt) + dice.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        log.info("SAM decoder epoch %d loss %.4f", epoch, total / max(1, len(cache)))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sam.mask_decoder.state_dict(), out)
    return out
