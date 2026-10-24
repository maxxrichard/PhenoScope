// Minimal web viewer over the PhenoScope REST API (no build step).
const $ = (sel) => document.querySelector(sel);
const api = async (path, opts = {}) => {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
};

let current = null;

async function init() {
  try {
    const h = await api("/health");
    $("#health").textContent = `v${h.version} · ${h.status}`;
    $("#health").classList.toggle("ok", h.status === "ok");
  } catch {
    $("#health").textContent = "offline";
  }
  const seg = await api("/segmenters");
  $("#segmenter").innerHTML = seg.available
    .map((s) => `<option ${s === seg.default ? "selected" : ""}>${s}</option>`)
    .join("");
  const datasets = await api("/datasets");
  $("#datasets").innerHTML = "";
  for (const d of datasets) {
    const li = document.createElement("li");
    li.textContent = `${d.name} (${d.n_images})`;
    li.onclick = () => selectDataset(d, li);
    $("#datasets").appendChild(li);
  }
}

async function selectDataset(d, li) {
  document.querySelectorAll("#datasets li").forEach((x) => x.classList.remove("active"));
  li.classList.add("active");
  const images = await api(`/datasets/${d.id}/images`);
  $("#images").innerHTML = "";
  for (const im of images) {
    const el = document.createElement("li");
    el.textContent = im.well ? `${im.plate ?? ""} ${im.well} · ${im.name}` : im.name;
    el.onclick = () => selectImage(im, el);
    $("#images").appendChild(el);
  }
}

async function selectImage(im, el) {
  document.querySelectorAll("#images li").forEach((x) => x.classList.remove("active"));
  el.classList.add("active");
  current = im;
  $("#channels").value = (im.channels.names || []).slice(0, 3).join(",");
  await refresh();
}

async function refresh() {
  if (!current) return;
  const runs = await api(`/images/${current.id}/segmentations`);
  const params = new URLSearchParams();
  if ($("#channels").value) params.set("channels", $("#channels").value);
  if ($("#outlines").checked && runs.length) params.set("labels", runs.at(-1).label_name);
  const img = document.createElement("img");
  img.alt = `Composite preview of ${current.name}`;
  img.src = `/images/${current.id}/render.png?${params}`;
  $("#canvas").replaceChildren(img);
  $("#status").textContent = `${current.name} · ${current.width}×${current.height} px · ${current.pixel_size_um} µm/px`;
  $("#runs tbody").innerHTML = runs
    .map((r) => `<tr><td>#${r.id}</td><td>${r.model}</td><td>${r.n_objects}</td><td>${r.runtime_s.toFixed(2)} s</td></tr>`)
    .join("");
}

$("#run").onclick = async () => {
  if (!current) return;
  $("#status").textContent = "Segmenting…";
  try {
    const r = await api(`/images/${current.id}/segment`, {
      method: "POST",
      body: JSON.stringify({ segmenter: $("#segmenter").value }),
    });
    $("#status").textContent = `${r.n_objects} objects in ${r.runtime_s.toFixed(2)} s`;
    await refresh();
  } catch (e) {
    $("#status").textContent = e.message;
  }
};
$("#outlines").onchange = refresh;
$("#channels").onchange = refresh;

init();
