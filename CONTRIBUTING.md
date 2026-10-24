# Contributing

1. Create a branch from `main`.
2. `make install`, then run `make lint test` before pushing.
3. New segmenters: subclass `phenoscope.segmentation.base.Segmenter`, decorate it with `@register_segmenter("name")`, import heavy dependencies lazily, and add an entry to `configs/models.yaml`.
4. New features or metrics need a test that runs on synthetic data (`phenoscope.benchmarks.synthetic`) on CPU.
5. Changes to the benchmark sets or the drift reference need a note in `CHANGELOG.md`.
