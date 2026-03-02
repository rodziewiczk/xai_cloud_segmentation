# XAISegmentation — Spectral Attribution for Cloud Segmentation (Sentinel-2)

Research code for **explainability (XAI) in multispectral semantic segmentation** on Sentinel-2 imagery, using the CloudSEN12 / CloudSEN12+ cloud segmentation task.

This repository includes:
- an **experiment** for running multiple attribution methods on segmentation models,
- **quantitative evaluation** of explanations (faithfulness / sensitivity / complexity),
- **local vs global ROI analysis** (single connected component vs all pixels of a class),
- **dataset-level spectral band importance** analysis,
- an **XAI-guided input dimensionality reduction** experiment that reduces **13 → 7 bands** using a reconstruction adapter.

> Notes
> - This is research/replication code (not a polished library).
> - Notebooks were developed for Colab/Drive; paths may require adjustment.

## Main findings (thesis summary)

- Attribution methods were compared numerically under a single evaluation protocol (faithfulness, sensitivity, complexity), enabling trade-off analysis across methods.
- Spectral attributions computed on a *local ROI* (one connected component) are typically close to those computed on the *global ROI* (all pixels of the same predicted class) within the same scene, suggesting explanations are stable with respect to ROI granularity.
- Dataset-level aggregation shows that **10 m Sentinel-2 bands dominate** attribution for cloud segmentation, in particular **{B2, B3, B4, B8}** consistently form the core informative set.
- Using band-importance ranking to keep the most informative bands and reconstruct the missing ones with a lightweight adapter enables a **13→7 band reduction** with **~≤1% relative degradation** (worst case) compared to the full 13-band input.