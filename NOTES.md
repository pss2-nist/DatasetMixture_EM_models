# DatasetMixture_EM_models: Project Walkthrough & Guide

This document serves as a step-by-step guide and tutorial for running and extending this repository. Each stage is modular and can be run independently or as part of a full pipeline.

---

## 1. Data Preparation

- **Input:** Two or more datasets, named according to the project convention.
- **Supported formats:** TIFF (`.tif`, `.tiff`), and other common image formats.
- **Action:** Place raw datasets in the `data/raw/` directory.

---

## 2. Preprocessing

- **Tiling:** Use `preprocess/tiling.py` to split large images into tiles.
    - Automatically records tile positions for later stitching.
    - Logs all parameters and outputs for provenance.
- **Inverse (Stitching):** Use `preprocess/one_dataset_stitch.sh` or equivalent to reconstruct original images from tiles.
- **Tip:** All preprocessing steps save metadata for reproducibility.

---

## 3. Dataset Mixture Generation

- **Mixing:** Run `preprocess/stratified_mix.py` to generate mixtures of datasets.
    - Specify mixture ratios and source datasets.
    - Output includes mixture metadata for traceability.

---

## 4. Hyperparameter Search

- **Script:** Use `HyperparameterSearch.py` for grid or Bayesian search.
    - Supports 5-fold cross-validation.
    - Example parameters: learning rate, batch size, model architecture.
    - All tried parameters and results are saved (CSV/JSON).

---

## 5. Model Training

- **Script:** `runmodels.py`
    - Supports data augmentation and multiple model architectures.
    - Optionally include published models.
    - Implements early stopping.
    - Logs all metrics, configs, and random seeds for reproducibility.

---

## 6. Postprocessing & Metrics

- **Stitching:** Reconstruct full-size predictions from tiles.
- **Metrics Assimilation:** Use scripts in `graph_and_fit/` to aggregate and analyze metrics.
    - Supports recursive assimilation for complex experiments.

---

## 7. Visualization

- **Graphing:** Use scripts in `graph_and_fit/` to visualize:
    - Training curves
    - Hyperparameter effects
    - Dataset mixture impacts
- **Outputs:** All plots and summary tables are saved in `results/`.

---

## 8. Automation & Reproducibility

- **Workflow:** Consider using a workflow manager (e.g., Snakemake, Makefile) to automate the full pipeline.
- **Environment:** Use `environment.yml` to set up a reproducible environment.
- **Logging:** All steps log parameters and outputs for full provenance.

---

## 9. Extending the Project

- Add new preprocessing or augmentation methods as new scripts/modules.
- Integrate additional models by extending `runmodels.py`.
- Add new metrics or visualizations in `graph_and_fit/`.

---

## 10. Troubleshooting & Tips

- Check logs in the `logs/` directory for errors or warnings.
- Use unit tests (see `tests/`) to verify code changes.
- For help, refer to in-code docstrings and comments.

---

**This guide will be updated as the project evolves. Contributions and suggestions are welcome!**