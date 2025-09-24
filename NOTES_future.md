# DatasetMixture_EM_models: Future Workflow & Guide

This document describes the **future, improved workflow** for this repository. The goal is to consolidate the current codebase so that the entire process—from raw data to results—requires only 1-3 user actions (scripts), with all steps automated, reproducible, and well-documented.

---

## Overview

The pipeline will be streamlined into a small number of scripts, each handling a major stage of the workflow. Users will only need to:

1. **Prepare data** (place files in the correct folder).
2. **Run the main pipeline script** (or up to three scripts for advanced/custom workflows).
3. **Review results** (automatically organized and visualized).

---

## Step 1: Prepare Your Data

- Place your datasets (e.g., `.tif`, `.tiff`, or other supported formats) in the `data/raw/` directory.
- Use the required naming convention (e.g., `dataset1.tif`, `dataset2.tif`).

---

## Step 2: Run the Pipeline

**Option A: One-Step Full Pipeline**

- Run the master script:
  ```sh
  python run_pipeline.py
  ```
  This script will:
  - Preprocess and tile datasets (with provenance logging).
  - Generate dataset mixtures as specified in `config.yaml`.
  - Perform hyperparameter search (grid or Bayesian, with CV).
  - Train models (with augmentation, early stopping, and metrics logging).
  - Postprocess, stitch outputs, and aggregate metrics.
  - Generate all graphs and visualizations.

**Option B: Modular Steps (Advanced/Debugging)**

- Run individual scripts for each stage:
  1. `python preprocess_and_mix.py`  
     (Preprocessing, tiling, and dataset mixing)
  2. `python train_and_evaluate.py`  
     (Hyperparameter search, model training, evaluation)
  3. `python postprocess_and_visualize.py`  
     (Stitching, metrics assimilation, visualization)

---

## Step 3: Review Results

- All results, logs, and visualizations are saved in the `results/` directory.
- Each run is documented with configuration, parameters, and provenance for reproducibility.

---

## Future Improvements Reflected Here

- **Single config file** (`config.yaml`) controls all pipeline options.
- **Automatic logging** of all parameters, metrics, and outputs.
- **Clear modularization**: Each script can be run independently or as part of the full pipeline.
- **Extensible**: Add new models, metrics, or visualizations with minimal changes.
- **Unit tests and CI**: Ensure reliability as the codebase evolves.

---

## Next Steps for Codebase Refactoring

- Consolidate preprocessing, mixing, and provenance into one script/module.
- Merge training, hyperparameter search, and evaluation into a unified script.
- Automate postprocessing and visualization.
- Ensure all scripts read from a shared config and write logs/results in a standardized format.
- Add comprehensive documentation and tests.

---

**This document will guide future development and serve as the main user tutorial once the improved workflow is implemented.**
