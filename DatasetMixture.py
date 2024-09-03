"""
Use selected hyperparameters to run models on dataset mixture.
----------
Objective:
----------
        0. Modular process where steps can be partially skipped
        1. Start with two datasets with specified naming convention
        2. Preprocess and its inverse postprocess data.
            a. Automatically tile dataset and note how to stitch back
                i. Save this information for provenance.
            b.
        3. Generate specified dataset mixtures.
        4. Grid or bayesian search parameters with 5-fold CV
            a. LR
            b. ??
        5. AI models
            a. Include augmentation?
            b. Add some published model?
            c. Record metrics while model runs.
            d. Early stopping approach,
        6. Postprocessing.
            a. Stitching - see 2.
            b. Metrics assimilation - if possible, recursive.
        7. Graph and fit.
            a. graph the data - visualize training
            b. relevant visualizations for hyperparameters
            c. Visualize effect of mixtures of dataset

"""
import argparse


def main():
    parser = argparse.ArgumentParser(prog='hyperparmsearch', description='Search for optimal hyperparameters')
    parser.add_argument('--dataset1', type=str, description="Path to dataset 1")
    parser.add_argument('--dataset2', type=str, description="Path to dataset 2")

    args, unknown = parser.parse_known_args()


if __name__ == "__main__":
    main()
