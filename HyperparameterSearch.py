"""
Questions:
    1. Should the hyperparameter selection be done only once?
        a. How about hyperparameter search on both pure datasets?
        b. If the hyperparameters are different, should they be averaged or some other approach?
----------
Objective:
----------
        0. Modular process where steps can be partially skipped
        1. Start with two datasets with specified naming convention
        2. Preprocess and its inverse postprocess data.
            a. Automatically tile dataset and note how to stitch back
                i. Save this information for provenance.
        3. Grid or bayesian search parameters with 5-fold CV
            a. LR
            b. ??
        4. AI models
            a. Include augmentation?
            b. Add some published model?
            c. Record metrics while model runs.
            d. Early stopping approach,
        6. Postprocessing.
            a. Stitching - see 2.
            b. Metrics assimilation - if possible, recursive.
                i. Calculate loss and best loss
                ii. Calculate Macro and micro dice, etc across images.
            c. Training metrics
                i. Find the best loss
                ii. Find global and local stability.
                iii. Convergence speed
        7. Graph and fit.
            a. graph the data - visualize training
            b. relevant visualizations for hyperparameters
            c. Visualize effect of mixtures of dataset
        8. Recommend and save the best hyperparameters based on loss, local stability and convergence rate

"""
import argparse
import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from pytorch_models.SEM_train import *
from pytorch_models.SEM_Dataset import *
from runmodels import *


def generate_metadata_file():
    pass


# class SEMModelWrapper(BaseEstimator):
#     def __init__(self, modelName, fulldataset, classes, pretrained, batchsize, dataloader, criterion, criterion_test,
#                  optimizer, epochs,
#                  device, outputdir):
#         self.model = initializeModel(output_channels=classes, pretrained=pretrained, name=modelName, bs=batchsize)
#         self.dataloader = dataloader
#         self.fulldataset = fulldataset
#         self.criterion = criterion
#         self.criterion_test = criterion_test
#         self.epochs = epochs
#         self.optimizer = optimizer
#         self.device = device
#         self.outputdir = outputdir
#
#     def fit(self, X=None, y=None, weights=None):
#         train_indices = np.array(X[:, 0], dtype=int)
#         val_indices = np.array(X[:, 1], dtype=int)
#
#         train_subset = Subset(self.full_dataset, train_indices)
#         val_subset = Subset(self.full_dataset, val_indices)
#
#         if weights is not None:
#             sampler = torch.utils.data.WeightedRandomSampler(weights[train_indices], len(train_indices),
#                                                              replacement=True)
#             train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, shuffle=True)
#         else:
#             train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
#
#         # self.model.train()
#         train_model(self.model, self.criterion, self.criterion_test, self.dataloader, self.optimizer,
#                     bpath=self.outputdir, num_epochs=self.epochs, devicetype=self.device, metricsfile=self.metricsfile,
#                     mfn=self.modelWeights, modelName=self.modelName, pretrained=self.pretrained, lr=self.learningRate,
#                     bs=self.batchsize, classes=self.classes)
#         return self
#
#     def predict(self, X):
#         val_indices = np.array(X[:, 1], dtype=int)


def autotile(*datasets, output_dir, metadata_file_name="tiling_metadata.json"):
    from preprocess import tiling
    for dataset in datasets:
        tiling.fixed_tile(dataset, output_dir, tile_dims=(200, 200), metadata_file=metadata_file_name)
        # tiling.smart_tile(image_dir_or_file=dataset, output_dir)
    # save object for inverse transform


def stitch(metadata_file_name="tiling_metadata.json"):
    from preprocess import tiling
    tiling.stitch_reconstruct(metadata_file=metadata_file_name, save_path="stitched")


def get_stratified_kfold_CV_object():
    pass


def hyperparamsearch():
    pass


def run_gridsearch():
    # TODO: custom wrapper for gridsuearch
    pass


def create_parser():
    parser = argparse.ArgumentParser(prog='hyperparmsearch', description='Search for optimal hyperparameters')
    parser.add_argument('--dataset1', type=str, help="Path to dataset 1")
    parser.add_argument('--dataset2', type=str, help="Path to dataset 2")

    # args, unknown = parser.parse_known_args()
    return parser


def test_parser():
    parser = create_parser()
    path1 = "/mnt/isgnas/home/pss2/data/EM-Vladar/DatadrivenEM/Synthetic_DDS"
    # path1 = "C:/Users/pss2/PycharmProjects/DatadrivenEM/data/measured_SEM_images/Synthetic_DDS"
    path2 = "/mnt/isgnas/home/pss2//data/EM-Vladar/DatadrivenEM/Synthetic_PBS"
    # path2 = "C:/Users/pss2/PycharmProjects/DatadrivenEM/data/measured_SEM_images/Synthetic_PBS"
    paths = [path1, path2]
    # path_output = "C:/Users/pss2/PycharmProjects/DatadrivenEM/data/measured_SEM_images/DDmix_output"
    args = parser.parse_args(['--dataset1', path1, '--dataset2', path2])
    # if not os.path.exists(path_output):
    #     os.mkdir(path_output)
    datasets = args.dataset1, args.dataset2
    # tiling (and stitching) to obtain tiles. Generates metadata
    # tiling for images and masks need to have same names
    for path in paths:
        root_folder = path
        image_input_folder = "Generated/designed"
        mask_input_folder = "Mask/designed"
        xPieces = 5
        yPieces = 5
        # tile_and_split(root_folder, image_input_folder, mask_input_folder, xPieces, yPieces)
        run_many_models(data=root_folder, learning_rates=None, are_pretrained=None)
    # autotile(datasets, output_dir=path_output, metadata_file_name="tiling_metadata.json")
    # generate actual mixtures and metadata
    # generate_mixtures() - only for datasetmixture

    # stitch(metadata_file_name="tiling_metadata.json")
    # gridsearch parameters within each mixture


def main():
    pass


if __name__ == "__main__":
    test_parser()
    # main()
