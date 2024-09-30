import os
import subprocess
from preprocess.tiling import *
from preprocess.split import *
from pytorch_models.SEM_train import *
from preprocess.stitching import *
from pytorch_models.SEM_inference import *


# from pytorch_models.inference import *


def infer_and_stitch(modelFilepath, imageDirpath, maskDirpath, maskNumClasses, outputDirpath, stitchdirpath):
    # os.makedirs(outputDirpath, exist_ok=True)
    # os.makedirs(stitchdirpath, exist_ok=True)
    for file in os.listdir(modelFilepath):
        filepath = os.path.join(modelFilepath, file)
        fbasename, ext = os.path.splitext(filepath)
        if ext == '.pt':
            output_maindir = os.path.join(maskDirpath, fbasename)
            output_dir = os.path.join(output_maindir, outputDirpath)
            stitchdir_path = os.path.join(output_maindir, stitchdirpath)
            print(f"filepath : {filepath}\n"
                  f"modelFilepath: {modelFilepath}\n"
                  f"imageDirpath: {imageDirpath}\n"
                  f"maskDirpath: {maskDirpath}\n"
                  f"maskNumClasses: {maskNumClasses}\n"
                  f"output_dir: {output_dir}\n"
                  f"stitchdir_path: {stitchdir_path}")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(stitchdir_path, exist_ok=True)
            inference_withmask(filepath, imageDirpath, maskDirpath, maskNumClasses, output_dir, use_avgs=True)
            stitch(output_dir, stitchdir_path)


def main():
    pure_ddspath = "/mnt/isgnas/home/pss2/data/EM-Vladar/DatadrivenEM/Synthetic_DDS"
    pure_pbspath = "/mnt/isgnas/home/pss2/data/EM-Vladar/DatadrivenEM/Synthetic_PBS"
    measured_path = "/mnt/isgnas/home/pss2/data/EM-Vladar/DatadrivenEM/measured"

    output_ddspath = "inference_dds_images"
    output_pbspath = "inference_pbs_images"
    output_measuredpath = "inference_measured"

    stitch_ddspath = "stitch_dds"
    stitch_pbspath = "stitch_pbs"
    stitch_measuredpath = "stitch_measured"
    model_path = ""
    modelstring = 'pytorchOutputMtoM'
    allpaths = [pure_ddspath, pure_pbspath, measured_path]
    all_tiled_out = [output_ddspath, output_pbspath, output_measuredpath]
    all_stitched = [stitch_ddspath, stitch_pbspath, stitch_measuredpath]
    combined = "/mnt/isgnas/home/pss2/data/EM-Vladar/DatadrivenEM/Combined"
    combined_sub = [os.path.join(combined, s) for s in os.listdir(combined) if os.path.isdir(os.path.join(combined, s))]
    for subfolder in combined_sub:
        model_Filepath = os.path.join(subfolder, modelstring)
        # tiled image directory
        for root, out, stitch_folder in zip(allpaths, all_tiled_out, all_stitched):
            print(f"STARTING : {root},\t{out},\t{stitch_folder}")
            image_Dirpath = os.path.join(root, "tiled_images")
            mask_Dirpath = os.path.join(root, "tiled_masks")
            maskNum_Classes = 256
            # output_Dirpath = os.path.join(model_Filepath, out)  # output - tiled prediction
            # stitch_Dirpath = os.path.join(model_Filepath, stitch_path)  # final stitched images
            infer_and_stitch(model_Filepath, image_Dirpath, mask_Dirpath, maskNum_Classes, out, stitch_folder)


if __name__ == "__main__":
    main()
