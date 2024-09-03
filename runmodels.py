import subprocess
from preprocess.tiling import *
from preprocess.split import *
from pytorch_models.SEM_train import *


def run_many_models(data, learning_rates=None, are_pretrained=None, name_dataset="SEM-GoldNP"):
    if learning_rates is None:
        learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    if are_pretrained is None:
        are_pretrained = [True, False]
    # print("$PWD")
    # source conda.sh
    # output = subprocess.run('''python -c "import sys; print(sys.executable)"
    #                   conda activate datadrivenem
    #                   python -c "import sys; print(sys.executable)"
    #                   conda env export''', executable='/bin/bash', shell=True, capture_output=True)
    # print(bytes.decode(output.stdout))
    # subprocess.call(['CALL', 'conda.bat', 'activate', 'datadrivenem'])
    counter = 1
    num_classes = 255
    model_name = "Deeplab50"

    for l, lr in enumerate(learning_rates):
        for p, is_pt in enumerate(are_pretrained):
            model_filename = f"deeplab50_{name_dataset}_lr{lr}_pt{is_pt}.pt"
            mn = f"deeplab50_metrics_{name_dataset}_lr{lr}_pt{is_pt}.csv"
            run_onemodel(data=data, learningRate=lr, modelName=model_filename, metricsfile=mn, model_name=model_name,
                         pretrained=is_pt, num_classes=num_classes)


def run_onemodel(data, learningRate, modelName, metricsfile, model_name, pretrained, num_classes):
    # mn # metrics file name
    print(f"number of classes:{num_classes}")

    # data="/home/pnb/trainingData/"{name_dataset}
    train_images = 'train_images'
    train_masks = 'train_masks'
    test_images = "test_images"
    test_masks = "test_masks"
    suffix = "pytorchOutputMtoM_INFER_final"
    output_dir = f"{data}/{suffix}"
    devicetype = "gpu"
    # devicetype = "cpu"
    epochs = 100
    batchsize = 200

    print(f"number of epochs: {epochs}")

    print(f"About to start running {metricsfile}")
    if model_name != "unet":
        print('pytorch model')
        print('input parameters:')
        print(
            f'--data={data} --train_images={train_images} --train_masks={train_masks}'
            f'--test_images={test_images} --test_masks={test_masks}')
        print(f'--output_dir={output_dir} --epochs={epochs} --model_filename={modelName} '
              f'--device_name={devicetype} --batch_size={batchsize}')
        print(f'--learning_rate={learningRate} --metrics_name={metricsfile} --model_name={model_name} '
              f'--pretrained={pretrained} --classes={num_classes}')
        args_to_parse = ['--data', f'{data}', '--trainImages', f'{train_images}',
                         '--trainMasks', f'{train_masks}', '--testImages', f'{test_images}',
                         '--testMasks', f'{test_masks}', '--outputDir', f'{output_dir}', '--epochs', f'{epochs}',
                         f'--modelName', f'{model_name}', '--devicetype', f'{devicetype}', f'--batchsize',
                         f'{batchsize}', f'--learningRate', f'{learningRate}', '--metricsfile', f'{metricsfile}',
                         '--pretrained', f'{pretrained}', '--classes', f'{num_classes}']
        call_main(args_to_parse)
        # subprocess.call(
        #     ['python', 'pytorch_models/INFER_train.py', '--data', f'{data}', '--train_images', f'{train_images}',
        #      '--train_masks', f'{train_masks}', '--output_dir', f'{output_dir}', '--epochs', f'{epochs}',
        #      f'--model_filename', f'{model_filename}', '--device_name', f'{device_name}', f'--batch_size',
        #      '{batch_size}', f'--learning_rate', f'{learning_rate}', '--metrics_name', f'{mn}', '--model_name',
        #      f'{model_name}', '--pretrained', f'{pretrained}', '--classes', f'{num_classes}'])


# !/bin/bash
#
# SBATCH --job-name=INFER_odommppre
# SBATCH --partition=heimdall
#
# SBATCH --cpus-per-task=128
# SBATCH --gres=gpu:4
#
def tile_and_split(root_folder, image_input_folder, mask_input_folder, xPieces, yPieces):
    print(f"root_folder: {root_folder}")
    print(f"image_input_folder:{image_input_folder}")
    print(f"mask_input_folder: {mask_input_folder}")
    print(f"xPieces: {xPieces}")
    print(f"yPieces: {yPieces}")

    image_dir = os.path.join(root_folder, image_input_folder)
    mask_dir = os.path.join(root_folder, mask_input_folder)
    tiledImageDir = os.path.join(root_folder, "tiled_images")
    tiledMaskDir = os.path.join(root_folder, "tiled_masks")
    trainImageDir = os.path.join(root_folder, "train_images")
    trainMaskDir = os.path.join(root_folder, "train_masks")
    testImageDir = os.path.join(root_folder, "test_images")
    testMaskDir = os.path.join(root_folder, "test_masks")
    fraction = 0.8

    tile(image_dir, tiledImageDir, xPieces, yPieces)
    tile(mask_dir, tiledMaskDir, xPieces, yPieces)
    split(tiledImageDir, tiledMaskDir, trainImageDir, trainMaskDir, testImageDir, testMaskDir, fraction)


if __name__ == "__main__":
    run_many_models()
