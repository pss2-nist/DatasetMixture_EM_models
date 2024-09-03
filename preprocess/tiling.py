# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import numpy as np
from PIL import Image
import os
import argparse
import tifffile
import json

"""
This class will tile each image in the input folder into tiles saved in the output folder.

The arguments refer to the number of cuts along x- and y-axes (xPieces and yPieces). 
Thus, an input image of size 1024 x 1024 will be cut into tiles of size 512 x 512 with arguments
--xPieces 2 --yPieces 2

TODO: add tile overlap option
TODO: modify the imcrop function in such a way that all pixels are used. In other words, 
currently tile size is a floor of dimensions (height = imgheight // yPieces) which implies that some rows/cols might not be included

"""


def count_int_digits(n):
    """
    Given an integer, returns the number of digits in the number.
    If a non integer is provided, it is automatically converted to an integer.
    """
    count = len(str(int(n)))
    print('The number of digits in the number:', n, ' are:', count)
    return count


def imgcrop(input, xPieces, yPieces, img_name, output_dir):
    im = Image.open(input)
    print(f"shape:{im.size}")
    img_name, file_extension = os.path.splitext(img_name)
    imgwidth, imgheight = im.size
    height = imgheight // yPieces
    width = imgwidth // xPieces

    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            a = im.crop(box)
            a.save(output_dir + "/" + img_name + "_" + str(i) + "-" + str(j) + file_extension)
            # TODO this change works here but has to be propagated to the stitching.py code


def imgcrop_tomo(input, xPieces, yPieces, zPieces, img_name, output_dir):
    im = tifffile.tifffile.imread(input)
    # im2 = Image.open(input)
    print("tomography image shape ", im.shape)
    # print("Image.open: ", im2.size)
    img_name, file_extension = os.path.splitext(img_name)
    dims = im.ndim
    if dims == 3:
        imglength, imgwidth, imgheight = im.shape  # assuming ZXY
        height = imgheight // yPieces
        width = imgwidth // xPieces
        length = imglength // zPieces

        for i in range(0, yPieces):
            for j in range(0, xPieces):
                for k in range(0, zPieces):
                    # print(output_dir + "/" + img_name + "_" + str(k) + "-" + str(i) + "-" + str(j) +
                    #       file_extension)
                    a = im[k * length:(k + 1) * length, j * width: (j + 1) * width, i * height:(i + 1) * height]
                    tifffile.tifffile.imwrite(output_dir + "/" + img_name + "_" + str(k) + "-" + str(i) + "-" + str(j) +
                                              file_extension, data=a)
    elif dims == 4:
        xis, imglength, imgwidth, imgheight = im.shape  # assuming ZXY
        height = imgheight // yPieces
        width = imgwidth // xPieces
        length = imglength // zPieces

        for i in range(0, yPieces):
            for j in range(0, xPieces):
                for k in range(0, zPieces):
                    # print(output_dir + "/" + img_name + "_" + str(k) + "-" + str(i) + "-" + str(j) +
                    #       file_extension)
                    a = im[:, k * length:(k + 1) * length, j * width: (j + 1) * width, i * height:(i + 1) * height]
                    tifffile.tifffile.imwrite(output_dir + "/" + img_name + "_" + str(k) + "-" + str(i) + "-" + str(j) +
                                              file_extension, data=a, imagej=True)


def tile(image_dir_or_file, output_dir, xPieces, yPieces, zPieces=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_array = []
    filename_array = []
    is_dir = os.path.isdir(image_dir_or_file)
    if is_dir:
        for filename in os.listdir(image_dir_or_file):
            filepath = os.path.join(image_dir_or_file, filename)
            if filename.split('.')[-1] in ["tif", "tiff"]:
                file_array.append(filepath)
                filename_array.append(filename)
        ##########################################################################
        i = 0
        for file in file_array:
            if zPieces is None:
                imgcrop(file, xPieces, yPieces, filename_array[i], output_dir)
            else:
                imgcrop_tomo(file, xPieces, yPieces, zPieces, filename_array[i], output_dir)
            i += 1
        ##########################################################################
    else:
        assert image_dir_or_file.split('.')[-1] in ["tif", "tiff"], f"no tiff file found: {image_dir_or_file}"
        image_name = os.path.basename(image_dir_or_file)
        if zPieces is None:
            imgcrop(image_dir_or_file, xPieces, yPieces, image_name, output_dir)
        else:
            imgcrop_tomo(image_dir_or_file, xPieces, yPieces, zPieces, image_name, output_dir)


def pad_image_to_fit_tiles(image, tile_size=(200, 200), padding_value=-1):
    img_array = np.array(image)
    imgwidth, imgheight = img_array.shape[:2]
    # extra modulo for perfect tilesizes
    pad_height = (tile_size[1] - (imgheight % tile_size[1])) % tile_size[1]
    pad_width = (tile_size[0] - (imgheight % tile_size[0])) % tile_size[0]
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    # padding using 0 values
    padded_img = np.pad(img_array, (pad_top, pad_bottom), (pad_left, pad_right), (0, 0), mode='constant',
                        constant_values=padding_value)

    return Image.fromarray(padded_img), (pad_left, pad_right), (pad_top, pad_bottom)


def fixed_crop(file, tile_dims, img_name, output_dir):
    if tile_dims is None:
        tile_dims = (200, 200)
    img = Image.open(file)
    padded_image, pad_width, pad_height = pad_image_to_fit_tiles(img, tile_dims)
    img_name, file_extension = os.path.splitext(img_name)
    im_arr = np.array(padded_image)
    print(f"shape:{padded_image.size}")
    imgwidth, imgheight = padded_image.size
    os.makedirs(output_dir, exist_ok=True)
    height, width = tile_dims
    # yPieces = imgheight // height
    x_tiles = int(np.ceil(imgwidth / tile_dims[0]))
    y_tiles = int(np.ceil(imgheight / tile_dims[0]))
    tiles_info = []
    for i in range(y_tiles):
        for j in range(x_tiles):
            left = j * width
            upper = i * height
            right = left + width
            lower = upper + height
            box = (left, upper, right, lower)
            tile = padded_image.crop(box)
            tiles_info.append({
                "filename": file,
                "position": (left, upper),
                "size": (right - left, lower - upper)
            })
    metadata = {
        "padded_size": (imgwidth, imgheight),
        "original_size": img.size,
        "padding": {
            "width": pad_width,
            "height": pad_height
        },
        "tile_size": tile_dims,
        "tiles_info": tiles_info
    }
    return metadata


def fixed_tile(image_dir_or_file, output_dir, tile_dims=(200, 200), metadata_file="tiling_metadata.json"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_array = []
    filename_array = []
    is_dir = os.path.isdir(image_dir_or_file)
    tiling_metadata = {}
    if is_dir:
        for filename in os.listdir(image_dir_or_file):
            filepath = os.path.join(image_dir_or_file, filename)
            if filename.split('.')[-1] in ["tif", "tiff"]:
                file_array.append(filepath)
                filename_array.append(filename)
        ##########################################################################
        i = 0
        for file in file_array:
            file_data = fixed_crop(file, tile_dims, filename_array[i], output_dir)
            tiling_metadata[file] = file_data
            i += 1
        ##########################################################################
    else:
        assert image_dir_or_file.split('.')[-1] in ["tif", "tiff"], "no tiff file found"
        image_name = os.path.basename(image_dir_or_file)
        file_data = fixed_crop(image_dir_or_file, tile_dims, image_name, output_dir)
        tiling_metadata[image_dir_or_file] = file_data
    with open(os.path.join(output_dir, metadata_file), 'w') as f:
        json.dump(tiling_metadata, f, indent=4)



def main():
    parser = argparse.ArgumentParser(prog='tiling', description='Script that tiles data')
    parser.add_argument('-i', '--image_dir', type=str, help='folder path to input images')
    parser.add_argument('-o', '--output_dir', type=str, help='folder path to saving output tiles')
    parser.add_argument('-x', '--xPieces', type=int, help='number of image cuts along x-axis')
    parser.add_argument('-y', '--yPieces', type=int, help='number of image cuts along y-axis')
    parser.add_argument('-z', '--zPieces', type=int, help='number of image cuts along y-axis')
    args, unknown = parser.parse_known_args()

    if args.image_dir is None:
        print('ERROR: missing input image dir ')
        return

    tile(args.image_dir, args.output_dir, args.xPieces, args.yPieces, args.zPieces)


if __name__ == "__main__":
    main()
