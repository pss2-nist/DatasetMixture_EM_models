# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import argparse
import random
from shutil import copyfile

"""
This class will split image collections of images and their corresponding masks into
four folders:
train_images
train_masks
test_images
test_masks
according to the split fraction

"""


def split(image_folder, mask_folder, train_image_folder, train_mask_folder, test_image_folder, test_mask_folder,
          fraction):
    # TODO: Test
    image_folder = os.path.abspath(image_folder)
    mask_folder = os.path.abspath(mask_folder)
    img_files = [f for f in os.listdir(mask_folder)]
    random.shuffle(img_files)
    idx = int(fraction * len(img_files))
    train_img_files = img_files[0:idx]
    test_img_files = img_files[idx:]
    if not os.path.exists(train_image_folder):
        os.mkdir(train_image_folder)
    if not os.path.exists(train_mask_folder):
        os.mkdir(train_mask_folder)
    if not os.path.exists(test_image_folder):
        os.mkdir(test_image_folder)
    if not os.path.exists(test_mask_folder):
        os.mkdir(test_mask_folder)
    # print(train_img_files, test_img_files)
    for fn in train_img_files:
        basename, ext = os.path.splitext(fn)
        file = os.path.join(image_folder, fn)
        if ext.lower() in [".tif", ".tiff"]:
            for mask_ext in [".tif", ".tiff"]:
                mask_file = os.path.join(mask_folder, f"{basename}{mask_ext}")
                print(file, os.path.isfile(file))
                print(mask_file, os.path.isfile(mask_file))
                if os.path.isfile(file) and os.path.isfile(mask_file):
                    copyfile(file, "{}/{}".format(train_image_folder, f"{basename}.tif"))
                    copyfile(mask_file, "{}/{}".format(train_mask_folder, f"{basename}.tif"))
                    # break

    for fn in test_img_files:
        basename, ext = os.path.splitext(fn)
        file = os.path.join(image_folder, fn)
        if ext.lower() in [".tif", ".tiff"]:
            for mask_ext in [".tif", ".tiff"]:
                mask_file = os.path.join(mask_folder, f"{basename}{mask_ext}")
                print(file, os.path.isfile(file))
                print(mask_file, os.path.isfile(mask_file))
                if os.path.isfile(file) and os.path.isfile(mask_file):
                    copyfile(file, "{}/{}".format(test_image_folder, f"{basename}.tif"))
                    copyfile(mask_file, "{}/{}".format(test_mask_folder, f"{basename}.tif"))


def main():
    parser = argparse.ArgumentParser(prog='split', description='Script that splits data')
    parser.add_argument('-i', '--image_dir', type=str, help="full path of image folder")  #
    parser.add_argument('-m', '--mask_dir', type=str, help="full path of mask folder")
    parser.add_argument('-f', '--fraction', type=float, help="fraction in train")
    parser.add_argument('-tri', '--trainImageDir', type=str, help="full path of train image folder destination")
    parser.add_argument('-tei', '--testImageDir', type=str, help="full path of test image folder destination")
    parser.add_argument('-trm', '--trainMaskDir', type=str, help="full path of train mask folder destination")
    parser.add_argument('-tem', '--testMaskDir', type=str, help="full path of test mask folder destination")
    args, unknown = parser.parse_known_args()

    if args.image_dir is None:
        print('ERROR: missing input image folder ')
        return
    # print(args)
    # print(unknown)
    split(args.image_dir, args.mask_dir, args.trainImageDir, args.trainMaskDir,
          args.testImageDir, args.testMaskDir, args.fraction)


if __name__ == "__main__":
    main()
