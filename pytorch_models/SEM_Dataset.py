import torch.utils.data
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import skimage
import skimage.io
from sklearn.utils.class_weight import compute_class_weight


def _get_image_from_path(image_path, astype=np.float32):
    _img = imread(image_path)
    if astype is not None:
        _img = _img.astype(astype)
    return _img


def return_images_from_paths(image_paths, astype=np.float32):
    """

    :param astype: convert image format. (Not recommended as this can cause many errors down the line)
    :param image_paths:
    :return:
    """

    if isinstance(image_paths, str):
        return _get_image_from_path(image_paths)
    else:
        # print(image_paths)
        assert isinstance(image_paths, (np.ndarray, list))
        image_list = []
        for image_path in image_paths:
            img = _get_image_from_path(image_path, astype)
            image_list.append(img)
        image_list = np.asarray(image_list)
        return image_list


class SEMDataset(VisionDataset):
    def __init__(self, root, train_image_folder, train_mask_folder, test_image_folder, test_mask_folder,
                 fraction, masks, transforms=None, target_transform=None, subset=None, n_classes=4):
        super(SEMDataset, self).__init__(root, transforms=transforms, target_transform=target_transform)
        self.weights = None
        # self.data_paths = data_paths
        self.masks = masks
        self.transforms = transforms
        # self.fraction = fraction
        # self.train_image_names = sorted(train_image_folder_path.glob("*"))
        # self.train_mask_names = sorted(train_mask_folder_path.glob("*"))
        # self.test_image_names = sorted(test_image_folder_path.glob("*"))
        # self.test_mask_names = sorted(test_mask_folder_path.glob("*"))
        if subset == "Train":
            weights = [0] * n_classes
            self.image_names = self.train_image_names
            self.mask_names = self.train_mask_names
            for mask_name in self.mask_names:
                image = np.asarray(Image.open(mask_name))
                for i in range(len(weights)):
                    weights[i] += np.count_nonzero(image == i)
        # if subset == "Train".casefold():
        #     weights = [0] * n_classes
        #     self.image_names = self.

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path = self.data_paths[index]
        mask_path = self.masks[index]
        image = _get_image_from_path(image_path)
        mask = _get_image_from_path(mask_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def get_class_weights(self):
        weights = compute_class_weight('balanced', classes=np.unique(self.masks), y=self.masks)
        self.weights = torch.tensor(weights, dtype=torch.float)


class GetDataloader:
    def __init__(self, data_dir, train_image_folder, train_mask_folder, test_image_folder, test_mask_folder, fraction,
                 batch_size, n_classes):
        self.data_dir = data_dir
        self.train_image_folder = train_image_folder
        self.train_mask_folder = train_mask_folder
        self.test_image_folder = test_image_folder
        self.test_mask_folder = test_mask_folder
        self.fraction = fraction
        self.batch_size = batch_size
        self.n_classes = n_classes
        # data_transforms = transforms.Compose([transforms.ToTensor(),
        #                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        data_transforms = transforms.Compose([transforms.ToTensor()])
        image_datasets = {
            x: SEMDataset(data_dir,
                          train_image_folder=train_image_folder,
                          train_mask_folder=train_mask_folder,
                          test_image_folder=test_image_folder,
                          test_mask_folder=test_mask_folder,
                          fraction=fraction,
                          subset=x,
                          transforms=data_transforms,
                          n_classes=n_classes)
            for x in ['Train', 'Test']
        }
        self.weights = image_datasets['Train'].weights
        dataloaders = {
            x: DataLoader(image_datasets[x],
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=8, drop_last=True)
            for x in ['Train', 'Test']
        }
        self.dataloaders = dataloaders


if __name__ == "__main__":
    test_image = np.zeros((500, 50, 2, 2, 3, 6))
    f_path = "../Data/image_test/test1.tiff"
    test_metadata = {"test_metadata": "this is test metadata",
                     "hello": "hello, hello!"}

    print("hi", test_image.ndim)
