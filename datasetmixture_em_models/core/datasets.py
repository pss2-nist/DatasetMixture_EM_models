"""Base classes for segmentation and classification datasets."""

from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union
import torch
import numpy as np
from torchvision.datasets import VisionDataset
from skimage import io


def _get_image_from_path(image_path: Union[str, Path], astype: Optional[type] = np.float32) -> np.ndarray:
    img = io.imread(str(image_path))
    if astype is not None:
        img = img.astype(astype)
    return img


def return_images_from_paths(image_paths: Union[str, Path, List], astype: Optional[type] = np.float32) -> np.ndarray:
    if isinstance(image_paths, (str, Path)):
        return _get_image_from_path(image_paths, astype)
    else:
        assert isinstance(image_paths, (np.ndarray, list))
        image_list = []
        for image_path in image_paths:
            img = _get_image_from_path(image_path, astype)
            image_list.append(img)
        return np.asarray(image_list)


def zscore_normalize(image: np.ndarray) -> np.ndarray:
    """Apply z-score normalization to image."""
    return (image - image.mean()) / (image.std() + 1e-8)


class BaseSegmentationDataset(VisionDataset):
    """Base class for segmentation datasets from filesystem."""

    def __init__(self, root: str, train_image_folder: str, train_mask_folder: str, test_image_folder: str, test_mask_folder: str,
                 subset: Optional[str] = None, fraction: Optional[float] = None, seed: int = 123, n_classes: int = 2, transforms: Optional[Any] = None,):
        """
        Initialize dataset.

        Args:
            root: Root directory containing train/test folders
            train_image_folder: Folder name for training images
            train_mask_folder: Folder name for training masks
            test_image_folder: Folder name for test images
            test_mask_folder: Folder name for test masks
            subset: 'train' or 'test' split
            fraction: Fraction of data to use (0-1)
            seed: Random seed for reproducibility
            n_classes: Number of classes
            transforms: Torchvision transforms
        """
        super().__init__(root=root, transforms=transforms)
        self.root = Path(root)
        self.subset = subset or "train"
        self.fraction = fraction
        self.seed = seed
        self.n_classes = n_classes

        # Select folders based on subset
        if self.subset == "train":
            image_folder = self.root / train_image_folder
            mask_folder = self.root / train_mask_folder
        else:  # test
            image_folder = self.root / test_image_folder
            mask_folder = self.root / test_mask_folder

        # Glob and sort image/mask pairs
        self.image_paths = sorted(image_folder.glob("*"))
        self.mask_paths = sorted(mask_folder.glob("*"))

        # Handle fraction sampling
        if self.fraction is not None and self.subset == "train":
            n_samples = int(len(self.image_paths) * self.fraction)
            rng = np.random.RandomState(self.seed)
            indices = rng.choice(len(self.image_paths), n_samples, replace=False)
            self.image_paths = [self.image_paths[i] for i in sorted(indices)]
            self.mask_paths = [self.mask_paths[i] for i in sorted(indices)]

        assert len(self.image_paths) == len(self.mask_paths), "Image/mask mismatch"

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load image/mask pair.

        Returns dict with keys 'image' and 'mask' by default.
        Subclasses can override to change return format.
        """
        image = _get_image_from_path(str(self.image_paths[idx]))
        mask = _get_image_from_path(str(self.mask_paths[idx]))

        # image = torch.from_numpy(image).float()
        # mask = torch.from_numpy(mask).long()

        if self.transforms:
            image = self.transforms(image)

        return {"image": image, "mask": mask} # using dict format for flexibility in subclasses

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights from training masks."""
        if self.subset != "train":
            raise ValueError("Class weights only computed for train subset")

        class_counts = np.zeros(self.n_classes)
        for mask_path in self.mask_paths:
            mask = io.imread(str(mask_path))
            unique, counts = np.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                if u < self.n_classes:
                    class_counts[u] += c

        # Compute weights (inverse frequency)
        total = class_counts.sum()
        weights = total / (class_counts + 1e-8)
        weights = torch.from_numpy(weights / weights.sum()).float()
        return weights
