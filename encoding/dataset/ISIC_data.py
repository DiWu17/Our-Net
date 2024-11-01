import os
import numpy as np
from PIL import Image
import torch
from PIL.Image import Resampling
from torch.utils.data import Dataset
import random

class ISIC2017Dataset(Dataset):
    NUM_CLASS = 2
    def __init__(self, mode, root="F:/Datasets/2DSegmentation/ISIC2017", transform=None, mask_transform=None, augment=False, img_size=(256, 256), whole_image=False):
        """
        Args:
            mode (str): Mode of the dataset, either 'train', 'val', or 'test'.
            root (str): Root directory of the dataset.
            transform (callable, optional): A function/transform to apply to the images.
            mask_transform (callable, optional): A function/transform to apply to the masks.
            augment (bool): Whether to apply data augmentation.
            img_size (tuple): The target size of the images (width, height).
            whole_image (bool): Whether to use the whole image without cropping.
        """
        self.mode = mode
        self.root = root
        self.transform = transform
        self.mask_transform = mask_transform
        self.augment = augment
        self.img_size = img_size
        self.whole_image = whole_image

        # Define directories for images and masks
        self.image_dir = os.path.join(root, mode, 'images')
        self.mask_dir = os.path.join(root, mode, 'masks')

        # Get list of image and mask file names
        self.image_list = sorted(os.listdir(self.image_dir))
        self.mask_list = sorted(os.listdir(self.mask_dir))

        assert len(self.image_list) == len(self.mask_list), "Mismatch between images and masks count"

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_list[index])
        mask_path = os.path.join(self.mask_dir, self.mask_list[index])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        origin_img = image.resize(self.img_size, resample=Resampling.BILINEAR)

        # Resize image and mask to target size
        image = image.resize(self.img_size, Resampling.BILINEAR)
        mask = mask.resize(self.img_size, Resampling.NEAREST)

        # Apply augmentation if needed
        if self.augment:
            image, mask = self.augmentation(image, mask, self.whole_image)

        # Convert to numpy arrays
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        mask = np.array(mask, dtype=np.uint8)

        # Ensure the mask is binary (0 or 1)
        mask[mask != 0] = 1

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        # Convert to torch tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        mask = torch.from_numpy(mask).long()

        return image, mask, torch.from_numpy(np.array(origin_img))

    @property
    def num_class(self):
        return self.NUM_CLASS

    def augmentation(self, img, mask, whole_image):
        """
        Apply data augmentation to the image and mask.
        Args:
            img (PIL.Image): Input image.
            mask (PIL.Image): Corresponding mask.
            whole_image (bool): Whether to use the whole image.
        Returns:
            img, mask: Augmented image and mask.
        """
        if whole_image:
            # Example: Apply random horizontal flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # Additional augmentations can be added here if needed
            pass
        return img, mask

    def _mask_transform(self, mask):
        """
        Apply transformations to the mask.
        Args:
            mask (numpy.ndarray): Input mask.
        Returns:
            torch.Tensor: Transformed mask.
        """
        mask[mask != 0] = 1
        return torch.from_numpy(mask).long()

if __name__ == "__main__":
    from torchvision import transforms

    # Define transformations for the images
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    # Initialize the dataset
    dataset = ISIC2017Dataset(mode='train', root='F:/Datasets/2DSegmentation/ISIC2017', transform=input_transform, augment=True)

    # Get a sample from the dataset
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
