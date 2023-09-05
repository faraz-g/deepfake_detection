import cv2
from albumentations import (Compose, FancyPCA, GaussianBlur, GaussNoise,
                            HorizontalFlip, HueSaturationValue,
                            ImageCompression, OneOf, PadIfNeeded,
                            RandomBrightnessContrast, Resize, ShiftScaleRotate,
                            ToGray)

def train_augmentations(height: int, width: int):
    augmentations = Compose([
        ImageCompression(quality_lower=35, quality_upper=100, p=0.7),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            Resize(height=height, width=width, interpolation=cv2.INTER_AREA),
            Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR)
            ],
            p=1
        ),
        PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ])
    return augmentations


def val_augmentations(height: int, width: int):
    augmentations = Compose([
        Resize(height=height, width=width, interpolation=cv2.INTER_AREA),
        PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
    ])

    return augmentations


def test_augmentations(height: int, width: int):
    augmentations = Compose([
        ImageCompression(quality_lower=35, quality_upper=100, p=0.8),
        GaussNoise(p=0.25),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            Resize(height=height, width=width, interpolation=cv2.INTER_AREA),
            Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR)
            ],
            p=1
        ),
        PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.3),
        ToGray(p=0.15),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ])
    return augmentations