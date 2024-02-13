import cv2
from albumentations import (
    Compose,
    FancyPCA,
    GaussianBlur,
    GaussNoise,
    HorizontalFlip,
    HueSaturationValue,
    ImageCompression,
    OneOf,
    PadIfNeeded,
    RandomBrightnessContrast,
    Resize,
    ShiftScaleRotate,
    ToGray,
    ImageOnlyTransform,
)
import numpy as np
import pilgram
from PIL import Image


def train_augmentations(height: int, width: int):
    augmentations = Compose(
        [
            ImageCompression(quality_lower=35, quality_upper=100, p=0.7),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            OneOf(
                [
                    Resize(height=height, width=width, interpolation=cv2.INTER_AREA),
                    Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
                ],
                p=1,
            ),
            PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
            ToGray(p=0.2),
            ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
        ]
    )
    return augmentations


def val_augmentations(height: int, width: int):
    augmentations = Compose(
        [
            Resize(height=height, width=width, interpolation=cv2.INTER_AREA),
            PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
        ]
    )

    return augmentations


def test_augmentations(height: int, width: int):
    augmentations = Compose(
        [
            ImageCompression(quality_lower=35, quality_upper=100, p=0.8),
            InstagramFilterAugmentation(p=0.8),
            OneOf(
                [
                    Resize(height=height, width=width, interpolation=cv2.INTER_AREA),
                    Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
                ],
                p=1,
            ),
            PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.3),
        ]
    )
    return augmentations


class InstagramFilterAugmentation(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        insta_filters = [
            "_1977",
            "aden",
            "brannan",
            "brooklyn",
            "clarendon",
            "earlybird",
            "gingham",
            "hudson",
            "inkwell",
            "kelvin",
            "lark",
            "lofi",
            "maven",
            "mayfair",
            "moon",
            "nashville",
            "perpetua",
            "reyes",
            "rise",
            "slumber",
            "stinson",
            "toaster",
            "valencia",
            "walden",
            "willow",
            "xpro2",
        ]

        image = Image.fromarray(img)

        filter_to_apply = np.random.choice(insta_filters)

        image = pilgram.__dict__[filter_to_apply](image)
        im_arr = np.array(image)

        return im_arr
