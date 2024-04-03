import numpy as np
import torch
from torchvision.transforms import Compose
from PIL import Image

from proj4_code.classification.data_transforms import get_fundamental_transforms, get_data_augmentation_transforms


def test_fundamental_transforms():
    """
    Tests the transforms using output from disk
    """

    transforms = get_fundamental_transforms(inp_size=(100, 50), pixel_mean=[0.5], pixel_std=[0.3])

    try:
        inp_img = Image.fromarray(np.loadtxt("proj4_code/proj4_unit_tests/classification/test_data/transform_inp.txt", dtype="uint8"))
        output_img = transforms(inp_img)
        expected_output = torch.load("proj4_code/proj4_unit_tests/classification/test_data/transform_out.pt")

    except:
        inp_img = Image.fromarray(
            np.loadtxt("../proj4_code/proj4_unit_tests/classification/test_data/transform_inp.txt", dtype="uint8")
        )
        output_img = transforms(inp_img)
        expected_output = torch.load("../proj4_code/proj4_unit_tests/classification/test_data/transform_out.pt")

    assert torch.allclose(expected_output, output_img)


def test_data_augmentation_transforms():
    """Tests the transforms by checking what functions are being used."""

    transforms_list = get_data_augmentation_transforms(inp_size=(100, 50), pixel_mean=[0.5], pixel_std=[0.3]).transforms

    assert len(transforms_list) > 3

    # last 3 should be fundamental
    augmentation_transforms = Compose(transforms_list[:-3])

    try:
        inp_img = Image.fromarray(np.loadtxt("proj4_code/proj4_unit_tests/classification/test_data/transform_inp.txt", dtype="uint8"))

    except:
        inp_img = Image.fromarray(
            np.loadtxt("../proj4_code/proj4_unit_tests/classification/test_data/transform_inp.txt", dtype="uint8")
        )
    augmented_img = augmentation_transforms(inp_img)
    assert isinstance(augmented_img, type(inp_img))
    assert not np.array_equal(augmented_img, inp_img)
