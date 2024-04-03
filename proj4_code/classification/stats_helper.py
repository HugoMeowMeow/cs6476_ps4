import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
    """
    Compute the mean and the standard deviation of the pixel values in the dataset.

    Note: convert the image in grayscale and then scale to [0,1] before computing
    mean and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    """

    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################
    # raise NotImplementedError('forward function of SimpleNet not implemented')

    pixel_values = []
    scaler = StandardScaler()
    files = glob.glob(dir_name + "/**/*", recursive=True)
    for file_name in glob.glob(dir_name + "/**/*", recursive=True):
        if os.path.isfile(file_name):
            with Image.open(file_name) as img:
                gray_img = img.convert('L')
                scaled_img = np.array(gray_img) / 255.0
                pixel_values.append(scaled_img.flatten())

    all_pixels = np.concatenate(pixel_values).reshape(-1, 1)
    scaler.fit(all_pixels)

    mean = scaler.mean_
    std = scaler.scale_

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
