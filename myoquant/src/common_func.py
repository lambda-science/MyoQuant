"""
Module that contains function common to both nuclei (HE/Fluo) and mitochondria analysis (SDH)
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import math

import torch
import tensorflow as tf

from cellpose.models import Cellpose
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from tensorflow import keras
import numpy as np
from PIL import Image
from skimage.measure import regionprops_table
from skimage.morphology import binary_erosion
import pandas as pd

# from .gradcam import make_gradcam_heatmap, save_and_display_gradcam
from .random_brightness import RandomBrightness

import imageio

tf.random.set_seed(42)
np.random.seed(42)

if len(tf.config.list_physical_devices("GPU")) >= 1:
    use_GPU = True
    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices("GPU")[0], True
    )

else:
    use_GPU = False


class HiddenPrints:
    """
    Class to hide the print of some function during the CLI analysis
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def is_gpu_availiable():
    """
    Check if the GPU is available for the neural network inference
    """
    return use_GPU


def load_cellpose():
    """
    Load and return the cellpose model
    """
    model_c = Cellpose(gpu=use_GPU, model_type="cyto2")
    return model_c


def load_stardist(fluo=False):
    """Load and return the stardist model in the right mode (fluo or HE)

    Args:
        fluo (bool, optional): If true return Fluo StarDist Model. Defaults to False.

    Returns:
        StarDist Model instance: the stardist model
    """
    if fluo:
        model_s = StarDist2D.from_pretrained("2D_versatile_fluo")
    else:
        model_s = StarDist2D.from_pretrained("2D_versatile_he")
    return model_s


def load_sdh_model(model_path: str):
    """Load and return the SDH model

    Args:
        model_path (str): path to the SDH model

    Returns:
        Keras model instance: tensorflow keras model
    """
    model_sdh = keras.models.load_model(
        model_path, custom_objects={"RandomBrightness": RandomBrightness}
    )
    return model_sdh


@torch.no_grad()
def run_cellpose(image, model_cellpose, diameter=None):
    """Run Cellpose analysis on input image

    Args:
        image (nd.array): Numpy array of the image
        model_cellpose (PyTorch Model): Cellpose model object loaded with load_cellpose()
        diameter (int, optional): Cell diameter estimation for cellpose. Defaults to None.

    Returns:
        nd.array: Label map (mask) of the cellpose prediction
    """
    channel = [[0, 0]]
    with torch.no_grad():
        mask_cellpose, _, _, _ = model_cellpose.eval(
            image, diameter=diameter, channels=channel
        )
        return mask_cellpose


def run_stardist(image, model_stardist, nms_thresh=0.4, prob_thresh=0.5):
    """Run StarDist analysis on input image

    Args:
        image (nd.array): Numpy array of the image
        model_stardist (Stardist Model): Stardist model object loaded with load_stardist()
        nms_thresh (float, optional): NMS Threshold for Stardist. Defaults to 0.4.
        prob_thresh (float, optional): Probability Threshold for Stardist. Defaults to 0.5.

    Returns:
        nd.array: Label map (mask) of the stardist prediction
    """
    img_norm = image / 255
    img_norm = normalize(img_norm, 1, 99.8)
    mask_stardist, _ = model_stardist.predict_instances(
        img_norm, nms_thresh=nms_thresh, prob_thresh=prob_thresh
    )
    return mask_stardist


def df_from_cellpose_mask(mask):
    props_cellpose = regionprops_table(
        mask,
        properties=[
            "label",
            "area",
            "centroid",
            "eccentricity",
            "bbox",
            "image",
            "perimeter",
            "feret_diameter_max",
        ],
    )
    df_cellpose = pd.DataFrame(props_cellpose)
    return df_cellpose


def df_from_stardist_mask(mask, intensity_image=None):
    props_stardist = regionprops_table(
        mask,
        intensity_image=intensity_image,
        properties=[
            "label",
            "area",
            "centroid",
            "eccentricity",
            "bbox",
            "image",
            "perimeter",
        ],
    )
    df_stardist = pd.DataFrame(props_stardist)
    return df_stardist


def extract_single_image(raw_image, df_props, index, erosion=None):
    single_entity_img = raw_image[
        df_props.iloc[index, 5] : df_props.iloc[index, 7],
        df_props.iloc[index, 6] : df_props.iloc[index, 8],
    ].copy()
    surface_area = df_props.iloc[index, 1]
    cell_radius = math.sqrt(surface_area / math.pi)
    single_entity_mask = df_props.iloc[index, 9].copy()
    if erosion is not None:
        erosion_size = cell_radius * (
            erosion / 100
        )  # Erosion in percentage of the cell radius
        for i in range(int(erosion_size)):
            single_entity_mask = binary_erosion(
                single_entity_mask, out=single_entity_mask
            )

    single_entity_img[~single_entity_mask] = 0
    return single_entity_img


def label2rgb(img_ndarray, label_map):
    """Convert a label map to RGB image

    Args:
        img_ndarray (nd.array): Numpy array of the image
        label_map (nd.array): Numpy array of the label map (mask)

    Returns:
        nd.array: RGB image of the label map with 3 colors (white, green, red) for 3 classes (background, control, abnormal)
    """
    label_to_color = {
        0: [255, 255, 255],
        1: [15, 157, 88],
        2: [219, 68, 55],
        3: [100, 128, 170],
        4: [231, 204, 143],
        5: [202, 137, 115],
        6: [178, 143, 172],
        7: [144, 191, 207],
        8: [148, 187, 187],
        9: [78, 86, 105],
        10: [245, 90, 87],
    }
    img_rgb = np.zeros((img_ndarray.shape[0], img_ndarray.shape[1], 3), dtype=np.uint8)

    for gray, rgb in label_to_color.items():
        img_rgb[label_map == gray, :] = rgb
    return img_rgb


def blend_image_with_label(image, label_rgb, fluo=False):
    """Blend the image with the label map together

    Args:
        image (nd.array): Numpy array of the image
        label_rgb (nd.array): Numpy array of the label map (mask)
        fluo (bool, optional): Indicate if the image is fluo (single channel). Defaults to False.

    Returns:
        PIL Image: An overlay image of the image and the label map with a 0.5 alpha
    """
    image = Image.fromarray(image)
    label_rgb = Image.fromarray(label_rgb)
    if fluo:
        image = image.convert("RGB")
    return Image.blend(image, label_rgb, 0.5)
