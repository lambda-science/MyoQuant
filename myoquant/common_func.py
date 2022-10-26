import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import torch
from cellpose.models import Cellpose
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from tensorflow import keras
from tensorflow.config import list_physical_devices

from .gradcam import *
from .random_brightness import *

tf.random.set_seed(42)
np.random.seed(42)

if len(list_physical_devices("GPU")) >= 1:
    use_GPU = True
    tf.config.experimental.set_memory_growth(list_physical_devices("GPU")[0], True)

else:
    use_GPU = False


def is_gpu_availiable():
    return use_GPU


def load_cellpose():
    model_c = Cellpose(gpu=use_GPU, model_type="cyto2")
    return model_c


def load_stardist():
    model_s = StarDist2D.from_pretrained("2D_versatile_he")
    return model_s


def load_sdh_model(model_path):
    model_sdh = keras.models.load_model(
        model_path, custom_objects={"RandomBrightness": RandomBrightness}
    )
    return model_sdh


@torch.no_grad()
def run_cellpose(image, model_cellpose, diameter=None):
    channel = [[0, 0]]
    with torch.no_grad():
        mask_cellpose, _, _, _ = model_cellpose.eval(
            image, batch_size=1, diameter=diameter, channels=channel
        )
        return mask_cellpose


def run_stardist(image, model_stardist, nms_thresh=0.4, prob_thresh=0.5):
    img_norm = image / 255
    img_norm = normalize(img_norm, 1, 99.8)
    mask_stardist, _ = model_stardist.predict_instances(
        img_norm, nms_thresh=nms_thresh, prob_thresh=prob_thresh
    )
    return mask_stardist
