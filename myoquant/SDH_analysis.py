import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import tensorflow as tf
from rich.progress import track
from skimage.measure import regionprops_table

from .gradcam import *
from .random_brightness import *

labels_predict = ["control", "sick"]
tf.random.set_seed(42)
np.random.seed(42)


def predict_single_cell(single_cell_img, _model_SDH):
    img_array = np.empty((1, 256, 256, 3))
    img_array[0] = tf.image.resize(single_cell_img, (256, 256))
    prediction = _model_SDH.predict(img_array)
    predicted_class = prediction.argmax()
    predicted_proba = round(np.amax(prediction), 2)
    heatmap = make_gradcam_heatmap(
        img_array, _model_SDH.get_layer("resnet50v2"), "conv5_block3_3_conv"
    )
    grad_cam_img = save_and_display_gradcam(img_array[0], heatmap)
    return grad_cam_img, predicted_class, predicted_proba


def resize_batch_cells(histo_img, cellpose_df):
    img_array_full = np.empty((len(cellpose_df), 256, 256, 3))
    for index in track(range(len(cellpose_df)), description="Resizing cells"):
        single_cell_img = histo_img[
            cellpose_df.iloc[index, 5] : cellpose_df.iloc[index, 7],
            cellpose_df.iloc[index, 6] : cellpose_df.iloc[index, 8],
        ].copy()

        single_cell_mask = cellpose_df.iloc[index, 9].copy()
        single_cell_img[~single_cell_mask] = 0

        img_array_full[index] = tf.image.resize(single_cell_img, (256, 256))
    return img_array_full


def predict_all_cells(histo_img, cellpose_df, _model_SDH):
    predicted_class_array = np.empty((len(cellpose_df)))
    predicted_proba_array = np.empty((len(cellpose_df)))
    img_array_full = resize_batch_cells(histo_img, cellpose_df)
    prediction = _model_SDH.predict(img_array_full)
    index_counter = 0
    for prediction_result in prediction:
        predicted_class_array[index_counter] = prediction_result.argmax()
        predicted_proba_array[index_counter] = np.amax(prediction_result)
        index_counter += 1
    return predicted_class_array, predicted_proba_array


def paint_full_image(image_sdh, df_cellpose, class_predicted_all):
    image_sdh_paint = np.zeros((image_sdh.shape[0], image_sdh.shape[1]), dtype=np.uint8)
    for index in track(range(len(df_cellpose)), description="Painting cells"):
        single_cell_mask = df_cellpose.iloc[index, 9].copy()
        if class_predicted_all[index] == 0:
            image_sdh_paint[
                df_cellpose.iloc[index, 5] : df_cellpose.iloc[index, 7],
                df_cellpose.iloc[index, 6] : df_cellpose.iloc[index, 8],
            ][single_cell_mask] = 1
        elif class_predicted_all[index] == 1:
            image_sdh_paint[
                df_cellpose.iloc[index, 5] : df_cellpose.iloc[index, 7],
                df_cellpose.iloc[index, 6] : df_cellpose.iloc[index, 8],
            ][single_cell_mask] = 2
    return image_sdh_paint


def run_sdh_analysis(image_array, model_SDH, mask_cellpose):
    props_cellpose = regionprops_table(
        mask_cellpose,
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
    df_cellpose = pd.DataFrame(props_cellpose)
    class_predicted_all, proba_predicted_all = predict_all_cells(
        image_array, df_cellpose, model_SDH
    )

    count_per_label = np.unique(class_predicted_all, return_counts=True)
    class_and_proba_df = pd.DataFrame(
        list(zip(class_predicted_all, proba_predicted_all)),
        columns=["class", "proba"],
    )

    # Result table dict
    headers = ["Feature", "Raw Count", "Proportion (%)"]
    data = []
    data.append(["Muscle Fibers", len(class_predicted_all), 100])
    for elem in count_per_label[0]:
        data.append(
            [
                labels_predict[int(elem)],
                count_per_label[1][int(elem)],
                100 * count_per_label[1][int(elem)] / len(class_predicted_all),
            ]
        )
    result_df = pd.DataFrame(columns=headers, data=data)
    # Paint The Full Image
    full_label_map = paint_full_image(image_array, df_cellpose, class_predicted_all)
    return result_df, full_label_map
