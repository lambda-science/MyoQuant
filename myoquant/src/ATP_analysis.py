import pandas as pd
from skimage.measure import regionprops_table
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

import numpy as np

labels_predict = {1: "fiber type 1", 2: "fiber type 2"}
np.random.seed(42)


def get_all_intensity(image_array, df_cellpose):
    all_cell_median_intensity = []
    for index in range(len(df_cellpose)):
        single_cell_img = image_array[
            df_cellpose.iloc[index, 5] : df_cellpose.iloc[index, 7],
            df_cellpose.iloc[index, 6] : df_cellpose.iloc[index, 8],
        ].copy()

        single_cell_mask = df_cellpose.iloc[index, 9].copy()
        single_cell_img[~single_cell_mask] = 0
        # Calculate median pixel intensity of the cell but ignore 0 values
        single_cell_median_intensity = np.median(single_cell_img[single_cell_img > 0])
        all_cell_median_intensity.append(single_cell_median_intensity)
    return all_cell_median_intensity


def estimate_threshold(intensity_list):
    density = gaussian_kde(intensity_list)
    density.covariance_factor = lambda: 0.25
    density._compute_covariance()

    # Create a vector of 256 values going from 0 to 256:
    xs = np.linspace(0, 255, 256)
    density_xs_values = density(xs)
    gmm = GaussianMixture(n_components=2).fit(np.array(intensity_list).reshape(-1, 1))

    # Find the x values of the two peaks
    peaks_x = np.sort(gmm.means_.flatten())
    # Find the minimum point between the two peaks
    min_index = np.argmin(density_xs_values[(xs > peaks_x[0]) & (xs < peaks_x[1])])
    threshold = peaks_x[0] + xs[min_index]

    return threshold


def predict_all_cells(histo_img, cellpose_df, intensity_threshold):
    all_cell_median_intensity = get_all_intensity(histo_img, cellpose_df)
    if intensity_threshold is None:
        intensity_threshold = estimate_threshold(all_cell_median_intensity)

    muscle_fiber_type_all = [
        1 if x > intensity_threshold else 2 for x in all_cell_median_intensity
    ]
    return muscle_fiber_type_all, all_cell_median_intensity


def paint_full_image(image_atp, df_cellpose, class_predicted_all):
    image_atp_paint = np.zeros(
        (image_atp.shape[0], image_atp.shape[1]), dtype=np.uint16
    )
    # for index in track(range(len(df_cellpose)), description="Painting cells"):
    for index in range(len(df_cellpose)):
        single_cell_mask = df_cellpose.iloc[index, 9].copy()
        if class_predicted_all[index] == 1:
            image_atp_paint[
                df_cellpose.iloc[index, 5] : df_cellpose.iloc[index, 7],
                df_cellpose.iloc[index, 6] : df_cellpose.iloc[index, 8],
            ][single_cell_mask] = 1
        elif class_predicted_all[index] == 2:
            image_atp_paint[
                df_cellpose.iloc[index, 5] : df_cellpose.iloc[index, 7],
                df_cellpose.iloc[index, 6] : df_cellpose.iloc[index, 8],
            ][single_cell_mask] = 2
    return image_atp_paint


def run_atp_analysis(image_array, mask_cellpose, intensity_threshold=None):
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
    class_predicted_all, intensity_all = predict_all_cells(
        image_array, df_cellpose, intensity_threshold
    )
    df_cellpose["muscle_cell_type"] = class_predicted_all
    df_cellpose["cell_intensity"] = intensity_all
    count_per_label = np.unique(class_predicted_all, return_counts=True)

    # Result table dict
    headers = ["Feature", "Raw Count", "Proportion (%)"]
    data = []
    data.append(["Muscle Fibers", len(class_predicted_all), 100])
    for index, elem in enumerate(count_per_label[0]):
        data.append(
            [
                labels_predict[int(elem)],
                count_per_label[1][int(index)],
                100 * count_per_label[1][int(index)] / len(class_predicted_all),
            ]
        )
    result_df = pd.DataFrame(columns=headers, data=data)
    # Paint The Full Image
    full_label_map = paint_full_image(image_array, df_cellpose, class_predicted_all)
    return result_df, full_label_map, df_cellpose
