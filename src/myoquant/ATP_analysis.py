import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from myoquant.common_func import extract_single_image, df_from_cellpose_mask
import numpy as np
import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt

labels_predict = {1: "fiber type 1", 2: "fiber type 2"}
np.random.seed(42)


def get_all_intensity(
    image_array, df_cellpose, intensity_method="median", erosion=None
):
    all_cell_median_intensity = []
    for index in range(len(df_cellpose)):
        single_cell_img = extract_single_image(image_array, df_cellpose, index, erosion)

        # Calculate median pixel intensity of the cell but ignore 0 values
        if intensity_method == "median":
            single_cell_median_intensity = np.median(
                single_cell_img[single_cell_img > 0]
            )
        elif intensity_method == "mean":
            single_cell_median_intensity = np.mean(single_cell_img[single_cell_img > 0])
        all_cell_median_intensity.append(single_cell_median_intensity)
    return all_cell_median_intensity


def estimate_threshold(intensity_list, n_classes=2):
    density = gaussian_kde(intensity_list)
    density.covariance_factor = lambda: 0.05
    density._compute_covariance()

    # Create a vector of 256 values going from 0 to 256:
    xs = np.linspace(0, 255, 256)
    density_xs_values = density(xs)
    gmm = GaussianMixture(n_components=n_classes).fit(
        np.array(intensity_list).reshape(-1, 1)
    )

    # Find the x values of the two peaks
    peaks_x = np.sort(gmm.means_.flatten())
    # Find the minimum point between the two peaks

    threshold_list = []
    length = len(peaks_x)
    for index, peaks in enumerate(peaks_x):
        if index == length - 1:
            break
        min_index = np.argmin(
            density_xs_values[(xs > peaks) & (xs < peaks_x[index + 1])]
        )
        threshold_list.append(peaks + xs[min_index])
    return threshold_list


def plot_density(all_cell_median_intensity, intensity_threshold, n_classes=2):
    if intensity_threshold == 0:
        intensity_threshold = estimate_threshold(all_cell_median_intensity, n_classes)
    fig, ax = plt.subplots(figsize=(10, 5))
    density = gaussian_kde(all_cell_median_intensity)
    density.covariance_factor = lambda: 0.1
    density._compute_covariance()

    # Create a vector of 256 values going from 0 to 25
    xs = np.linspace(0, 255, 256)
    density_xs_values = density(xs)
    ax.hist(
        all_cell_median_intensity, bins=255, density=True, alpha=0.5, label="Histogram"
    )
    ax.plot(xs, density_xs_values, label="Estimated Density", linewidth=3)
    if isinstance(intensity_threshold, int):
        intensity_threshold = [intensity_threshold]
    for values in intensity_threshold:
        ax.axvline(x=values, color="red", label="Threshold")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.legend()
    return fig


def merge_peaks_too_close(peak_list):
    pass


def classify_cells_intensity(all_cell_median_intensity, intensity_threshold):
    muscle_fiber_type_all = []
    for intensity in all_cell_median_intensity:
        if isinstance(intensity_threshold, int):
            intensity_threshold = [intensity_threshold]
        class_cell = np.searchsorted(intensity_threshold, intensity, side="right")
        muscle_fiber_type_all.append(class_cell)
    return muscle_fiber_type_all


def predict_all_cells(
    histo_img,
    cellpose_df,
    intensity_threshold,
    n_classes=2,
    intensity_method="median",
    erosion=None,
):
    all_cell_median_intensity = get_all_intensity(
        histo_img, cellpose_df, intensity_method, erosion
    )
    if intensity_threshold is None:
        intensity_threshold = estimate_threshold(all_cell_median_intensity, n_classes)

    muscle_fiber_type_all = classify_cells_intensity(
        all_cell_median_intensity, intensity_threshold
    )
    return muscle_fiber_type_all, all_cell_median_intensity, intensity_threshold


def paint_full_image(image_atp, df_cellpose, class_predicted_all):
    image_atp_paint = np.zeros(
        (image_atp.shape[0], image_atp.shape[1]), dtype=np.uint16
    )
    # for index in track(range(len(df_cellpose)), description="Painting cells"):
    for index in range(len(df_cellpose)):
        single_cell_mask = df_cellpose.iloc[index, 9].copy()
        image_atp_paint[
            df_cellpose.iloc[index, 5] : df_cellpose.iloc[index, 7],
            df_cellpose.iloc[index, 6] : df_cellpose.iloc[index, 8],
        ][single_cell_mask] = (
            class_predicted_all[index] + 1
        )
    return image_atp_paint


def label_list_from_threhsold(threshold_list):
    label_list = []
    length = len(threshold_list)
    for index, threshold in enumerate(threshold_list):
        if index == 0:
            label_list.append(f"<{threshold}")
        if index == length - 1:
            label_list.append(f">{threshold}")
        else:
            label_list.append(f">{threshold} & <{threshold_list[index+1]}")
    return label_list


def run_atp_analysis(
    image_array,
    mask_cellpose,
    intensity_threshold=None,
    n_classes=2,
    intensity_method="median",
    erosion=None,
):
    df_cellpose = df_from_cellpose_mask(mask_cellpose)
    class_predicted_all, intensity_all, intensity_threshold = predict_all_cells(
        image_array,
        df_cellpose,
        intensity_threshold,
        n_classes,
        intensity_method,
        erosion,
    )
    fig = plot_density(intensity_all, intensity_threshold, n_classes)
    df_cellpose["muscle_cell_type"] = class_predicted_all
    df_cellpose["cell_intensity"] = intensity_all
    count_per_label = np.unique(class_predicted_all, return_counts=True)

    # Result table dict
    headers = ["Feature", "Raw Count", "Proportion (%)"]
    data = []
    data.append(["Muscle Fibers", len(class_predicted_all), 100])
    label_list = label_list_from_threhsold(intensity_threshold)
    for index, elem in enumerate(count_per_label[0]):
        data.append(
            [
                label_list[int(elem)],
                count_per_label[1][int(index)],
                100 * count_per_label[1][int(index)] / len(class_predicted_all),
            ]
        )
    result_df = pd.DataFrame(columns=headers, data=data)
    # Paint The Full Image
    full_label_map = paint_full_image(image_array, df_cellpose, class_predicted_all)
    return result_df, full_label_map, df_cellpose, fig
