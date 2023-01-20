import numpy as np
import pandas as pd
from skimage.draw import line
from skimage.measure import regionprops_table

from .draw_line import *


def extract_ROIs(histo_img, index, cellpose_df, mask_stardist):
    single_cell_img = histo_img[
        cellpose_df.iloc[index, 5] : cellpose_df.iloc[index, 7],
        cellpose_df.iloc[index, 6] : cellpose_df.iloc[index, 8],
    ].copy()
    nucleus_single_cell_img = mask_stardist[
        cellpose_df.iloc[index, 5] : cellpose_df.iloc[index, 7],
        cellpose_df.iloc[index, 6] : cellpose_df.iloc[index, 8],
    ].copy()
    single_cell_mask = cellpose_df.iloc[index, 9]
    single_cell_img[~single_cell_mask] = 0
    nucleus_single_cell_img[~single_cell_mask] = 0

    props_nuc_single = regionprops_table(
        nucleus_single_cell_img,
        intensity_image=single_cell_img,
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
    df_nuc_single = pd.DataFrame(props_nuc_single)
    return single_cell_img, nucleus_single_cell_img, single_cell_mask, df_nuc_single


def single_cell_analysis(
    single_cell_img,
    single_cell_mask,
    df_nuc_single,
    x_fiber,
    y_fiber,
    cell_label,
    internalised_threshold=0.75,
    draw_and_return=False,
):
    n_nuc, n_nuc_intern, n_nuc_periph = 0, 0, 0
    new_row_lst = []
    new_col_names = df_nuc_single.columns.tolist()
    new_col_names.append("internalised")
    new_col_names.append("score_eccentricity")
    new_col_names.append("cell label")
    for _, value in df_nuc_single.iterrows():
        n_nuc += 1
        value = value.tolist()
        # Extend line and find closest point

        # Handling of the case where the nucleus is at the exact center of the fiber
        if x_fiber == value[3] and y_fiber == value[2]:
            n_nuc_intern += 1
            continue

        m, b = line_equation(x_fiber, y_fiber, value[3], value[2])

        intersections_lst = calculate_intersection(
            m, b, (single_cell_img.shape[0], single_cell_img.shape[1])
        )
        border_point = calculate_closest_point(value[3], value[2], intersections_lst)
        rr, cc = line(
            int(y_fiber),
            int(x_fiber),
            int(border_point[1]),
            int(border_point[0]),
        )
        for index3, coords in enumerate(list(zip(rr, cc))):
            try:
                if single_cell_mask[coords] == 0:
                    dist_nuc_cent = calculate_distance(
                        x_fiber, y_fiber, value[3], value[2]
                    )
                    dist_out_of_fiber = calculate_distance(
                        x_fiber, y_fiber, coords[1], coords[0]
                    )
                    ratio_dist = dist_nuc_cent / dist_out_of_fiber
                    if ratio_dist <= internalised_threshold:
                        n_nuc_intern += 1
                        value.append(True)
                        value.append(ratio_dist)
                        value.append(cell_label)
                    else:
                        n_nuc_periph += 1
                        value.append(False)
                        value.append(ratio_dist)
                        value.append(cell_label)
                    new_row_lst.append(value)
                    break
            except IndexError:
                coords = list(zip(rr, cc))[index3 - 1]
                dist_nuc_cent = calculate_distance(x_fiber, y_fiber, value[3], value[2])
                dist_out_of_fiber = calculate_distance(
                    x_fiber, y_fiber, coords[1], coords[0]
                )
                ratio_dist = dist_nuc_cent / dist_out_of_fiber
                if ratio_dist < internalised_threshold:
                    n_nuc_intern += 1
                    value.append(True)
                    value.append(ratio_dist)
                    value.append(cell_label)
                else:
                    n_nuc_periph += 1
                    value.append(True)
                    value.append(ratio_dist)
                    value.append(cell_label)
                new_row_lst.append(value)
                break
    df_nuc_single_stats = pd.DataFrame(new_row_lst, columns=new_col_names)
    return n_nuc, n_nuc_intern, n_nuc_periph, df_nuc_single_stats


def predict_all_cells(
    histo_img, cellpose_df, mask_stardist, internalised_threshold=0.75
):
    list_n_nuc, list_n_nuc_intern, list_n_nuc_periph, list_nuc_df = [], [], [], []
    for index in range(len(cellpose_df)):
        (
            single_cell_img,
            _,
            single_cell_mask,
            df_nuc_single,
        ) = extract_ROIs(histo_img, index, cellpose_df, mask_stardist)
        x_fiber = cellpose_df.iloc[index, 3] - cellpose_df.iloc[index, 6]
        y_fiber = cellpose_df.iloc[index, 2] - cellpose_df.iloc[index, 5]
        n_nuc, n_nuc_intern, n_nuc_periph, df_nuc_single_stats = single_cell_analysis(
            single_cell_img,
            single_cell_mask,
            df_nuc_single,
            x_fiber,
            y_fiber,
            index + 1,
            internalised_threshold,
        )
        list_n_nuc.append(n_nuc)
        list_n_nuc_intern.append(n_nuc_intern)
        list_n_nuc_periph.append(n_nuc_periph)
        list_nuc_df.append(df_nuc_single_stats)
        df_nuc_analysis = pd.DataFrame(
            list(zip(list_n_nuc, list_n_nuc_intern, list_n_nuc_periph)),
            columns=["N° Nuc", "N° Nuc Intern", "N° Nuc Periph"],
        )
    all_nuc_df_stats = pd.concat(list_nuc_df, ignore_index=True)
    cellpose_df_stat = pd.concat([cellpose_df, df_nuc_analysis], axis=1)
    return cellpose_df_stat, all_nuc_df_stats


def paint_histo_img(histo_img, cellpose_df, prediction_df):
    paint_img = np.zeros((histo_img.shape[0], histo_img.shape[1]), dtype=np.uint16)
    for index in range(len(cellpose_df)):
        single_cell_mask = cellpose_df.iloc[index, 9].copy()
        if prediction_df["N° Nuc Intern"].iloc[index] == 0:
            paint_img[
                cellpose_df.iloc[index, 5] : cellpose_df.iloc[index, 7],
                cellpose_df.iloc[index, 6] : cellpose_df.iloc[index, 8],
            ][single_cell_mask] = 1
        elif prediction_df["N° Nuc Intern"].iloc[index] > 0:
            paint_img[
                cellpose_df.iloc[index, 5] : cellpose_df.iloc[index, 7],
                cellpose_df.iloc[index, 6] : cellpose_df.iloc[index, 8],
            ][single_cell_mask] = 2
    return paint_img


def run_he_analysis(image_ndarray, mask_cellpose, mask_stardist, eccentricity_thresh):
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
            "feret_diameter_max",
        ],
    )
    df_cellpose = pd.DataFrame(props_cellpose)
    df_nuc_analysis, all_nuc_df_stats = predict_all_cells(
        image_ndarray, df_cellpose, mask_stardist, eccentricity_thresh
    )

    # Result table dict
    headers = ["Feature", "Raw Count", "Proportion (%)"]
    data = []
    data.append(["N° Nuclei", df_nuc_analysis["N° Nuc"].sum(), 100])
    data.append(
        [
            "N° Intern. Nuclei",
            df_nuc_analysis["N° Nuc Intern"].sum(),
            100
            * df_nuc_analysis["N° Nuc Intern"].sum()
            / df_nuc_analysis["N° Nuc"].sum(),
        ]
    )
    data.append(
        [
            "N° Periph. Nuclei",
            df_nuc_analysis["N° Nuc Periph"].sum(),
            100
            * df_nuc_analysis["N° Nuc Periph"].sum()
            / df_nuc_analysis["N° Nuc"].sum(),
        ]
    )
    data.append(
        [
            "N° Cells with 1+ intern. nuc.",
            df_nuc_analysis["N° Nuc Intern"].astype(bool).sum(axis=0),
            100
            * df_nuc_analysis["N° Nuc Intern"].astype(bool).sum(axis=0)
            / len(df_nuc_analysis),
        ]
    )

    result_df = pd.DataFrame(columns=headers, data=data)
    label_map_he = paint_histo_img(image_ndarray, df_cellpose, df_nuc_analysis)
    return result_df, label_map_he, df_nuc_analysis, all_nuc_df_stats
