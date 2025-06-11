import streamlit as st
from streamlit.components.v1 import html
import matplotlib
import requests
from io import BytesIO


try:
    from imageio.v2 import imread
except:
    from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from myoquant.common_func import (
    load_cellpose,
    run_cellpose,
    is_gpu_availiable,
    df_from_cellpose_mask,
)
from myoquant.ATP_analysis import (
    get_all_intensity,
    estimate_threshold,
    plot_density,
    predict_all_cells,
    paint_full_image,
)

labels_predict = {1: "fiber type 1", 2: "fiber type 2"}

use_GPU = is_gpu_availiable()
np.random.seed(42)

@st.cache_resource
def st_load_cellpose():
    return load_cellpose()


@st.cache_data
def st_run_cellpose(image_atp, _model):
    return run_cellpose(image_atp, _model)


@st.cache_data
def st_df_from_cellpose_mask(mask):
    return df_from_cellpose_mask(mask)


@st.cache_data
def st_get_all_intensity(image_atp, df_cellpose):
    return get_all_intensity(image_atp, df_cellpose)


@st.cache_data
def st_estimate_threshold(intensity_list):
    return estimate_threshold(intensity_list)


@st.cache_data
def st_plot_density(all_cell_median_intensity, intensity_threshold):
    return plot_density(all_cell_median_intensity, intensity_threshold)


@st.cache_data
def st_predict_all_cells(image_atp, cellpose_df, intensity_threshold):
    return predict_all_cells(image_atp, cellpose_df, intensity_threshold)


@st.cache_data
def st_paint_full_image(image_atp, df_cellpose, class_predicted_all):
    return paint_full_image(image_atp, df_cellpose, class_predicted_all)


model_cellpose = st_load_cellpose()

with st.sidebar:
    st.write("Threshold Parameters")
    intensity_threshold = st.slider("Intensity Threshold (0=auto)", 0, 255, 0, 5)

st.title("ATP Staining Analysis")
st.write(
    "This demo will automatically quantify the number of type 1 muscle fibers vs the number of type 2 muscle fiber on ATP stained images."
)
default_file_url_5 = "https://raw.githubusercontent.com/lambda-science/MyoQuant/refs/heads/main/sample_img/sample_atp.jpg"


st.write("Upload your ATP Staining image OR click the Load Default File button !")
col1, col2 = st.columns(2)
with col1:
    uploaded_file_atp = st.file_uploader("Choose a file")
    if uploaded_file_atp is not None:
        st.session_state["uploaded_file5"] = uploaded_file_atp

with col2:
    if st.button("Load Default File", type="primary"):
        # Download the default file
        response = requests.get(default_file_url_5)
        # Convert the downloaded content into a file-like object
        uploaded_file_atp = BytesIO(response.content)
        st.session_state["uploaded_file5"] = uploaded_file_atp

if "uploaded_file5" in st.session_state:
    uploaded_file_atp = st.session_state["uploaded_file5"]
    # Now you can use the uploaded_file as needed

if uploaded_file_atp is not None:
    image_ndarray_atp = imread(uploaded_file_atp)

    st.write("Raw Image")
    image = st.image(image_ndarray_atp)

    mask_cellpose = st_run_cellpose(image_ndarray_atp, model_cellpose)

    st.header("Segmentation Results")
    st.subheader("CellPose results")
    fig, ax = plt.subplots(1, 1)
    ax.imshow(mask_cellpose, cmap="viridis")
    ax.axis("off")
    st.pyplot(fig)

    df_cellpose = st_df_from_cellpose_mask(mask_cellpose)

    st.header("Cell Intensity Plot")
    all_cell_median_intensity = st_get_all_intensity(image_ndarray_atp, df_cellpose)
    figure_intensity = st_plot_density(all_cell_median_intensity, intensity_threshold)
    st.pyplot(figure_intensity)

    st.header("ATP Cell Classification Results")
    if intensity_threshold == 0:
        (
            muscle_fiber_type_all,
            all_cell_median_intensity,
            intensity_threshold,
        ) = st_predict_all_cells(
            image_ndarray_atp, df_cellpose, intensity_threshold=None
        )
    else:
        (
            muscle_fiber_type_all,
            all_cell_median_intensity,
            intensity_threshold,
        ) = st_predict_all_cells(
            image_ndarray_atp, df_cellpose, intensity_threshold=intensity_threshold
        )
    df_cellpose["muscle_fiber_type"] = muscle_fiber_type_all
    df_cellpose["median_intensity"] = all_cell_median_intensity
    count_per_label = np.unique(muscle_fiber_type_all, return_counts=True)

    st.dataframe(
        df_cellpose.drop(
            [
                "centroid-0",
                "centroid-1",
                "bbox-0",
                "bbox-1",
                "bbox-2",
                "bbox-3",
                "image",
            ],
            axis=1,
        )
    )
    st.write("Total number of cells detected: ", len(muscle_fiber_type_all))
    for index, elem in enumerate(count_per_label[0]):
        st.write(
            "Number of cells classified as ",
            labels_predict[int(elem)+1],
            ": ",
            count_per_label[1][int(index)],
            " ",
            100 * count_per_label[1][int(index)] / len(muscle_fiber_type_all),
            "%",
        )

    st.header("Painted predicted image")
    st.write(
        "Green color indicates cells classified as control, red color indicates cells classified as sick"
    )
    paint_img = st_paint_full_image(
        image_ndarray_atp, df_cellpose, muscle_fiber_type_all
    )
    fig3, ax3 = plt.subplots(1, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["white", "green", "red"]
    )
    ax3.imshow(image_ndarray_atp)
    ax3.imshow(paint_img, cmap=cmap, alpha=0.5)
    ax3.axis("off")
    st.pyplot(fig3)

html(
    f"""
    <script defer data-domain="lbgi.fr/myoquant" src="https://plausible.cmeyer.fr/js/script.js"></script>
    """
)
