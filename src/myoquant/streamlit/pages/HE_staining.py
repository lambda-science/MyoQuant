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
from stardist import random_label_cmap
from myoquant.common_func import (
    load_cellpose,
    load_stardist,
    run_cellpose,
    run_stardist,
    is_gpu_availiable,
    df_from_cellpose_mask,
    df_from_stardist_mask,
)
from myoquant.HE_analysis import (
    predict_all_cells,
    extract_ROIs,
    single_cell_analysis,
    paint_histo_img,
)


use_GPU = is_gpu_availiable()


@st.cache_resource
def st_load_cellpose():
    return load_cellpose()


@st.cache_resource
def st_load_stardist():
    return load_stardist()


@st.cache_data
def st_run_cellpose(image_ndarray, _model):
    return run_cellpose(image_ndarray, _model)


@st.cache_data
def st_run_stardist(image_ndarray, _model, nms_thresh, prob_thresh):
    return run_stardist(image_ndarray, _model, nms_thresh, prob_thresh)


@st.cache_data
def st_df_from_cellpose_mask(mask):
    return df_from_cellpose_mask(mask)


@st.cache_data
def st_df_from_stardist_mask(mask):
    return df_from_stardist_mask(mask)


@st.cache_data
def st_predict_all_cells(
    image_ndarray, df_cellpose, mask_stardist, internalised_threshold
):
    return predict_all_cells(
        image_ndarray, df_cellpose, mask_stardist, internalised_threshold
    )


@st.cache_data
def st_extract_ROIs(image_ndarray, selected_fiber, df_cellpose, mask_stardist):
    return extract_ROIs(image_ndarray, selected_fiber, df_cellpose, mask_stardist)


@st.cache_data
def st_single_cell_analysis(
    single_cell_img,
    single_cell_mask,
    df_nuc_single,
    x_fiber,
    y_fiber,
    selected_fiber,
    internalised_threshold,
    draw_and_return=True,
):
    return single_cell_analysis(
        single_cell_img,
        single_cell_mask,
        df_nuc_single,
        x_fiber,
        y_fiber,
        selected_fiber + 1,
        internalised_threshold,
        draw_and_return=True,
    )


@st.cache_data
def st_paint_histo_img(image_ndarray, df_cellpose, cellpose_df_stat):
    return paint_histo_img(image_ndarray, df_cellpose, cellpose_df_stat)


with st.sidebar:
    st.write("Nuclei detection Parameters (Stardist)")
    nms_thresh = st.slider("Stardist NMS Tresh", 0.0, 1.0, 0.4, 0.1)
    prob_thresh = st.slider("Stardist Prob Tresh", 0.5, 1.0, 0.5, 0.05)
    st.write("Nuclei Classification Parameter")
    eccentricity_thresh = st.slider("Eccentricity Score Tresh", 0.0, 1.0, 0.75, 0.05)

model_cellpose = st_load_cellpose()
model_stardist = st_load_stardist()

st.title("HE Staining Analysis")
st.write(
    "This demo will automatically detect cells and nucleus in the image and try to quantify a certain number of features."
)

default_file_url = "https://raw.githubusercontent.com/lambda-science/MyoQuant/refs/heads/main/sample_img/sample_he.jpg"


st.write("Upload your HE Staining image OR click the Load Default File button !")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file

with col2:
    if st.button("Load Default File", type="primary"):
        # Download the default file
        response = requests.get(default_file_url)
        # Convert the downloaded content into a file-like object
        uploaded_file = BytesIO(response.content)
        st.session_state["uploaded_file"] = uploaded_file

if "uploaded_file" in st.session_state:
    uploaded_file = st.session_state["uploaded_file"]
    uploaded_file.seek(0)

if uploaded_file is not None:
    image_ndarray = imread(uploaded_file)

    st.write("Raw Image")
    image = st.image(image_ndarray)

    mask_cellpose = st_run_cellpose(image_ndarray, model_cellpose)
    mask_stardist = st_run_stardist(
        image_ndarray, model_stardist, nms_thresh, prob_thresh
    )
    mask_stardist_copy = mask_stardist.copy()
    st.header("Segmentation Results")
    st.subheader("CellPose and Stardist overlayed results")
    fig, ax = plt.subplots(1, 1)
    ax.imshow(mask_cellpose, cmap="viridis")
    lbl_cmap = random_label_cmap()
    ax.imshow(mask_stardist, cmap=lbl_cmap, alpha=0.5)
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("All cells detected by CellPose")
    df_cellpose = st_df_from_cellpose_mask(mask_cellpose)

    st.header("Full Nucleus Analysis Results")
    cellpose_df_stat, all_nuc_df_stats = st_predict_all_cells(
        image_ndarray,
        df_cellpose,
        mask_stardist,
        internalised_threshold=eccentricity_thresh,
    )
    st.dataframe(
        cellpose_df_stat.drop(
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
    st.write("Total number of nucleus : ", cellpose_df_stat["N° Nuc"].sum())
    st.write(
        "Total number of internalized nucleus : ",
        cellpose_df_stat["N° Nuc Intern"].sum(),
        " (",
        round(
            100
            * cellpose_df_stat["N° Nuc Intern"].sum()
            / cellpose_df_stat["N° Nuc"].sum(),
            2,
        ),
        "%)",
    )
    st.write(
        "Total number of peripherical nucleus : ",
        cellpose_df_stat["N° Nuc Periph"].sum(),
        " (",
        round(
            100
            * cellpose_df_stat["N° Nuc Periph"].sum()
            / cellpose_df_stat["N° Nuc"].sum(),
            2,
        ),
        "%)",
    )
    st.write(
        "Number of cell with at least one internalized nucleus : ",
        cellpose_df_stat["N° Nuc Intern"].astype(bool).sum(axis=0),
        " (",
        round(
            100
            * cellpose_df_stat["N° Nuc Intern"].astype(bool).sum(axis=0)
            / len(cellpose_df_stat),
            2,
        ),
        "%)",
    )

    st.header("Single Nucleus Analysis Details")
    selected_fiber = st.selectbox("Select a cell", list(range(len(df_cellpose))))
    selected_fiber = int(selected_fiber)
    (
        single_cell_img,
        nucleus_single_cell_img,
        single_cell_mask,
        df_nuc_single,
    ) = st_extract_ROIs(image_ndarray, selected_fiber, df_cellpose, mask_stardist)

    # df_nuc_single = df_from_stardist_mask(mask_stardist)
    st.markdown(
        """
        * White point represent cell centroid. 
        * Green point represent nucleus centroid. Green dashed line represent the fiber centrer - nucleus distance. 
        * Red point represent the cell border from a straight line between the cell centroid and the nucleus centroid. The red dashed line represent distance between the nucelus and the cell border. 
        * The periphery ratio is calculated by the division of the distance centroid - nucleus and the distance centroid - cell border."""
    )

    fig2, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(single_cell_img)
    ax2.imshow(nucleus_single_cell_img, cmap="viridis")
    # Plot Fiber centroid
    x_fiber = df_cellpose.iloc[selected_fiber, 3] - df_cellpose.iloc[selected_fiber, 6]
    y_fiber = df_cellpose.iloc[selected_fiber, 2] - df_cellpose.iloc[selected_fiber, 5]

    (
        n_nuc,
        n_nuc_intern,
        n_nuc_periph,
        df_nuc_single_stats,
        ax_nuc,
    ) = st_single_cell_analysis(
        single_cell_img,
        single_cell_mask,
        df_nuc_single,
        x_fiber,
        y_fiber,
        selected_fiber + 1,
        internalised_threshold=eccentricity_thresh,
        draw_and_return=True,
    )
    for index, value in df_nuc_single_stats.iterrows():
        st.write("Nucleus #{} has a periphery ratio of: {}".format(index, value[12]))
    ax1.axis("off")
    ax2.axis("off")
    # st.pyplot(fig2)
    ax_nuc.imshow(single_cell_img)
    ax_nuc.imshow(nucleus_single_cell_img, cmap="viridis", alpha=0.5)
    f = ax_nuc.figure

    st.pyplot(fig2)
    st.pyplot(f)
    st.subheader("All nucleus inside selected cell")

    st.dataframe(
        df_nuc_single_stats.drop(
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

    st.header("Painted predicted image")
    st.write(
        "Green color indicates cells with only peripherical nuclei, red color indicates cells with at least one internal nucleus."
    )
    painted_img = st_paint_histo_img(image_ndarray, df_cellpose, cellpose_df_stat)
    fig4, ax4 = plt.subplots(1, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["white", "green", "red"]
    )
    ax4.imshow(image_ndarray)
    ax4.imshow(painted_img, cmap=cmap, alpha=0.5)
    ax4.axis("off")
    st.pyplot(fig4)

html(
    f"""
    <script defer data-domain="lbgi.fr/myoquant" src="https://plausible.cmeyer.fr/js/script.js"></script>
    """
)
