"""
Module that contains the main function to run the nuclei analysis for HE and fluo images
"""
import time
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

app = typer.Typer()
console = Console()
table = Table(title="Analysis Results 🥳")


@app.command()
def he_analysis(
    image_path: Path = typer.Argument(
        ...,
        help="The HE image file path to analyse. If using single channel images, this will be used as cytoplasm image to run CellPose. Please use the --fluo-nuc option to indicate the path to the nuclei single image to run Stardist.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    mask_path: Path = typer.Option(
        None,
        help="The path to a binary mask to hide slide region during analysis. It needs to be of the same resolution as input image and only pixel marked as 1 will be analyzed.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    cellpose_path: Path = typer.Option(
        None,
        help="The pre-computed CellPose mask to use for analysis. Will run Cellpose if no path provided. Required as an image file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    stardist_path: Path = typer.Option(
        None,
        help="The pre-computed Stardist mask to use for analysis. Will run Stardist if no path provided. Required as an image file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    output_path: Path = typer.Option(
        None,
        help="The path to the folder to save the results. Will save in the same folder as input image if not specified.",
    ),
    cellpose_diameter: int = typer.Option(
        None,
        help="Approximative single cell diameter in pixel for CellPose detection. If not specified, Cellpose will try to deduce it.",
    ),
    nms_thresh: float = typer.Option(
        0.4, help="NMS Threshold for Stardist nuclei detection.", min=0, max=1
    ),
    prob_thresh: float = typer.Option(
        0.5, help="Probability Threshold for Stardist nuclei detection.", min=0.5, max=1
    ),
    eccentricity_thresh: float = typer.Option(
        0.75,
        help="Eccentricity threshold value for a nucleus to be considered as internalized during nuclei classification. When very close to 1 almost all nuclei are considered as internalized.",
        min=0,
        max=1,
    ),
    export_map: bool = typer.Option(
        True,
        help="Export the original image with cells painted by classification label.",
    ),
    export_stats: bool = typer.Option(
        True, help="Export per fiber and per nuclei stat table."
    ),
    fluo_nuc: Path = typer.Option(
        None,
        help="The path to single channel fluo image for nuclei.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
):
    """Run the nuclei position analysis on HE and fluo images.
    First input arguments and option are printed in stdout and all modules are imported. Then the input image is mask with the binary mask if provided.
    Then depending on the presence of cellpose and stardist path, Cellpose and Stardist are run or not and mask accordingly if binary mask is provided.
    Finally the nuclei analysis is run with run_he_analysis() function and the results are saved in the output folder and some info are printed in stdout.
    """
    start_time = time.time()
    console.print(
        "👋 [bold dark_orange]Welcome to the nuclei position analysis (HE and fluo images)",
    )

    # Print input arguments and options
    console.print(f"📄 INPUT: raw image: {image_path}", style="blue")

    if fluo_nuc is not None:
        console.print(f"📄 INPUT: raw nuclear fluo image: {fluo_nuc}", style="blue")

    if cellpose_path is None:
        console.print(
            "💡 INFO: No CellPose mask provided, will run CellPose during the analysis.",
            style="blue",
        )
    else:
        console.print(f"📄 INPUT: CellPose mask: {cellpose_path}", style="blue")

    if stardist_path is None:
        console.print(
            "💡 INFO: No Stardist mask provided, will run Stardist during the analysis.",
            style="blue",
        )
    else:
        console.print(f"📄 INPUT: Stardist mask: {stardist_path}", style="blue")
    if mask_path is not None:
        console.print(f"📄 INPUT: binary mask: {mask_path}", style="blue")

    # Import all modules
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        progress.add_task(description="Importing the libraries...", total=None)
        from ..src.common_func import (
            is_gpu_availiable,
            load_cellpose,
            load_stardist,
            run_cellpose,
            run_stardist,
            label2rgb,
            blend_image_with_label,
            HiddenPrints,
        )
        from ..src.HE_analysis import run_he_analysis
        import numpy as np
        from PIL import Image

        try:
            from imageio.v2 import imread
        except ImportError:
            from imageio import imread

    if output_path is None:
        output_path = image_path.parents[0]
    else:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    if is_gpu_availiable():
        console.print("💡 INFO: GPU is available.", style="blue")
    else:
        console.print("❌ INFO: GPU is not available. Using CPU only.", style="red")

    # Load raw image, binary mask, cellpose and stardist mask if provided.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        progress.add_task(description="Reading all inputs...", total=None)
        image_ndarray = imread(image_path)

        if fluo_nuc is not None:
            fluo_nuc_ndarray = imread(fluo_nuc)
            mix_cyto_nuc = np.maximum(image_ndarray, fluo_nuc_ndarray)
        if mask_path is not None:
            mask_ndarray = imread(mask_path)
            if np.unique(mask_ndarray).shape[0] != 2:
                console.print(
                    "The mask image should be a binary image with only 2 values (0 and 1).",
                    style="red",
                )
                raise ValueError
            if len(image_ndarray.shape) > 2:
                mask_ndarray = np.repeat(
                    mask_ndarray.reshape(
                        mask_ndarray.shape[0], mask_ndarray.shape[1], 1
                    ),
                    image_ndarray.shape[2],
                    axis=2,
                )
            image_ndarray = image_ndarray * mask_ndarray
            if fluo_nuc is not None:
                fluo_nuc_ndarray = fluo_nuc_ndarray * mask_ndarray
        if cellpose_path is not None:
            mask_cellpose = imread(cellpose_path)
        if stardist_path is not None:
            mask_stardist = imread(stardist_path)

    # Run Cellpose if no mask provided
    if cellpose_path is None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            progress.add_task(description="Running CellPose...", total=None)
            model_cellpose = load_cellpose()
            mask_cellpose = run_cellpose(
                image_ndarray, model_cellpose, cellpose_diameter
            )
            mask_cellpose = mask_cellpose.astype(np.uint16)
            cellpose_mask_filename = image_path.stem + "_cellpose_mask.tiff"
            Image.fromarray(mask_cellpose).save(output_path / cellpose_mask_filename)
        console.print(
            f"💾 OUTPUT: CellPose mask saved as {output_path/cellpose_mask_filename}",
            style="green",
        )

    # Run Stardist if no mask provided
    if stardist_path is None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            progress.add_task(description="Running Stardist...", total=None)
            with HiddenPrints():
                if fluo_nuc is None:
                    model_stardist = load_stardist(fluo=False)
                else:
                    model_stardist = load_stardist(fluo=True)
            if fluo_nuc is not None:
                with HiddenPrints():
                    mask_stardist = run_stardist(
                        fluo_nuc_ndarray, model_stardist, nms_thresh, prob_thresh
                    )
            else:
                with HiddenPrints():
                    mask_stardist = run_stardist(
                        image_ndarray, model_stardist, nms_thresh, prob_thresh
                    )
            mask_stardist = mask_stardist.astype(np.uint16)
            stardist_mask_filename = image_path.stem + "_stardist_mask.tiff"
            Image.fromarray(mask_stardist).save(output_path / stardist_mask_filename)
        console.print(
            f"💾 OUTPUT: Stardist mask saved as {output_path/stardist_mask_filename}",
            style="green",
        )

    # If binary mask provided, mask cellpose and stardist mask
    if mask_path is not None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            progress.add_task(
                description="Masking Cellpose and Stardist mask with binary mask...",
                total=None,
            )
            mask_ndarray = imread(mask_path)
            mask_stardist = mask_stardist * mask_ndarray
            mask_cellpose = mask_cellpose * mask_ndarray

    # Run the nuclei position analysis and get the results table, label map and dataframes
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        progress.add_task(
            description="Calculating all nuclei eccentricity scores...", total=None
        )
        result_df, full_label_map, df_nuc_analysis, all_nuc_df_stats = run_he_analysis(
            image_ndarray, mask_cellpose, mask_stardist, eccentricity_thresh
        )
    if export_map:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            progress.add_task(
                description="Blending label and original image together...", total=None
            )
            if fluo_nuc is not None:
                labelRGB_map = label2rgb(mix_cyto_nuc, full_label_map)
                overlay_img = blend_image_with_label(
                    mix_cyto_nuc, labelRGB_map, fluo=True
                )
            else:
                labelRGB_map = label2rgb(image_ndarray, full_label_map)
                overlay_img = blend_image_with_label(image_ndarray, labelRGB_map)
            overlay_filename = image_path.stem + "_label_blend.tiff"
            overlay_img.save(output_path / overlay_filename)

    # Construct the summary table, print all output in stdout and save files in output folder.
    table.add_column("Feature", justify="left", style="cyan")
    table.add_column("Raw Count", justify="center", style="magenta")
    table.add_column("Proportion (%)", justify="right", style="green")
    for index, row in result_df.iterrows():
        table.add_row(
            str(row[0]),
            str(row[1]),
            str(row[2]),
        )
    console.print(table)
    csv_name = image_path.stem + "_results_summary.csv"
    cell_details_name = image_path.stem + "_cell_details.csv"
    nuc_details_name = image_path.stem + "_nuc_details.csv"
    result_df.to_csv(
        output_path / csv_name,
        index=False,
    )
    console.print(
        f"💾 OUTPUT: Summary Table saved as {output_path/csv_name}",
        style="green",
    )
    if export_map:
        console.print(
            f"💾 OUTPUT: Overlay image saved as {output_path/overlay_filename}",
            style="green",
        )
    if export_stats:
        df_nuc_analysis.drop("image", axis=1).to_csv(
            output_path / cell_details_name,
            index=False,
        )
        console.print(
            f"💾 OUTPUT: Cell Table saved as {output_path/cell_details_name}",
            style="green",
        )
        all_nuc_df_stats.drop("image", axis=1).to_csv(
            output_path / nuc_details_name,
            index=False,
        )
        console.print(
            f"💾 OUTPUT: Nuclei Table saved as {output_path/nuc_details_name}",
            style="green",
        )
    label_map_name = image_path.stem + "_label_map.tiff"
    Image.fromarray(full_label_map).save(output_path / label_map_name)
    console.print(
        f"💾 OUTPUT: Label map saved as {output_path/label_map_name}", style="green"
    )
    console.print("--- %s seconds ---" % (int(time.time() - start_time)))
