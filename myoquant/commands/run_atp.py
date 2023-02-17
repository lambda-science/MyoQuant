"""
Module that contains the main function to run the ATP analysis
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
def atp_analysis(
    image_path: Path = typer.Argument(
        ...,
        help="The ATP image file path to analyse.",
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
    output_path: Path = typer.Option(
        None,
        help="The path to the folder to save the results. Will save in the same folder as input image if not specified.",
    ),
    intensity_threshold: int = typer.Option(
        None,
        min=1,
        max=254,
        help="Fiber intensity threshold to differenciate between the two fiber types. If not specified, the analysis will try to deduce it.",
    ),
    cellpose_diameter: int = typer.Option(
        None,
        help="Approximative single cell diameter in pixel for CellPose detection. If not specified, Cellpose will try to deduce it.",
    ),
    channel: int = typer.Option(
        None,
        help="Image channel to use for the analysis. If not specified, the analysis will be performed on all three channels.",
    ),
    channel_first: bool = typer.Option(
        False,
        help="If the channel is the first dimension of the image, set this to True. False by default.",
    ),
    rescale_exposure: bool = typer.Option(
        False,
        help="Rescale the image exposure if your image is not in the 0 255 forma, False by default.",
    ),
    n_classes: int = typer.Option(
        2,
        max=10,
        help="The number of classes of cell to detect. If not specified this is defaulted to two classes.",
    ),
    intensity_method: str = typer.Option(
        "median",
        help="The method to use to compute the intensity of the cell. Can be either 'median' or 'mean'.",
    ),
    erosion: int = typer.Option(
        False,
        max=45,
        help="Perform an erosion on the cells images to remove signal in the cell membrane (usefull for fluo). Expressed in percentage of the cell radius",
    ),
    export_map: bool = typer.Option(
        True,
        help="Export the original image with cells painted by classification label.",
    ),
    export_stats: bool = typer.Option(
        True, help="Export per fiber and per nuclei stat table."
    ),
):
    """Run the fibre type 1 vs type 2 analysis on ATP images.
    First input arguments and option are printed in stdout and all modules are imported. Then the input image is mask with the binary mask if provided.
    Then depending on the presence of cellpose , Cellpose is run or not and mask accordingly if binary mask is provided.
    Finally the ATP analysis is run with run_atp_analysis() function and the results are saved in the output folder and some info are printed in stdout.
    """
    start_time = time.time()
    console.print(
        "👋 [bold dark_orange]Welcome to the fiber type 1 vs 2 analysis (ATP images)",
    )

    # Print input arguments and options
    console.print(f"📄 INPUT: raw image: {image_path}", style="blue")

    if cellpose_path is None:
        console.print(
            "💡 INFO: No CellPose mask provided, will run CellPose during the analysis.",
            style="blue",
        )
    else:
        console.print(f"📄 INPUT: CellPose mask: {cellpose_path}", style="blue")

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
            run_cellpose,
            label2rgb,
            blend_image_with_label,
            HiddenPrints,
        )
        from ..src.ATP_analysis import run_atp_analysis
        import numpy as np
        from PIL import Image
        from skimage.exposure import rescale_intensity

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

    # Load raw image, binary mask, cellpose mask if provided.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        progress.add_task(description="Reading all inputs...", total=None)
        image_ndarray = imread(image_path)
        if channel is not None:
            if channel_first:
                # Put the channel as third dimension instead of first
                image_ndarray = np.moveaxis(image_ndarray, 0, -1)
        image_ndarray = image_ndarray[:, :, channel]
        if rescale_exposure:
            image_ndarray = rescale_intensity(
                image_ndarray,
                in_range=(np.amin(image_ndarray), np.amax(image_ndarray)),
                out_range=np.uint8,
            )
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
        if cellpose_path is not None:
            mask_cellpose = imread(cellpose_path)

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

    # If binary mask provided, mask cellpose and stardist mask
    if mask_path is not None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            progress.add_task(
                description="Masking Cellpose mask with binary mask...",
                total=None,
            )
            mask_ndarray = imread(mask_path)
            mask_cellpose = mask_cellpose * mask_ndarray

    # Run the fiber type 1 vs 2 analysis and get the results table, label map and dataframes
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        progress.add_task(description="Detecting fiber types...", total=None)
        result_df, full_label_map, df_cellpose_details, fig = run_atp_analysis(
            image_ndarray,
            mask_cellpose,
            intensity_threshold,
            n_classes,
            intensity_method,
            erosion,
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
            labelRGB_map = label2rgb(image_ndarray, full_label_map)
            if channel is not None:
                overlay_img = blend_image_with_label(
                    image_ndarray, labelRGB_map, fluo=True
                )
            else:
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
    result_df.to_csv(
        output_path / csv_name,
        index=False,
    )
    console.print(
        f"💾 OUTPUT: Summary Table saved as {output_path/csv_name}",
        style="green",
    )
    plot_name = image_path.stem + "_intensity_plot.png"
    fig.savefig(output_path / plot_name)
    console.print(
        f"💾 OUTPUT: Intensity Plot saved as {output_path/plot_name}", style="green"
    )
    if export_map:
        console.print(
            f"💾 OUTPUT: Overlay image saved as {output_path/overlay_filename}",
            style="green",
        )
    if export_stats:
        df_cellpose_details.drop("image", axis=1).to_csv(
            output_path / cell_details_name,
            index=False,
        )
        console.print(
            f"💾 OUTPUT: Cell Table saved as {output_path/cell_details_name}",
            style="green",
        )
    label_map_name = image_path.stem + "_label_map.tiff"
    Image.fromarray(full_label_map).save(output_path / label_map_name)
    console.print(
        f"💾 OUTPUT: Label map saved as {output_path/label_map_name}", style="green"
    )
    console.print("--- %s seconds ---" % (int(time.time() - start_time)))
