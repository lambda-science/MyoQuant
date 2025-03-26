"""
Module that contains the main function to run the mitochondrial distribution analysis for SDH images
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
def sdh_analysis(
    image_path: Path = typer.Argument(
        ...,
        help="The image file path to analyse.",
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
    model_path: Path = typer.Option(
        None,
        help="The SDH model path to use for analysis. Will download latest one if no path provided.",
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
        help="The path to the folder to save the results. Will save in the current folder if not specified.",
    ),
    cellpose_diameter: int = typer.Option(
        None,
        help="Approximative single cell diameter in pixel for CellPose detection. If not specified, Cellpose will try to deduce it.",
    ),
    export_map: bool = typer.Option(
        True,
        help="Export the original image with cells painted by classification label.",
    ),
    export_stats: bool = typer.Option(True, help="Export per fiber stat table."),
):
    """Run the mitochondiral analysis and quantification on the image.
    First input arguments and option are printed in stdout and all modules are imported and latest SDH model is downloaded.
    Then the input image is mask with the binary mask if provided.
    Then depending on the presence of cellpose path, Cellpose is run or not and mask accordingly if binary mask is provided.
    Finally the mitochondiral classificaiton is run with run_sdh_analysis() function and the results are saved in the output folder and some info are printed in stdout.
    """
    start_time = time.time()
    console.print(
        "👋 [bold dark_orange]Welcome to the mitochondrial distribution analysis (SDH images)",
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
    import os

    # If the model path is not provided, download latest version or check existence.
    if model_path is None:
        import urllib.request

        console.print(
            "💡 INFO: No SDH model provided, will download or use latest one.",
            style="blue",
        )
        model_path_abs = Path(os.path.abspath(__file__)).parents[0] / "model.h5"
        if not os.path.exists(model_path_abs):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                transient=False,
            ) as progress:
                progress.add_task(description="Downloading SDH Model...", total=None)
                urllib.request.urlretrieve(
                    "https://lbgi.fr/~meyer/SDH_models/model.h5",
                    model_path_abs,
                )
        model_path = model_path_abs
    console.print(f"📄 INPUT: SDH Model: {model_path}", style="blue")

    # Import all modules
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        progress.add_task(description="Importing the libraries...", total=None)
        from myoquant.common_func import (
            is_gpu_availiable,
            load_cellpose,
            load_sdh_model,
            run_cellpose,
            label2rgb,
            blend_image_with_label,
            HiddenPrints,
        )
        from myoquant.SDH_analysis import run_sdh_analysis
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
        image_ndarray_sdh = imread(image_path)

        if mask_path is not None:
            mask_ndarray = imread(mask_path)
            if np.unique(mask_ndarray).shape[0] != 2:
                console.print(
                    "The mask image should be a binary image with only 2 values (0 and 1).",
                    style="red",
                )
                raise ValueError
            if len(image_ndarray_sdh.shape) > 2:
                mask_ndarray = np.repeat(
                    mask_ndarray.reshape(
                        mask_ndarray.shape[0], mask_ndarray.shape[1], 1
                    ),
                    image_ndarray_sdh.shape[2],
                    axis=2,
                )
            image_ndarray_sdh = image_ndarray_sdh * mask_ndarray
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
                image_ndarray_sdh, model_cellpose, cellpose_diameter
            )
            mask_cellpose = mask_cellpose.astype(np.uint16)
            cellpose_mask_filename = image_path.stem + "_cellpose_mask.tiff"
            Image.fromarray(mask_cellpose).save(output_path / cellpose_mask_filename)
        console.print(
            f"💾 OUTPUT: CellPose mask saved as {output_path/cellpose_mask_filename}",
            style="green",
        )

    # Load Tensorflow SDH Model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        progress.add_task(
            description="Loading SDH Model...",
            total=None,
        )
        with HiddenPrints():
            model_SDH = load_sdh_model(model_path)

    # If binary mask provided, mask cellpose mask
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
        mask_cellpose = mask_cellpose * mask_ndarray

    # Run the mitoC distribution analysis and get the results table, label map and dataframes
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        progress.add_task(
            description="Predicting all muscle fibers class...", total=None
        )
        with HiddenPrints():
            result_df, full_label_map, df_cellpose_details = run_sdh_analysis(
                image_ndarray_sdh, model_SDH, mask_cellpose
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
            labelRGB_map = label2rgb(image_ndarray_sdh, full_label_map)
            overlay_img = blend_image_with_label(image_ndarray_sdh, labelRGB_map)
            overlay_filename = image_path.stem + "_label_blend.tiff"
            overlay_img.save(output_path / overlay_filename)

    # Construct the summary table, print all output in stdout and save files in output folder.
    table.add_column("Feature", justify="left", style="cyan")
    table.add_column("Raw Count", justify="center", style="magenta")
    table.add_column("Proportion (%)", justify="right", style="green")
    for index, row in result_df.iterrows():
        table.add_row(
            str(row.iloc[0]),
            str(row.iloc[1]),
            str(row.iloc[2]),
        )
    console.print(table)
    csv_name = image_path.stem + "_results.csv"
    result_df.to_csv(
        output_path / csv_name,
        index=False,
    )
    console.print(
        f"💾 OUTPUT: Summary Table saved as {output_path/csv_name}", style="green"
    )
    if export_stats:
        cell_details_name = image_path.stem + "_cell_details.csv"
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
    console.print("--- %s seconds ---" % int((time.time() - start_time)))
