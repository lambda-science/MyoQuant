# `myoquant`

MyoQuant Analysis Command Line Interface

**Usage**:

```console
$ myoquant [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `docs`: Generate documentation
* `he-analysis`: Run the HE analysis and quantification on...
* `sdh-analysis`: Run the SDH analysis and quantification on...

## `myoquant docs`

Generate documentation

**Usage**:

```console
$ myoquant docs [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `generate`: Generate markdown version of usage...

### `myoquant docs generate`

Generate markdown version of usage documentation

**Usage**:

```console
$ myoquant docs generate [OPTIONS]
```

**Options**:

* `--name TEXT`: The name of the CLI program to use in docs.
* `--output FILE`: An output file to write docs to, like README.md.
* `--help`: Show this message and exit.

## `myoquant he-analysis`

Run the HE analysis and quantification on the image.

**Usage**:

```console
$ myoquant he-analysis [OPTIONS] IMAGE_PATH
```

**Arguments**:

* `IMAGE_PATH`: The HE image file path to analyse. If using single channel images, this will be used as cytoplasm image to run CellPose. Please use the --fluo-nuc option to indicate the path to the nuclei single image to run Stardist.  [required]

**Options**:

* `--mask-path FILE`: The path to a binary mask to hide slide region during analysis. It needs to be of the same resolution as input image and only pixel marked as 1 will be analyzed.
* `--cellpose-path FILE`: The pre-computed CellPose mask to use for analysis. Will run Cellpose if no path provided. Required as an image file.
* `--stardist-path FILE`: The pre-computed Stardist mask to use for analysis. Will run Stardist if no path provided. Required as an image file.
* `--output-path PATH`: The path to the folder to save the results. Will save in the same folder as input image if not specified.
* `--cellpose-diameter INTEGER`: Approximative single cell diameter in pixel for CellPose detection. If not specified, Cellpose will try to deduce it.
* `--nms-thresh FLOAT RANGE`: NMS Threshold for Stardist nuclei detection.  [default: 0.4; 0<=x<=1]
* `--prob-thresh FLOAT RANGE`: Probability Threshold for Stardist nuclei detection.  [default: 0.5; 0.5<=x<=1]
* `--eccentricity-thresh FLOAT RANGE`: Eccentricity threshold value for a nucleus to be considered as internalized during nuclei classification. When very close to 1 almost all nuclei are considered as internalized.  [default: 0.75; 0<=x<=1]
* `--export-map / --no-export-map`: Export the original image with cells painted by classification label.  [default: export-map]
* `--export-stats / --no-export-stats`: Export per fiber and per nuclei stat table.  [default: export-stats]
* `--fluo-nuc FILE`: The path to single channel fluo image for nuclei.
* `--help`: Show this message and exit.

## `myoquant sdh-analysis`

Run the SDH analysis and quantification on the image.

**Usage**:

```console
$ myoquant sdh-analysis [OPTIONS] IMAGE_PATH
```

**Arguments**:

* `IMAGE_PATH`: The image file path to analyse.  [required]

**Options**:

* `--mask-path FILE`: The path to a binary mask to hide slide region during analysis. It needs to be of the same resolution as input image and only pixel marked as 1 will be analyzed.
* `--model-path FILE`: The SDH model path to use for analysis. Will download latest one if no path provided.
* `--cellpose-path FILE`: The pre-computed CellPose mask to use for analysis. Will run Cellpose if no path provided. Required as an image file.
* `--output-path PATH`: The path to the folder to save the results. Will save in the current folder if not specified.
* `--cellpose-diameter INTEGER`: Approximative single cell diameter in pixel for CellPose detection. If not specified, Cellpose will try to deduce it.
* `--export-map / --no-export-map`: Export the original image with cells painted by classification label.  [default: export-map]
* `--export-stats / --no-export-stats`: Export per fiber stat table.  [default: export-stats]
* `--help`: Show this message and exit.
