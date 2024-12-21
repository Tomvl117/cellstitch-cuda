# CellStitch-Cuda: CUDA-accelerated CellStitch 3D labeling.
![cuda-version](https://img.shields.io/badge/CUDA-11.x%2C_12.x-instanseg)

## About this repo
An overhaul of the CellStitch algorithm, developed by Yining Liu and Yinuo Jin ([original repository](https://github.com/imyiningliu/cellstitch)), publication can be found [here](https://doi.org/10.1186/s12859-023-05608-2).

Some major adjustments:
* Replaced NumPy with CuPy for GPU-accelerated calculations.
* Replaced nested for-loops with vectorized calculations for dramatic speedups (~100x).
* Included novel segmentation method InstanSeg, which enables multichannel inputs ([repo](https://github.com/instanseg/instanseg) and [publication](https://doi.org/10.1101/2024.09.04.611150)).
* An all-in-one method that takes an ZCYX-formatted .tif file, performs the correct transposes, and writes stitched labels.
* Included a histogram-based bleach correction to adjust for signal degradation over the Z-axis (originally developed for ImageJ in (Miura 2020) and released for Python by [marx-alex](https://github.com/marx-alex) in [napari-bleach-correct](https://github.com/marx-alex/napari-bleach-correct)).

### Some comparisons
The calculations were run on the same machine (GPU: NVIDIA Quadro RTX 6000 24 GB; CPU: Intel Xeon Gold 6252 (48/96 cores); RAM: 1024 GB), the core count of which gave it a clear parallel-processing advantage. This particularly affects the `fill_holes_and_remove_small_masks` function, which has been rewritten to utilize parallel processing.
![img-a](figures/cellstitch_img-a.svg)

In image A, GPU VRAM load was ~200 MB at its peak.

![img-b](figures/cellstitch_img-b.svg)

In image B, GPU VRAM load was ~2442 MB at its peak

## Installation
### Notes
This setup has so far only been verified on Windows-based, CUDA-accelerated machines. Testing has only been performed on CUDA 12.x. There are no reasons why 11.x should not work (check instructions), but your mileage may vary.
### Conda setup
```bash
conda create -n cellstitch-cuda python=3.11
conda activate cellstitch-cuda
```
### Install using PyPi
```bash
pip install cellstitch-cuda
pip uninstall torch
conda install pytorch pytorch-cuda=12.4 -c conda-forge -c pytorch -c nvidia
```
You may replace the version number for `pytorch-cuda` with whatever is applicable for you.
#### Additional steps for CUDA 11.x
```bash
pip uninstall cupy-cuda12x
pip install cupy-cuda11x
```
## Instructions
### Example code
For more detail, see `examples/`.
#### From an image
This assumes a multichannel grayscale image in the order ZCYX. Single-channel images are currently not supported, but will be in the future.
```python
from cellstitch_cuda.pipeline import cellstitch_cuda

img = "path/to/image.tif"  # ZCYX
# or feed img as a numpy ndarray

volumetric_masks = cellstitch_cuda(img)
```
#### From pre-existing orthogonal labels
These are label images over the Z-, X, and Y-axis. They are assumed to be in the order ZYX. If you set `output_masks=True` in the `cellstitch_cuda()`-function, these masks will be written to disk (either in the input folder, or in the folder set in `output_path`).
```python
from cellstitch_cuda.pipeline import full_stitch
import tifffile

# Define xy_masks, yz_masks, xz_masks
yx_masks = tifffile.imread("path/to/yx_masks.tif")  # ZYX
yz_masks = tifffile.imread("path/to/yz_masks.tif")  # ZYX
xz_masks = tifffile.imread("path/to/xz_masks.tif")  # ZYX

volumetric_masks = full_stitch(yx_masks, yz_masks, xz_masks)
```
### Arguments
#### cellstitch_cuda.pipeline.cellstitch_cuda()
`cellstitch_cuda()` takes the following arguments:
* **img**: Either a path pointing to an existing image, or a numpy.ndarray. Must be 4D (ZCYX).
* **output_masks**: True to write all masks to the output path, or False to only return the final stitched mask.
    Default False
* **output_path**: Set to None to write to the input file location (if provided). Ignored of output_masks is False.
    Default None
* **stitch_method**: "iou" for Cellpose IoU stitching, or "cellstitch" for CellStitch stitching.
    Default "cellstitch"
* **seg_mode**: Instanseg segmentation mode: "nuclei" to only return nuclear masks, "cells" to return all the cell
    masks (including those without nuclei), or "nuclei_cells", which returns only cells with detected nuclei.
    Default "nuclei_cells"
* **pixel_size**: XY pixel size in microns per pixel. When set to None, will be read from img metadata if possible.
    Default None
* **z_step**: Z pixel size (z step) in microns per step. When set to None, will be read from img metadata if possible.
    Default None
* **bleach_correct**: Whether histogram-based signal degradation correction should be applied to img.
    Default True
* **filtering**: Whether the fill_holes_and_remove_small_masks function should be executed. With larger datasets, this
    has the tendency to massively slow down the postprocessing.
    Default True
* **n_jobs**: Set the number of threads to be used in parallel processing tasks. Use 1 for debugging. Generally, best
    left at the default value.
    Default -1
* **verbose**: Verbosity.
    Default False
#### cellstitch_cuda.pipeline.full_stitch()
`full_stitch()` takes the following arguments:
* **xy_masks_prior**: numpy.ndarray with XY masks, order ZYX
* **yz_masks**: numpy.ndarray with YZ masks, order ZYX
* **xz_masks**: numpy.ndarray with XZ masks, order ZYX
* **nuclei**: numpy.ndarray with XY masks of nuclei, order ZYX. If provided, it will run the function `filter_nuclei_cells()` to filter volumetric masks by the presence of a 2D nucleus mask. Default None
* **filter**: Use CellPose-based fill_holes_and_remove_small_masks() function. Default True
* **n_jobs**: Number of threads used. Set n_jobs to 1 for debugging parallel processing tasks. Default -1
* **verbose**: Verbosity. Default False


## References
Goldsborough, T., O’Callaghan, A., Inglis, F., Leplat, L., Filbey, A., Bilen, H., & Bankhead, P. (2024) A novel channel invariant architecture for the segmentation of cells and nuclei in multiplexed images using InstanSeg. bioRxiv, 2024.09.04.611150. doi: [10.1101/2024.09.04.611150](https://doi.org/10.1101/2024.09.04.611150)

Liu, Y., Jin, Y., Azizi, E., & Blumberg, E. (2023) Cellstitch: 3D cellular anisotropic image segmentation via optimal transport. BMC Bioinformatics, 24(480). doi: [10.1186/s12859-023-05608-2](https://doi.org/10.1186/s12859-023-05608-2)

Miura, K. (2020) Bleach correction ImageJ plugin for compensating the photobleaching of time-lapse sequences. F1000Res, 9:1494. doi: [10.12688/f1000research.27171.1](https://doi.org/10.12688/f1000research.27171.1)

Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021) Cellpose: a generalist algorithm for cellular segmentation. Nature Methods, 18(1), 100-106. doi: [10.1038/s41592-020-01018-x](https://doi.org/10.1038/s41592-020-01018-x)
