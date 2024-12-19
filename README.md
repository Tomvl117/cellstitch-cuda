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
### From an image
```python
from cellstitch_cuda.pipeline import cellstitch_cuda

img = "path/to/image.tif"
# or feed img as a numpy ndarray

volumetric_masks = cellstitch_cuda(img)
```
### From pre-existing orthogonal labels
```python
from cellstitch_cuda.pipeline import full_stitch

# Define xy_masks, yz_masks, xz_masks in some way

volumetric_masks = full_stitch(xy_masks, yz_masks, xz_masks)
```

## References
Goldsborough, T., O’Callaghan, A., Inglis, F., Leplat, L., Filbey, A., Bilen, H., & Bankhead, P. (2024) A novel channel invariant architecture for the segmentation of cells and nuclei in multiplexed images using InstanSeg. bioRxiv, 2024.09.04.611150. doi: [10.1101/2024.09.04.611150](https://doi.org/10.1101/2024.09.04.611150)

Liu, Y., Jin, Y., Azizi, E., & Blumberg, E. (2023) Cellstitch: 3D cellular anisotropic image segmentation via optimal transport. BMC Bioinformatics, 24(480). doi: [10.1186/s12859-023-05608-2](https://doi.org/10.1186/s12859-023-05608-2)

Miura, K. (2020) Bleach correction ImageJ plugin for compensating the photobleaching of time-lapse sequences. F1000Res, 9:1494. doi: [10.12688/f1000research.27171.1](https://doi.org/10.12688/f1000research.27171.1)

Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021) Cellpose: a generalist algorithm for cellular segmentation. Nature Methods, 18(1), 100-106. doi: [10.1038/s41592-020-01018-x](https://doi.org/10.1038/s41592-020-01018-x)
