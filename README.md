# CellStitch-Instanseg: CUDA-accelerated CellStitch 3D labeling using Instanseg segmentation.

## About this repo
Here, we combine the powerful 3D stitching tool CellStitch in combination with the recently released InstanSeg, which enables multiplexed volumetric data to be used as input.
An overhaul of the CellStitch algorithm, developed by Yining Liu and Yinuo Jin ([orignal repo](https://github.com/imyiningliu/cellstitch)), publication can be found [here](https://doi.org/10.1186/s12859-023-05608-2).
Some major adjustments:
* Replaced NumPy with CuPy for GPU-accelerated calculations
* Replaced nested for-loops with vectorized calculations for dramatic speedups (~100x)
* Included novel segmentation method InstanSeg, which enables multichannel inputs ([repo](https://github.com/instanseg/instanseg) and [publication](https://doi.org/10.1101/2024.09.04.611150))
* An all-in-one method that takes an ZCYX-formatted .tif file, performs the correct transposes, and writes stitched labels

## Installation
### Notes
This setup has so far only been verified on Windows-based, CUDA-accelerated machines. Testing has only been performed on CUDA 12.x. There are no reasons why 11.x should not work (check instructions), but your mileage may vary.
### Conda setup
```bash
conda create -n cellstitch-instanseg python=3.9
conda activate cellstitch-instanseg
```
### Clone repo and install
```bash
conda install git
git clone https://github.com/Tomvl117/cellstitch-instanseg.git
cd cellstitch-instanseg
pip install -e .
```
#### For CUDA 11.x
```bash
pip uninstall cupy-cuda12x
pip install cupy-cuda11x
```
### GPU acceleration (Windows)
#### CUDA 12.1
```bash
conda install pytorch pytorch-cuda=12.1 -c conda-forge -c pytorch -c nvidia
```
You may replace the version number for `pytorch-cuda` with whatever is applicable for you.

## Instructions
### From an image
```python
from cellstitch_cuda.pipeline import cellstitch_cuda

img = "path/to/image.tif"
# or feed img as a numpy ndarray

3d_masks = cellstitch_cuda(img)
```
### From pre-existing orthogonal labels
```python
from cellstitch_cuda.pipeline import full_stitch

# Define xy_masks, yz_masks, xz_masks in some way

3d_masks = full_stitch(xy_masks, yz_masks, xz_masks)
```
