"""Example code

This is the bare minimum to run CUDA-accelerated CellStitch. Per the default settings, it will do the following:
    1. Load the .tif file. It can also read a numpy ndarray containing your imagedata.
    2. Determine pixel size and z-step size from image metadata, if possible.
    3. Correct for signal degradation over the Z-axis of each channel.
    4. Segment XY planes using InstanSeg.
    5. Correct for anisotropic measurements (XY to Z measurement difference).
    6. Segment YZ and XZ planes.
    7. Perform an optimized CellStitch calculation.
    8. It will not return any major progress messages or write any files to disk.
"""


from src.cellstitch_cuda.pipeline import cellstitch_cuda


img = r"path/to/image.tif"

masks = cellstitch_cuda(img)
