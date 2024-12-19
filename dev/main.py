from src.cellstitch_cuda.pipeline import cellstitch_cuda


img = r"path/to/image.tif"

masks = cellstitch_cuda(img)
