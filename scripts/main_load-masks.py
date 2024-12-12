import tifffile
import os
from cellpose.utils import stitch3D
from cellstitch.pipeline import full_stitch


stitch_method = "cellstitch"  # "iou" or "cellstitch"
file_path_yx_masks = r"E:\1_DATA\Rheenen\tvl_jr\3d stitching\unmixed3\yx_masks.tif"
file_path_yz_masks = r"E:\1_DATA\Rheenen\tvl_jr\3d stitching\unmixed3\yz_masks.tif"
file_path_xz_masks = r"E:\1_DATA\Rheenen\tvl_jr\3d stitching\unmixed3\xz_masks.tif"
out_path = os.path.split(file_path_yx_masks)[0]

mode = "nuclei_cells"  # Segmentation mode: "nuclei" or "cells" or "nuclei_cells"

# Read YX masks
with tifffile.TiffFile(file_path_yx_masks) as tif:
    yx_masks = tif.asarray()

if stitch_method == "iou":

    print("Running IoU stitching...")

    iou_masks = stitch3D(yx_masks, stitch_threshold=0.25)
    tifffile.imwrite(os.path.join(out_path, "iou_masks.tif"), iou_masks)

elif stitch_method == "cellstitch":

    # Read YZ masks
    with tifffile.TiffFile(file_path_yz_masks) as tif:
        yz_masks = tif.asarray()

    # Read XZ masks
    with tifffile.TiffFile(file_path_xz_masks) as tif:
        xz_masks = tif.asarray()

    print("Running CellStitch stitching...")

    cellstitch_masks = full_stitch(yx_masks, yz_masks, xz_masks)

    tifffile.imwrite(os.path.join(out_path, "cellstitch_masks.tif"), cellstitch_masks)

else:
    print("Incompatible stitching method.")
