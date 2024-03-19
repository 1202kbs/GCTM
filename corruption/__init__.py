# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

def build_corruption(opt, corrupt_type=None):

    if corrupt_type is None: corrupt_type = opt.corrupt

    if 'inpaint' in corrupt_type:
        from .inpaint import (build_inpaint_center, build_inpaint_freeform,
                              build_inpaint_random) 
        mask = corrupt_type.split("-")[1]
        assert mask in ["center", "random", "freeform1020", "freeform2030"]
        if mask == "center":
            method = build_inpaint_center(opt, mask)
        elif "freeform" in mask:
            method = build_inpaint_freeform(opt, mask)
        elif mask == "random":
            method = build_inpaint_random(opt, mask)

    elif 'jpeg' in corrupt_type:
        from .jpeg import build_jpeg
        quality_factor = int(corrupt_type.split("-")[1])
        method = build_jpeg(quality_factor)

    elif 'sr4x' in corrupt_type:
        from .superresolution import build_sr4x
        sr_filter = corrupt_type.split("-")[1]
        assert sr_filter in ["pool", "bicubic"]
        method = build_sr4x(opt, sr_filter, image_size=opt.image_size)
    
    elif 'sr2x' in corrupt_type:
        from .superresolution import build_sr2x
        sr_filter = corrupt_type.split("-")[1]
        assert sr_filter in ["pool", "bicubic"]
        method = build_sr2x(opt, sr_filter, image_size=opt.image_size)

    elif 'blur' in corrupt_type:
        from .blur import build_blur
        kernel = corrupt_type.split("-")[1]
        assert kernel in ["uni", "gauss"]
        method = build_blur(opt, kernel)

    elif 'mixture' in corrupt_type:
        method = None #
    else:
        raise RuntimeWarning(f"Unknown corruption: {corrupt_type}!")

    return method
