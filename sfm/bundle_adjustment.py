import pycolmap

def run_bundle_adjustment(reconstruction):
    # pycolmap wraps COLMAP's C++ BA engine — runs at full C++ speed
    options = pycolmap.BundleAdjustmentOptions()
    options.solver_options.max_num_iterations = 100
    options.refine_focal_length = True
    options.refine_principal_point = False
    options.refine_extra_params = True  # distortion coefficients

    pycolmap.bundle_adjustment(reconstruction, options)
    return reconstruction