try:
    import xfaster as xf
except ImportError as e:
    import os

    pth = os.path.abspath(os.path.dirname(__file__))
    pth = pth.split("/scripts", 1)[0]
    if not pth.endswith("xfaster"):
        msg = """{}

Please follow the setup instructions in xfaster/README.
""".format(
            e
        )
    else:
        msg = """{}

Please add the following lines to your .bash_profile:

    export XFASTER_PATH={}
    source $XFASTER_PATH/scripts/xfaster_env.sh

Then reload your profile by running

    source ~/.bash_profile
""".format(
            e, pth
        )
    raise ImportError(msg)

# use this script to run the xfaster algorithm with your own options
# like the example below

submit = True  # submit to queue
submit_opts = {
    "wallt": 5,
    "mem": 3,
    "ppn": 1,
    "omp_threads": 1,
}

xfaster_opts = {
    # run options
    "pol": True,
    "pol_mask": True,
    "bin_width": 25,
    "lmax": 500,
    "multi_map": True,
    "likelihood": True,
    "output_root": "/data/agambrel/spectra/example",
    "output_tag": "testing",
    # input files
    "config": "config_example.ini",
    "data_root": "/data/agambrel/XF_NonNull_Sept18",
    "data_subset": "full/*90,full/*150a",
    "clean_type": "raw",
    "noise_type": "stationary",
    "mask_type": "pointsource_latlon",
    "signal_type": "r0",
    "data_root2": None,
    "data_subset2": None,
    # residual fitting
    "residual_fit": True,
    "foreground_fit": False,
    "bin_width_res": 100,
    # spectrum
    "ensemble_mean": False,
    "sim_index": None,
    "tbeb": True,
    "converge_criteria": 0.005,
    "iter_max": 200,
    "save_iters": False,
    # likelihood
    "like_lmin": 26,
    "like_lmax": 250,
    # beams
    "pixwin": True,
    "verbose": "detail",
    "checkpoint": None,
}

if submit:
    xfaster_opts.update(**submit_opts)
    xf.xfaster_submit(**xfaster_opts)
else:
    xf.xfaster_run(**xfaster_opts)
