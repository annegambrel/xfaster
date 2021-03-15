from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import healpy as hp
import datetime
import os
import sys
import glob
import json
import re
import warnings
import copy
import configparser
from collections import OrderedDict
from scipy import integrate, stats
import pickle
from xfaster import xfaster_tools as xft
from xfaster import parse_tools as pt
from xfaster import base as base
from xfaster import batch_tools as bt

__all__ = ["XFaster"]


class XFaster(object):

    checkpoints = [
        "files",
        "masks",
        "kernels",
        "sims_transfer",
        "shape_transfer",
        "transfer",
        "sims",
        "beams",
        "data",
        "template_noise",
        "shape",
        "bandpowers",
        "likelihood",
    ]

    # if starting from KEY, force rerun all steps in VALUES
    checkpoint_tree = {
        "files": ["masks", "sims", "data"],
        "masks": ["kernels", "sims_transfer", "sims", "data"],
        "kernels": ["transfer", "bandpowers"],
        "sims_transfer": ["transfer"],
        "shape_transfer": ["transfer"],
        "transfer": ["bandpowers"],
        "sims": ["bandpowers"],
        "beams": ["transfer"],
        "data": ["bandpowers"],
        "template_noise": ["bandpowers"],
        "shape": ["bandpowers"],
        "bandpowers": ["likelihood"],
    }

    def __init__(
        self,
        config,
        lmax=500,
        pol=True,
        pol_mask=True,
        output_root=None,
        output_tag=None,
        verbose=True,
        debug=False,
        checkpoint=None,
        add_log=False,
        ref_freq=353.0,
        beta_ref=1.54,
        **kwargs
    ):
        """
        Initialize an XFaster instance for computing binned power spectra
        using a set of data maps along with signal and noise simulations.

        Arguments
        ---------
        lmax : int
            The maximum multipole for which spectra are computed
        pol : bool
            If True, polarized spectra are computed from the input maps
        pol_mask : bool
            If True, two independent masks are applied to every map:
            one for T maps and one for Q/U maps.
        output_root : string
            Path to data output directory
        output_tag : string
            Tag to use for output data.  Results are typically stored in
            the form `<output_root>/<output_tag>/<name>_<output_tag>.npz`
        verbose : string
            Verbosity level to use for log messages.  Can be one of
            ['user', 'info', 'task', 'part', 'detail', 'all'].
        debug : bool
            Store extra data in output files for debugging.
        checkpoint : string
            If output data from this step forward exist on disk, they are
            are re-computed rather than loading from file.
            Options are {checkpoints}.
        add_log : bool
            If True, write log output to a file instead of to STDOUT.
            The log will be in `<output_root>/run_<output_tag>.log`.
        """
        # verbosity
        self.log = self.init_log(**kwargs)
        if verbose is not None:
            self.set_verbose(verbose)

        self.debug = debug
        self.lmax = lmax
        self.pol = pol
        self.pol_dim = 3 if self.pol else 1
        self.pol_mask = pol_mask if self.pol else False
        self.planck_freqs = ["100", "143", "217", "353"]

        self.ref_freq = ref_freq
        self.beta_ref = beta_ref

        # Fix this
        # Priors on frequency spectral index
        self.delta_beta_fix = 1.0e-8

        cfg = configparser.ConfigParser()
        if not os.path.exists(config):
            config = os.path.join(os.getenv("XFASTER_PATH"), "config", config)
        assert os.path.exists(config)

        cfg.read(config)
        self.dict_freqs = cfg["freqs"]
        self.fwhm = cfg["fwhm"]
        self.beam_product = cfg["beam"]["beam_product"]
        if self.beam_product is not None:
            if not os.path.exists(self.beam_product):
                self.beam_product = os.path.join(
                    os.getenv("XFASTER_PATH"), "config", self.beam_product
                )
            assert os.path.exists(self.beam_product)

        # TO DO: shouldn't need this list, just use original tags
        self.dict_freqs_nom = {
            "x1": "150",
            "x2": "90",
            "x3": "150",
            "x4": "90",
            "x5": "150",
            "x6": "90",
            "90": "90",
            "100": "100",
            "143": "143",
            "150": "150",
            "150a": "150",
            "217": "217",
            "353": "353",
        }

        # checkpointing
        if checkpoint is not None:
            if checkpoint not in self.checkpoints:
                raise ValueError(
                    "Invalid checkpoint {}, must be one of {}".format(
                        checkpoint, self.checkpoints
                    )
                )
        self.checkpoint = checkpoint
        self.force_rerun = {cp: False for cp in self.checkpoints}

        if output_root is None:
            output_root = os.getcwd()
            warnings.warn("No output root supplied, using {}".format(output_root))

        self.output_root = output_root
        self.output_tag = output_tag
        if self.output_tag is not None:
            self.output_root = os.path.join(self.output_root, self.output_tag)
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

        if add_log:
            self.set_logfile(self.get_filename("run", ext=".log"))

    __init__.__doc__ = __init__.__doc__.format(checkpoints=checkpoints)

    def init_log(
        self,
        verbose=0,
        logger=base.Logger,
        logfile=None,
        log_timestamp=True,
        log_prefix=None,
        **kwargs
    ):
        """
        Initialize the logger from the input keyword arguments.

        Arguments
        ---------
        logger : logging class, optional
            Class to initialize
        verbose : bool, int, or string; optional
            Verbosity level, non-negative.  Default: 0 (print user-level
            messages only). String options are 'info', 'time', 'gd', or 'samp'.
        logfile : string, optional
            Logging output filename.  Default: None (print to sys.stdout)
        log_timestamp : bool, optional
            If True, add timestamps to log entries. Default: True
        log_prefix : string, optional
            If supplied, this prefix will be pre-pended to log strings,
            before the timestamp.

        Returns
        -------
        log : log object
            Initialized logging object
        """
        if verbose is None:
            verbose = 0
        if logfile is None:
            # try to get logfile suitable for job environment
            logfile = bt.get_job_logfile()
        timestamp = log_timestamp
        prefix = log_prefix

        levels = {"user": 0, "info": 1, "task": 2, "part": 3, "detail": 4, "all": 5}
        log = logger(
            verbosity=verbose,
            logfile=logfile,
            timestamp=timestamp,
            prefix=prefix,
            levels=levels,
            **kwargs
        )
        return log

    def set_verbose(self, level):
        """
        Change verbosity level.  Can be an integer or a string name.
        Valid strings are 'info', 'time', 'gd' or 'samp'.
        """
        self.log.set_verbosity(level)

    def set_logfile(self, logfile=None):
        """
        Change the location where logs are written.  If logfile is None,
        log to STDOUT.
        """
        self.log.set_logfile(logfile)

    def _get_files(
        self,
        data_root,
        data_subset="full/*0",
        signal_subset="*",
        noise_subset="*",
        clean_type="raw",
        noise_type="stationary",
        noise_type_sim=None,
        mask_type="hitsmask_tailored",
        signal_type="r0p03",
        signal_type_sim=None,
        signal_transfer_type=None,
        suffix="",
        foreground_type=None,
        template_type=None,
        sub_planck=False,
        planck_reobs=True,
    ):
        """
        Find all files for the given data root.  Internal function, see
        `get_files` for a complete docstring.
        """

        if signal_transfer_type is None:
            signal_transfer_type = signal_type

        # regularize data root
        if not os.path.exists(data_root) and not os.path.isabs(data_root):
            xfaster_root = os.getenv("XFASTER_DATA_ROOT")
            if xfaster_root is None:
                raise OSError(
                    "Missing XFaster data root name.  Please supply the "
                    "data_root keyword argument as an absolute path or define "
                    "the XFASTER_DATA_ROOT environment variable."
                )
            data_root = os.path.join(xfaster_root, data_root)
            self.log("Loading XFaster input data from {}".format(data_root), "info")

        data_root = os.path.abspath(data_root)
        if not os.path.exists(data_root):
            raise OSError("Missing data root {}".format(data_root))

        # find all map files
        map_root = os.path.join(data_root, "data_{}".format(clean_type))
        map_files = []
        data_subset = data_subset.split(",")
        for f in np.atleast_1d(data_subset):
            files = glob.glob(os.path.join(map_root, "{}.fits".format(f)))
            if not len(files):
                raise OSError("Missing files in data subset {}".format(f))
            map_files.extend(files)
        data_subset = ",".join(data_subset)
        map_files = sorted(map_files)
        map_files = [f for f in map_files if os.path.basename(f).startswith("map_")]
        map_tags = [
            os.path.splitext(os.path.basename(f))[0].split("_", 1)[1] for f in map_files
        ]
        nom_freqs = []
        for t in map_tags:
            # if map tag is not a plain frequency, extract plain frequency
            if int(sys.version[0]) < 3:
                nfreq = filter(lambda x: re.search(x, t), list(self.dict_freqs))[0]
            else:
                nfreq = next(filter(lambda x: re.search(x, t), list(self.dict_freqs)))
            nom_freqs.append(self.dict_freqs_nom[str(nfreq)])
        map_freqs = []
        for t in nom_freqs:
            map_freqs.append(self.dict_freqs[t])
        self.log("Found {} map files in {}".format(len(map_files), map_root), "task")
        self.log("Map files: {}".format(map_files), "detail")
        self.log("Map freqs: {}".format(map_freqs), "detail")

        raw_root = None
        raw_files = None
        # find all corresponding signal sims
        signal_root = os.path.join(data_root, "signal_{}".format(signal_type))
        num_signal = None
        signal_files = []
        for f in map_files:
            sfiles = sorted(
                glob.glob(
                    f.replace(map_root, signal_root).replace(
                        ".fits", "_{}.fits".format(signal_subset)
                    )
                )
            )
            nsims1 = len(sfiles)
            if not nsims1:
                raise OSError("Missing signal sims for {}".format(f))
            if num_signal is None:
                num_signal = nsims1
            else:
                if nsims1 != num_signal:
                    raise OSError(
                        "Found {} signal sims for map {}, expected {}".format(
                            nsims1, f, num_signal
                        )
                    )
                    num_signal = min(num_signal, nsims1)
            signal_files.append(sfiles)
        signal_files = np.asarray([x[:num_signal] for x in signal_files])
        self.log("{} {} {}".format(len(signal_files[0]), num_signal, nsims1))
        self.log("Found {} signal sims in {}".format(num_signal, signal_root), "task")
        self.log(
            "First signal sim files: {}".format(signal_files[:, 0].tolist()), "detail"
        )

        # find all corresponding signal transfer function sims
        signal_transfer_root = os.path.join(
            data_root, "signal_{}".format(signal_transfer_type)
        )
        num_signal_transfer = None
        signal_transfer_files = []
        for f in map_files:
            sfiles = sorted(
                glob.glob(
                    f.replace(map_root, signal_transfer_root).replace(
                        ".fits", "_{}.fits".format(signal_subset)
                    )
                )
            )
            nsims1 = len(sfiles)
            if not nsims1:
                raise OSError("Missing signal sims for {}".format(f))
            if num_signal_transfer is None:
                num_signal_transfer = nsims1
            else:
                if nsims1 != num_signal_transfer:
                    raise OSError(
                        "Found {} signal_transfer sims for map {}, expected {}".format(
                            nsims1, f, num_signal_transfer
                        )
                    )
                    num_signal_transfer = min(num_signal_transfer, nsims1)
            signal_transfer_files.append(sfiles)
        signal_transfer_files = np.asarray(
            [x[:num_signal_transfer] for x in signal_transfer_files]
        )
        self.log(
            "Found {} signal transfer sims in {}".format(
                num_signal_transfer, signal_transfer_root
            ),
            "task",
        )
        self.log(
            "First signal transfer sim files: {}".format(
                signal_transfer_files[:, 0].tolist()
            ),
            "detail",
        )

        # find all corresponding noise sims
        if noise_type is not None:
            noise_root = os.path.join(data_root, "noise_{}".format(noise_type))
            num_noise = None
            noise_files = []
            for f in map_files:
                nfiles = sorted(
                    glob.glob(
                        f.replace(map_root, noise_root).replace(
                            ".fits", "_{}.fits".format(noise_subset)
                        )
                    )
                )
                nsims1 = len(nfiles)
                if not nsims1:
                    raise OSError("Missing noise sims for {}".format(f))
                if num_noise is None:
                    num_noise = nsims1
                else:
                    if nsims1 != num_noise:
                        raise OSError(
                            "Found {} noise sims for map {}, expected {}".format(
                                nsims1, f, num_noise
                            )
                        )
                        num_noise = min(num_noise, nsims1)
                noise_files.append(nfiles)
            noise_files = np.asarray([x[:num_noise] for x in noise_files])
            self.log("{} {} {}".format(len(noise_files[0]), num_noise, nsims1))
            self.log("Found {} noise sims in {}".format(num_noise, noise_root), "task")
            self.log(
                "First noise sim files: {}".format(noise_files[:, 0].tolist()), "detail"
            )
        else:
            noise_root = None
            noise_files = None

        # find all corresponding noise sims for sim_index run
        if noise_type_sim is not None:
            noise_root_sim = os.path.join(data_root, "noise_{}".format(noise_type_sim))
            num_noise_sim = None
            noise_files_sim = []
            for f in map_files:
                nfiles = sorted(
                    glob.glob(
                        f.replace(map_root, noise_root_sim).replace(
                            ".fits", "_{}.fits".format(noise_subset)
                        )
                    )
                )
                nsims1 = len(nfiles)
                if not nsims1:
                    raise OSError("Missing noise sims for {}".format(f))
                if num_noise_sim is None:
                    num_noise_sim = nsims1
                else:
                    if nsims1 != num_noise_sim:
                        raise OSError(
                            "Found {} noise sims for map {}, expected {}".format(
                                nsims1, f, num_noise_sim
                            )
                        )
                        num_noise_sim = min(num_noise_sim, nsims1)
                noise_files_sim.append(nfiles)
            noise_files_sim = np.asarray(noise_files_sim)
            self.log("{} {} {}".format(len(noise_files_sim[0]), num_noise_sim, nsims1))
            self.log(
                "Found {} noise sims in {}".format(num_noise_sim, noise_root_sim),
                "task",
            )
            self.log(
                "First noise sim files: {}".format(noise_files_sim[:, 0].tolist()),
                "detail",
            )
        else:
            noise_root_sim = noise_root
            noise_files_sim = noise_files

        # find all corresponding signal sims for sim_index run
        if signal_type_sim is not None:
            signal_root_sim = os.path.join(
                data_root, "signal_{}".format(signal_type_sim)
            )
            num_signal_sim = None
            signal_files_sim = []
            for f in map_files:
                nfiles = sorted(
                    glob.glob(
                        f.replace(map_root, signal_root_sim).replace(
                            ".fits", "_{}.fits".format(signal_subset)
                        )
                    )
                )
                nsims1 = len(nfiles)
                if not nsims1:
                    raise OSError("Missing signal sims for {}".format(f))
                if num_signal_sim is None:
                    num_signal_sim = nsims1
                else:
                    if nsims1 != num_signal_sim:
                        raise OSError(
                            "Found {} signal sims for map {}, expected {}".format(
                                nsims1, f, num_signal_sim
                            )
                        )
                        num_signal_sim = min(num_signal_sim, nsims1)
                signal_files_sim.append(nfiles)
            signal_files_sim = np.asarray(signal_files_sim)
            self.log(
                "{} {} {}".format(len(signal_files_sim[0]), num_signal_sim, nsims1)
            )
            self.log(
                "Found {} signal sims in {}".format(num_signal_sim, signal_root_sim),
                "task",
            )
            self.log(
                "First signal sim files: {}".format(signal_files_sim[:, 0].tolist()),
                "detail",
            )
        else:
            signal_root_sim = signal_root
            signal_files_sim = signal_files

        # find all corresponding foreground sims for sim_index run
        if foreground_type is not None:
            foreground_root = os.path.join(
                data_root, "foreground_{}".format(foreground_type)
            )
            num_foreground_sim = None
            foreground_files = []
            for f in map_files:
                nfiles = sorted(
                    glob.glob(
                        f.replace(map_root, foreground_root).replace(".fits", "_*.fits")
                    )
                )
                nsims1 = len(nfiles)
                if not nsims1:
                    raise OSError("Missing foreground sims for {}".format(f))
                if num_foreground_sim is None:
                    num_foreground_sim = nsims1
                else:
                    if nsims1 != num_foreground_sim:
                        raise OSError(
                            "Found {} foreground sims for map {}, expected {}".format(
                                nsims1, f, num_foreground_sim
                            )
                        )
                        num_foreground_sim = min(num_foreground_sim, nsims1)
                foreground_files.append(nfiles)
            foreground_files = np.asarray(
                [x[:num_foreground_sim] for x in foreground_files]
            )
            self.log(
                "{} {} {}".format(len(foreground_files[0]), num_foreground_sim, nsims1)
            )
            self.log(
                "Found {} foreground sims in {}".format(
                    num_foreground_sim, foreground_root
                ),
                "task",
            )
            self.log(
                "First foreground sim files: {}".format(
                    foreground_files[:, 0].tolist()
                ),
                "detail",
            )
        else:
            foreground_root = None
            foreground_files = None

        # find all corresponding masks
        if mask_type is None:
            raise ValueError("Argument mask_type required")
        # If mask is a fits file, use the same mask for all maps
        if os.path.splitext(mask_type)[1] == ".fits":
            if os.path.exists(mask_type):
                # it's an absolute path
                mask_files = np.tile(mask_type, len(map_tags))
                mask_root = os.path.dirname(mask_type)
            else:
                # it's relative to base directory structure
                mask_files = np.tile(os.path.join(data_root, mask_type), len(map_tags))
                mask_root = os.path.dirname(os.path.join(data_root, mask_type))
        else:
            mask_root = os.path.join(data_root, "masks_{}".format(mask_type))
            # XXX Do this smarter
            mask_files = [
                os.path.join(mask_root, "mask_map_{}.fits".format(tag))
                for tag in map_tags
            ]
        for f in mask_files:
            if not os.path.exists(f):
                raise OSError("Missing mask file {}".format(f))
        self.log("Found {} masks in {}".format(len(mask_files), mask_root), "task")
        self.log("Mask files: {}".format(mask_files), "detail")

        # Also need a list of unique map tags for populating dictionaries
        # in data structures
        map_tags_orig = list(map_tags)  # copy
        map_tags = pt.unique_tags(map_tags)

        # make a list of names corresponding to the order of the cross spectra
        map_pairs = pt.tag_pairs(map_tags)

        # make a dictionary of map freqs for each unique map tag
        map_freqs_dict = {}
        for im0, m0 in enumerate(map_tags):
            map_freqs_dict[m0] = map_freqs[im0]
        map_freqs = map_freqs_dict

        nom_freqs_dict = {}
        for im0, m0 in enumerate(map_tags):
            nom_freqs_dict[m0] = nom_freqs[im0]
        nom_freqs = nom_freqs_dict

        # dict of reobs frequencies, for if planck is reobs
        map_reobs_freqs = {}
        for im0, m0 in enumerate(map_tags):
            if planck_reobs and nom_freqs[m0] in self.planck_freqs:
                map_reobs_freqs[m0] = "150a"
            else:
                map_reobs_freqs[m0] = map_tags_orig[im0]

        fields = [
            "data_root",
            "data_subset",
            "map_root",
            "map_files",
            "map_tags",
            "map_pairs",
            "map_tags_orig",
            "map_freqs",
            "nom_freqs",
            "map_reobs_freqs",
            "raw_root",
            "raw_files",
            "signal_root",
            "signal_files",
            "signal_root_sim",
            "signal_files_sim",
            "signal_transfer_root",
            "signal_transfer_files",
            "noise_root",
            "noise_files",
            "noise_root_sim",
            "noise_files_sim",
            "mask_root",
            "mask_files",
            "foreground_root",
            "foreground_files",
        ]
        out = dict()
        local = locals()
        for f in fields:
            out[f + suffix] = local[f]
        return out

    def get_files(
        self,
        data_root,
        data_subset="full/*0",
        signal_subset="*",
        noise_subset="*",
        clean_type="raw",
        noise_type="stationary",
        noise_type_sim=None,
        mask_type="hitsmask_tailored",
        signal_type="r0p03",
        signal_type_sim=None,
        signal_transfer_type=None,
        data_root2=None,
        data_subset2=None,
        foreground_type=None,
        template_type=None,
        sub_planck=False,
        planck_reobs=True,
    ):
        """
        Find all files for the given data root.  The data structure is:

            <data_root>
                -> data_<clean_type>
                    -> full
                        -> map_x1.fits
                        ...
                        -> map_150.fits
                        -> map_90.fits
                    -> 1of4 (same filenames as full)
                    -> 2of4 ('')
                    -> 3of4 ('')
                    -> 4of4 ('')
                -> signal_<signal_type>
                   -> spec_signal_<signal_type>.dat
                   -> full
                      -> map_x1_0000.fits
                      ...
                      -> map_90_####.fits
                   -> 1of4 (same filenames as full)
                   -> 2of4 (same filenames as full)
                   -> 3of4 (same filenames as full)
                   -> 4of4 (same filenames as full)
                -> noise_<noise_type> (same filenames as signal_<signal_type>)
                -> masks_<mask_type>
                    -> mask_map_x1.fits
                    ...
                    -> mask_map_90.fits
                    -> mask_map_150.fits
                -> foreground_<foreground_type>
                    (same filenames as signal_<signal_type>)
                -> templates_<template_type>
                   -> halflmission-1
                      (same filenames as data_<clean_type>)
                   -> halflmission-2
                      (same filenames as data_<clean_type>)
                -> reobs_planck (if sub_planck=True)
                   -> halfmission-1
                      (same filenames as data_<clean_type>)
                   -> halfmission-2
                      (same filenames as data_<clean_type>)

            <data_root2>
                ...

        Arguments
        ---------
        data_root : string
            Top level path containing subdirectories for data, signal sims,
            noise sims, and masks.
        data_subset : string
            Subset of maps to use for spectrum estimation.  This should be
            a string that is parseable using `glob` on the path
            `data_<data_type>/<data_subset>.fits`.  For example,
            'full/*0' will expand to read in the 150 GHz and 90GHz maps.
            Maps are then sorted in alphabetical order, and identified
            by their file tag, where each filename is `map_<tag>.fits`.
        signal_subset : string
            Subset of map tags to use for spectrum estimation for signal
            sims.  This should be a string that is parseable using `glob`
            that is added onto the data_subset path to indicate which sims
            to use. For example, for all, use '*'. For the first 300 sims,
            use '0[0-2]*'.
        noise_subset : string
            Subset of map tags to use for spectrum estimation for noise
            sims.  This should be a string that is parseable using `glob`
            that is added onto the data_subset path to indicate which sims
            to use. For example, for all, use '*'. For the first 300 sims,
            use '0[0-2]*'.
        clean_type : string
            The type of data to use.  If this is 'raw' (default), then
            the directory contains raw output maps from a `unimap` run.
            Other tags indicate the type of foreground cleaning
            performed on the data.
        noise_type: string
            The variant of noise simulation to use, e.g. 'stationary',
            'variable', etc.  The directory should contain the same number
            of simulations for each map tag.
        noise_type_sim : string
            The variant of noise sims to use for sim_index fake data map.
            This enables having a different noise sim ensemble to use for
            sim_index run than the ensemble from which the noise is computed.
        mask_type : string
            The variant of mask to use, e.g. 'hitsmask', etc.
            XXX: for the time being we assume a mask per file tag,
            rather than a mask per file in `data_<data_type>`.
        signal_type : string
            The variant of signal simulation to use, typically identified
            by the input spectrum model used to generate it, e.g 'r0p03'.
        signal_type_sim : string
            The variant of signal sims to use for sim_index fake data map.
            This enables having a different noise sim ensemble to use for
            sim_index run than the ensemble from which the signal is computed.
        signal_transfer_type : string
            The variant of signal simulation to use for transfer function
            calculation, typically identified by the input spectrum model
            used to generate it, e.g 'r0p03'. This directory may also contain
            a copy of the input spectrum, to make sure that the correct
            spectrum is used to compute the transfer function.
        data_root2, data_subset2 : string
            The root and subset for a second set of data.
            If either of these is keywords is supplied, then the two data
            sets are treated as two halves of a null test.  In this case,
            XFaster computes the sum and difference spectra for each map
            tag in order to estimate a null spectrum.
        foreground_type : string
            Tag for directory (foreground_<foreground_type>) where foreground
            sims are that should be added to the signal and noise sims
            when running in sim_index mode. Note: the same foreground sim
            map is used for each sim_index, despite signal and noise sims
            changing.
        template_type : string
            Tag for directory (templates_<template_type>) containing templates
            (e.g. a foreground model) to be scaled by a scalar value per
            map tag and subtracted from the data. The directory contains one
            template per map tag.
        sub_planck : bool
            If True, subtract reobserved Planck from maps. Properly uses half
            missions so no Planck autos are used. Useful for removing expected
            signal residuals from null tests. Maps are expected to be in
            reobs_planck directory
        planck_reobs : bool
            If True, input Planck maps have been reobserved with 150a Bl/Fl.
            Else, input Planck maps have only been smoothed to common 15'
            beam

        Returns
        -------
        file_settings : dict
            A dictionary of file settings used throughout the run.
            These are stored in full as `<output_root>/files_<output_tag>.npz`,
            and a subset are added to the run configuration file
            `<output_root>/config_<output_tag>.txt`.
        """

        if signal_transfer_type is None:
            signal_transfer_type = signal_type
        # one of these must be set to do a null test
        null_run = False
        if data_root2 is not None or data_subset2 is not None:
            if data_root2 is None:
                data_root2 = data_root
            if data_subset2 is None:
                data_subset2 = data_subset
            if (data_root, data_subset) == (data_root2, data_subset2):
                raise ValueError(
                    "Either data_root2 or data_subset2 must differ "
                    "from data_root/data_subset"
                )
            null_run = True

        opts = dict(
            clean_type=clean_type,
            noise_type=noise_type,
            noise_type_sim=noise_type_sim,
            mask_type=mask_type,
            signal_type=signal_type,
            signal_type_sim=signal_type_sim,
            signal_transfer_type=signal_transfer_type,
            signal_subset=signal_subset,
            noise_subset=noise_subset,
            foreground_type=foreground_type,
            planck_reobs=planck_reobs,
        )
        ref_opts = dict(data_subset=data_subset, **opts)
        if null_run:
            ref_opts.update(data_subset2=data_subset2)

        no_return = [
            "signal_files",
            "signal_files_sim",
            "signal_transfer_files",
            "noise_files",
            "noise_files_sim",
            "signal_files2",
            "signal_files_sim2",
            "signal_transfer_files2",
            "noise_files2",
            "noise_files_sim2",
            "foreground_files",
            "foreground_files2",
        ]

        def get_template_files(fs, template_type):
            """
            Update options for template cleaning
            """
            # no template fitting for null runs
            if fs["null_run"]:
                template_type = None

            if "template_type" in fs:
                if template_type == fs["template_type"]:
                    return

            fs["template_type"] = template_type

            # find all corresponding foreground templates
            if template_type is None:
                fs["template_root"] = None
                fs["template_root2"] = None
                fs["template_files"] = None
                fs["template_files2"] = None
                fs["template_noise_root"] = None
                fs["template_noise_root2"] = None
                fs["template_noise_files"] = None
                fs["template_noise_files2"] = None
                fs["num_template"] = 0
                fs["num_template_noise"] = 0
            else:
                num_template_noise = None
                for hm in ["1", "2"]:
                    suff = "" if hm == "1" else "2"
                    troot = os.path.join(
                        fs["data_root"],
                        "templates_{}".format(template_type),
                        "halfmission-{}".format(hm),
                    )
                    ### this block is so sims with template type like
                    # 353_100_gauss_003 can use ensemble in 353_100_gauss
                    tp = template_type.split("_")
                    ttype = template_type
                    if tp[-1].isdigit():
                        if ttype[-7:] not in ["353_100", "217_100"]:
                            ttype = "_".join(tp[:-1])

                    tnroot = os.path.join(
                        fs["data_root"],
                        "templates_noise_{}".format(ttype),
                        "halfmission-{}".format(hm),
                    )

                    tfiles = []
                    tnfiles = []
                    for f in fs["map_files"]:
                        nfile = f.replace(fs["map_root"], troot)
                        if not os.path.exists(nfile):
                            raise OSError("Missing hm-{} template for {}".format(hm, f))
                        tfiles.append(nfile)
                        nfiles = sorted(
                            glob.glob(
                                f.replace(fs["map_root"], tnroot).replace(
                                    ".fits", "_*.fits"
                                )
                            )
                        )
                        if not len(nfiles):
                            raise OSError(
                                "Missing hm-{} template noise for {}".format(hm, f)
                            )
                        tnfiles.append(nfiles)
                        if num_template_noise is not None:
                            if len(nfiles) != num_template_noise:
                                self.log(
                                    "num template noise: {}".format(num_template_noise)
                                )
                                self.log("nfiles: {}".format(len(nfiles)))
                                raise OSError("Wrong number of template noise sims")

                        num_template_noise = len(nfiles)

                    tfiles = np.asarray(tfiles)
                    tnfiles = np.asarray(tnfiles)
                    fs["template_root{}".format(suff)] = troot
                    fs["template_files{}".format(suff)] = tfiles
                    fs["template_noise_root{}".format(suff)] = tnroot
                    fs["template_noise_files{}".format(suff)] = tnfiles

                fs["num_template"] = len(fs["template_files"])
                fs["num_template_noise"] = num_template_noise
                self.log(
                    "Found {} templates in {}".format(
                        fs["num_template"], fs["template_root"]
                    ),
                    "task",
                )
                self.log(
                    "Found {} template noise files in {}".format(
                        fs["num_template_noise"], fs["template_noise_root"]
                    ),
                    "task",
                )
                self.log("Template files: {}".format(fs["template_files"]), "detail")

            fields = [
                "template_type",
                "template_root",
                "template_root2",
                "template_files",
                "template_files2",
                "template_noise_root",
                "template_noise_root2",
                "template_noise_files",
                "template_noise_files2",
                "num_template",
                "num_template_noise",
            ]
            for k in fields:
                setattr(self, k, fs[k])

        def get_planck_files(fs, sub_planck=False):
            """
            Update options for planck subtraction
            """
            if not sub_planck:
                fs["planck_root1_hm1"] = None
                fs["planck_root2_hm1"] = None
                fs["planck_root1_hm2"] = None
                fs["planck_root2_hm2"] = None
                fs["planck_files1_hm1"] = None
                fs["planck_files2_hm1"] = None
                fs["planck_files1_hm2"] = None
                fs["planck_files2_hm2"] = None
                fs["num_planck"] = 0
            else:
                for null_split in [1, 2]:
                    if null_split == 1:
                        suff = ""
                    else:
                        suff = 2
                    for hm in ["1", "2"]:
                        fs["num_planck"] = 0
                        proot = os.path.join(
                            fs["data_root{}".format(suff)],
                            "reobs_planck",
                            "halfmission-{}".format(hm),
                        )
                        pfiles = []
                        for f in fs["map_files{}".format(suff)]:
                            nfile = f.replace(fs["map_root{}".format(suff)], proot)
                            if not os.path.exists(nfile):
                                raise OSError("Missing hm-{} map for {}".format(hm, f))
                            pfiles.append(nfile)
                            fs["num_planck"] += 1
                        pfiles = np.asarray(pfiles)
                        fs["planck_root{}_hm{}".format(null_split, hm)] = proot
                        fs["planck_files{}_hm{}".format(null_split, hm)] = pfiles

                        self.log(
                            "Found {} planck maps in {}".format(
                                fs["num_planck"],
                                fs["planck_root{}_hm{}".format(null_split, hm)],
                            ),
                            "task",
                        )
                        self.log("Planck files: {}".format(pfiles), "detail")

            fields = [
                "planck_root1_hm1",
                "planck_root2_hm1",
                "planck_root1_hm2",
                "planck_root2_hm2",
                "planck_files1_hm1",
                "planck_files2_hm1",
                "planck_files1_hm2",
                "planck_files2_hm2",
                "num_planck",
            ]
            for k in fields:
                setattr(self, k, fs[k])

        save_name = "files"
        alt_name = None
        if clean_type != "raw":
            alt_name = save_name
            save_name = "{}_{}".format(save_name, clean_type)
        if template_type is not None:
            alt_name = save_name
            save_name = "{}_clean_{}".format(save_name, template_type)
        if sub_planck:
            alt_name = save_name
            save_name = "{}_planck_sub".format(save_name)
        # load file info from disk
        if signal_type_sim is not None or noise_type_sim is not None:
            ret = None
            save = False
        else:
            ret = self.load_data(
                save_name, "files", to_attrs=True, value_ref=ref_opts, alt_name=alt_name
            )
            save = True
        if ret is not None:
            # fix data root
            ret_data_root = ret["data_root"]
            ret_data_root2 = ret.get("data_root2", data_root2)
            if data_root == ret_data_root and (
                not null_run or data_root2 == ret_data_root2
            ):
                if template_type != ret.get("template_type", None):
                    get_template_files(ret, template_type)
                    self.save_data(save_name, **ret)
                if sub_planck and ret.get("planck_root1", None) is None:
                    get_planck_files(ret, sub_planck)
                    self.save_data(save_name, **ret)
                for k in no_return:
                    ret.pop(k, None)
                return ret

            def replace_root(k, v):
                if not isinstance(v, str):
                    return v
                if null_run and ret_data_root2 != ret_data_root:
                    if k.endswith("2") and v.startswith(ret_data_root2):
                        return v.replace(ret_data_root2, data_root2)
                if v.startswith(ret_data_root):
                    return v.replace(ret_data_root, data_root)
                return v

            for k, v in ret.items():
                if isinstance(v, str):
                    ret[k] = replace_root(k, v)
                    setattr(self, k, ret[k])
                elif isinstance(v, np.ndarray) and isinstance(v[0], str):
                    ret[k] = np.array(
                        [replace_root(k, vv) for vv in v.ravel()]
                    ).reshape(v.shape)
                    setattr(self, k, ret[k])
                if k in no_return:
                    ret.pop(k)

            if template_type != ret.get("template_type", None):
                get_template_files(ret, template_type)
                self.save_data(save_name, **ret)

            if sub_planck and ret.get("planck_root1", None) is None:
                get_planck_files(ret, sub_planck)
                self.save_data(save_name, **ret)
            return ret

        # find all map files
        fs = self._get_files(data_root, data_subset, **opts)
        fs.update(**opts)
        # count all the things
        fs["num_maps"] = len(fs["map_files"])
        fs["num_signal"] = len(fs["signal_files"][0])
        fs["num_signal_transfer"] = len(fs["signal_transfer_files"][0])
        fs["num_noise"] = (
            len(fs["noise_files"][0]) if fs["noise_files"] is not None else 0
        )
        fs["num_noise_sim"] = (
            len(fs["noise_files_sim"][0]) if fs["noise_files_sim"] is not None else 0
        )
        fs["num_signal_sim"] = (
            len(fs["signal_files_sim"][0]) if fs["signal_files_sim"] is not None else 0
        )
        fs["num_foreground"] = (
            len(fs["foreground_files"][0]) if fs["foreground_files"] is not None else 0
        )
        fs["num_corr"] = pt.num_corr(fs["num_maps"])
        fs["num_spec"] = pt.num_corr(3 if self.pol else 1)
        fs["num_spec_mask"] = pt.num_corr(2 if self.pol_mask else 1)
        fs["num_kern"] = fs["num_corr"]
        fs["data_shape"] = (fs["num_corr"] * fs["num_spec"], self.lmax + 1)
        fs["mask_shape"] = (fs["num_corr"] * fs["num_spec_mask"], self.lmax + 1)
        fs["kern_shape"] = (fs["num_kern"] * (self.lmax + 1), 2 * self.lmax + 1)
        fs["null_run"] = null_run

        if null_run:
            # find all map files for null tests
            fs2 = self._get_files(data_root2, data_subset2, suffix="2", **opts)
            # check for matching number of maps
            n = len(fs2["map_files2"])
            if n != fs["num_maps"]:
                raise RuntimeError(
                    "Found {} map2 files, expected {}".format(n, fs["num_maps"])
                )
            # XXX make sure tags match?

            # check for matching number of sims
            n = len(fs2["signal_files2"][0])
            if n != fs["num_signal"]:
                raise RuntimeError(
                    "Found {} signal2 sims, expected {}".format(n, fs["num_signal"])
                )
            n = len(fs2["signal_transfer_files2"][0])
            if n != fs["num_signal_transfer"]:
                raise RuntimeError(
                    "Found {} signal2 transfer sims, expected {}".format(
                        n, fs["num_signal_transfer"]
                    )
                )
            if fs["noise_files"] is not None:
                n = len(fs2["noise_files2"][0])
                if n != fs["num_noise"]:
                    raise RuntimeError(
                        "Found {} noise2 sims, expected {}".format(n, fs["num_noise"])
                    )
            if fs["noise_files_sim"] is not None:
                n = len(fs2["noise_files_sim2"][0])
                if n != fs["num_noise_sim"]:
                    raise RuntimeError(
                        "Found {} noise_sim2 sims, expected {}".format(
                            n, fs["num_noise_sim"]
                        )
                    )
            if fs["signal_files_sim"] is not None:
                n = len(fs2["signal_files_sim2"][0])
                if n != fs["num_signal_sim"]:
                    raise RuntimeError(
                        "Found {} signal_sim2 sims, expected {}".format(
                            n, fs["num_signal_sim"]
                        )
                    )
            if fs["foreground_files"] is not None:
                n = len(fs2["foreground_files2"][0])
                if n != fs["num_foreground"]:
                    raise RuntimeError(
                        "Found {} foreground2 sims, expected {}".format(
                            n, fs["num_foreground"]
                        )
                    )
            # XXX make sure sim numbers match?

            # we're doing a null test
            fs.update(null_run=True, **fs2)

        get_template_files(fs, template_type)
        get_planck_files(fs, sub_planck)

        # store and return settings dictionary
        if save:
            self.save_data(save_name, **fs)
        for k in list(fs):
            setattr(self, k, fs[k])
            if k in no_return:
                fs.pop(k)
        return fs

    def get_map(self, filename, check_nside=True, cache=False, **kwargs):
        """
        Load an input map from file or from an internal cache.  Maps are
        checked to make sure they all have a consistent size, and optionally
        cached to limit disk I/O.

        Arguments
        ---------
        filename : string
            Path to file on disk.
        check_nside : bool
            If True (default), make sure that all maps have the same `nside`,
            and that it satisfies `lmax <= 4 * nside`.
        cache : bool
            If True, cache the map in memory to avoid rereading from disk.
            Use this for maps that are used multiple times by the algoritm
            (e.g. masks).

        Any remaining arguments are passed to `healpy.read_map`.

        Returns
        -------
        map : array_like
            2D map array containing 1 (T) or 3 (T/Q/U) maps.
            If the XFaster class was initialized with `pol = True`, this
            returns a 2D array of T/Q/U maps from the file. Otherwise a
            (1, npix) array is returned containing only the T map.
        """

        # initialize map cache
        if not hasattr(self, "_map_cache"):
            self._map_cache = dict()
        if kwargs.pop("reset", False):
            self._map_cache.pop(filename, None)

        # return a copy from cache if found
        if cache and filename in self._map_cache:
            return np.copy(self._map_cache[filename])

        kwargs.setdefault("field", [0, 1, 2] if self.pol else [0])

        self.log("Reading map from {}".format(filename), "all")
        m = np.atleast_2d(hp.read_map(filename, **kwargs))
        m[hp.mask_bad(m)] = 0

        if check_nside:

            if not hasattr(self, "nside"):
                self.nside = None

            # check nside
            nside = hp.get_nside(m)
            if self.nside is None:
                self.nside = nside
            else:
                if nside != self.nside:
                    raise ValueError(
                        "Input map {} has nside {} expected {}".format(
                            filename, nside, self.nside
                        )
                    )

            # check npix
            npix = hp.nside2npix(nside)
            if getattr(self, "npix", None) is None:
                self.npix = npix
            else:
                if npix != self.npix:
                    raise ValueError(
                        "Input map {} has npix {} expected {}".format(
                            filename, npix, self.npix
                        )
                    )

            # check lmax
            if self.lmax is None:
                self.lmax = 4 * self.nside
            elif self.lmax > 4 * self.nside:
                warnings.warn(
                    "lmax {} may be too large for nside {}".format(
                        self.lmax, self.nside
                    )
                )

        if cache:
            self._map_cache[filename] = m
        return m

    def get_mask(self, filename, cache=True, check_lims=True, **kwargs):
        """
        Load an input mask from file or from an internal cache.
        See `XFaster.get_map` for details.

        Arguments
        ---------
        filename : string
            Path to mask file on disk.
        cache : bool
            This option defaults to True, since masks are typically used
            for all data and sims for a given map tag.
        check_lims : bool
            If True, values in the mask outside of [0,1] are fixed to
            these limits.

        Any remaining arguments are passed to `XFaster.get_map`.

        Returns
        -------
        mask : array_like
            2D array containing 1 (T) or 2 (T/P) maps;  If the XFaster class
            was initialized with `pol_mask = True`, this returns a 2D array
            containing both T and P masks.  Otherwise, a (1, npix) is
            returned containing only the T map.
        """

        fields_mask = [0, 1] if self.pol_mask else [0]
        kwargs.setdefault("field", fields_mask)
        m = self.get_map(filename, cache=cache, **kwargs)
        if check_lims:
            m[m < 0] = 0
            m[m > 1] = 1
        return m

    def get_filename(
        self,
        name,
        ext=".npz",
        map_tag=None,
        iter_index=None,
        extra_tag=None,
        bp_opts=False,
    ):
        """
        Define a standard output file path to read or write.

        Arguments
        ---------
        name : string
            String name of output type.  E.g. 'data_xcorr' for data
            cross-correlation spectra.
            If an output tag is set, the name is appended with
            '_<output_tag>'.
        ext : string
            File extension.  The default ('.npz') is used for storing
            output data dictionaries.
        map_tag : string
            If supplied, the name is appended with '_map_<map_tag>'.
            Use this argument when storing output data in a loop over
            input maps.
        iter_index : int
            If supplied, the name is appended with '_iter<iter_index>'
        extra_tag : string
            If supplied the extra tag is appended to the name as is.
        bp_opts : bool
            If True, the output filename is constructed  by checking the
            following list  of options used in constructing bandpowers:
            ensemble_mean, ensemble_median, sim_index, template_cleaned,
            weighted_bins, signal_type_sim, noise_type_sim

        Returns
        -------
        filename : string
            Output filename as `<output_root>/<name><ext>`, where
            <name> can optionally include the map index, iteration index
            or output tag.
        """
        if self.output_root is None:
            return None

        if bp_opts:
            if self.ensemble_mean:
                name = "{}_mean".format(name)
            elif self.ensemble_median:
                name = "{}_median".format(name)
            elif self.sim_index is not None:
                name = "{}_sim{:04d}".format(name, self.sim_index)
                if self.signal_type_sim:
                    name = "{}_{}".format(name, self.signal_type_sim)
                if self.noise_type_sim:
                    name = "{}_{}".format(name, self.noise_type_sim)
            else:
                if self.clean_type != "raw":
                    name = "{}_{}".format(name, self.clean_type)
                if getattr(self, "template_cleaned", False):
                    name = "{}_clean_{}".format(name, self.template_type)
                if getattr(self, "planck_sub", False):
                    name = "{}_planck_sub".format(name)
            if self.weighted_bins:
                name = "{}_wbins".format(name)
            if self.return_cls:
                name = "{}_cl".format(name)

        if map_tag is not None:
            name = "{}_map_{}".format(name, map_tag)
        if iter_index is not None:
            name = "{}_iter{:03d}".format(name, iter_index)
        if extra_tag is not None:
            name = "{}_{}".format(name, extra_tag)

        tag = "_{}".format(self.output_tag) if self.output_tag else ""
        if not ext.startswith("."):
            ext = ".{}".format(ext)
        return os.path.join(self.output_root, "{}{}{}".format(name, tag, ext))

    def load_data(
        self,
        name,
        checkpoint,
        fields=None,
        to_attrs=True,
        shape=None,
        shape_ref=None,
        alt_name=None,
        value_ref=None,
        optional=None,
        **file_opts
    ):
        """
        Load xfaster data from an output npz file on disk.

        This method is called throughout the code at various checkpoints.
        If the data exist on disk, they are loaded and returned.  If the
        data are missing or otherwise incompatible, they are recomputed
        by the calling method, and trigger all subsequent data to also be
        recomputed.  Methods that use this functionality contain a
        `Data Handling` section in their docstring.

        Arguments
        ---------
        name : string
            The name of the data set.  The filename is contructed from this
            as `<output_root>/<name>_<output_tag>.npz`.  If the file is not
            found then the data are recomputed.
        checkpoint : string
            The name of the checkpoint to which this dataset applies.
            If XFaster is initialized at this checkpoint, or if any of the
            file checks enabled with the following options fails, all
            quantities from this point forward are recomputed.
        fields : list of strings
            List of fields that should be present in the data file.
            If any are not found, the entire dataset and all subsequent
            step are recomputed.
        to_attrs : bool or list of bools or strings
            If True, all items in `fields` are stored as attributes of the
            parent object.  If A list of booleans, must have the same length
            as `fields`; any field for which this list item is True is then
            stored as an attribute of the object.  If any list item is a string,
            then the corresponding field is stored as an attribute with this
            new name.
        shape : tuple of ints
            If set, the field specified by `shape_ref` is checked to have this
            shape.  If this check fails, then all data are recomputed.
        shape_ref : string
            The reference field whose shape is checked against `shape`.
            If None and `shape` is set, use the first field in `fields`.

        Remaining options are passed to `get_filename` for constructing the
        output file path.

        Returns
        -------
        data : dict
            If all checks above succeed, the requested data are returned.
            If any tests fail, None is returned, and all subsequent calls
            to `load_data` also return None to trigger recomputing all data
            that may depend on this dataset.
            The output dictionary has the additional key 'output_file' which
            is set to the path to the data file on disk.
        """

        # checkpointing
        def force_rerun_children():
            """Trigger rerunning steps that depend on this checkpoint."""
            for step in self.checkpoint_tree.get(checkpoint, []):
                if step not in self.checkpoints:
                    raise ValueError(
                        "Invalid checkpoint {}, must be one of {}".format(
                            step, self.checkpoints
                        )
                    )
                self.force_rerun[step] = True
            return None

        if checkpoint not in self.checkpoints:
            raise ValueError(
                "Invalid checkpoint {}, must be one of {}".format(
                    checkpoint, self.checkpoints
                )
            )

        if self.checkpoint == checkpoint:
            self.force_rerun[checkpoint] = True
        if self.force_rerun[checkpoint]:
            return force_rerun_children()

        use_alt = False
        output_file = self.get_filename(name, ext=".npz", **file_opts)
        if not output_file:
            return force_rerun_children()
        errmsg = "Error loading {}".format(output_file)
        if not os.path.exists(output_file):
            self.log("{}: Output file not found".format(errmsg))
            if alt_name is not None:
                output_file = self.get_filename(alt_name, ext=".npz", **file_opts)
                errmsg = "Error loading {}".format(output_file)
                if not os.path.exists(output_file):
                    self.log("{}: Alternate output file not found".format(errmsg))
                    return force_rerun_children()
                else:
                    use_alt = True
            else:
                return force_rerun_children()

        try:
            data = pt.load_and_parse(output_file)
        except Exception as e:
            self.log("{}: {}".format(errmsg, str(e)))
            return force_rerun_children()

        if fields is None:
            fields = list(data)

        if not isinstance(fields, list):
            fields = [fields]

        if shape_ref is not None and shape_ref not in fields:
            self.log("{}: Field {} not found".format(errmsg, shape_ref))
            return force_rerun_children()

        if to_attrs is True or to_attrs is False:
            to_attrs = [to_attrs] * len(fields)

        if shape is not None and shape_ref is None:
            shape_ref = fields[0]

        if value_ref is not None:
            value_ref = copy.deepcopy(value_ref)

        ret = dict()
        for field, attr in zip(fields, to_attrs):
            if field not in data:
                if optional is not None and field in optional:
                    data[field] = None
                else:
                    self.log("{}: Field {} not found".format(errmsg, field))
                    return force_rerun_children()
            v = pt.dict_to_arr(data[field])
            try:
                v.shape
            except AttributeError:
                v = np.asarray(v)
            if not v.shape:
                v = v.tolist()
            if shape_ref in [field, attr]:
                if v.shape != tuple(shape):
                    self.log(
                        "{}: Field {} has shape {}, expected {}".format(
                            errmsg, shape_ref, v.shape, shape
                        )
                    )
                    return force_rerun_children()
            if value_ref is not None:
                for k in [field, attr]:
                    vref = value_ref.pop(k, "undef")
                    if not vref == "undef" and np.any(v != vref):
                        self.log(
                            "{}: Field {} has value {}, expected {}".format(
                                errmsg, k, v, vref
                            )
                        )
                        return force_rerun_children()
            ret[field] = pt.parse_data(data, field)
            if attr:
                key = field if attr is True else attr
                setattr(self, key, ret[field])

        if value_ref:
            self.log(
                "{}: Missing reference fields {}".format(errmsg, list(value_ref.keys()))
            )
            return force_rerun_children()

        self.log("Loaded input data from {}".format(output_file), "detail")
        if use_alt:
            # copy data to original file name
            ret.update(**file_opts)
            self.save_data(name, **ret)
            for k in file_opts:
                ret.pop(k)
        ret["output_file"] = output_file
        return ret

    def save_data(self, name, from_attrs=[], **data):
        """
        Save xfaster data to an output npz file on disk.

        Arguments
        ---------
        name : string
            The name of the data set.  The filename is contructed from this as
            `<output_root>/<name>_<output_tag>.npz`.  If the file is not found
            then the data are recomputed.
        from_attrs : list of strings
            A list of object attributes which should be stored in the data file.
        map_tag : str
            Load the dataset corresponding to this map.
            See `get_filename` for documentation.
        iter_index : int
            Load the dataset corresponding to this iteration index.
            See `get_filename` for documentation.
        bp_opts : bool
            Format output bandpowers file.  See `get_filename` for documentation.

        Any remaining keyword arguments are added to the output dictionary.

        Returns
        -------
        data : dict
            A copy of the data dictionary that was stored to disk.
            The output dictionary has the additional key 'output_file' which
            is set to the path to the data file on disk.
        """
        data["data_version"] = 1

        file_opts = {}
        for opt in ["map_tag", "iter_index", "bp_opts", "extra_tag"]:
            if opt in data:
                file_opts[opt] = data.pop(opt)

        output_file = self.get_filename(name, ext=".npz", **file_opts)
        if not output_file:
            return

        for attr in from_attrs:
            if hasattr(self, attr):
                data[attr] = getattr(self, attr)

        np.savez_compressed(output_file, **data)
        self.log("Saved output data to {}".format(output_file), "detail")
        data["output_file"] = output_file
        return data

    def save_config(self, cfg):
        """
        Save a configuration file for the current run on disk.
        This method is used by `xfaster_run` to store the config
        in `<output_root>/config_<output_tag>.txt`.
        """
        filename = self.get_filename("config", ext=".txt")
        if filename is None:
            return

        if not isinstance(cfg, base.XFasterConfig):
            cfg = base.XFasterConfig(cfg)

        try:
            creator = os.getlogin()
        except OSError:
            creator = "unknown"
        with open(filename, "w") as f:
            f.write(
                "# Created by {} on {:%Y-%m-%d %H:%M:%S}\n\n".format(
                    creator, datetime.datetime.now()
                )
            )
            cfg.write(f)

        return filename

    def apply_mask(self, m, mask):
        """
        Apply the input mask to the data map, in place.

        If the map is polarized, the appropriate mask is applied
        to the polarization data, depending on whether the mask
        is also polarized.

        Arguments
        ---------
        m : array_like
            Input map (T/Q/U if polarized, T-only if not)
            This array is modified in place.
        mask : array_like
            Mask to apply (T/P if polarized, T-only if not)
        """

        m[0] *= mask[0]
        if self.pol:
            m[1:] *= (mask[1] if self.pol_mask else mask[0])[None, :]

        return m

    def map2alm(self, m, pol=None):
        """
        Wrapper for healpy.map2alm.

        Arguments
        ---------
        m : array_like
            Masked input map for which Alms are computed.
        pol : bool
            If None, this is set using the value with which the object
            was initialized.

        Returns
        -------
        alms : array_like
            Alms for the input map, computed using the equivalent of
            `healpy.map2alm(m, lmax, pol=self.pol, use_weights=True)`.
        """
        if pol is None:
            pol = self.pol
        return np.asarray(hp.map2alm(m, self.lmax, pol=pol, use_weights=True))

    def alm2cl(self, m1, m2=None, lmin=2, lmax=None, symmetric=True):
        """
        Wrapper for healpy.alm2cl.

        Arguments
        ---------
        m1 : array_like
            Masked alms for map1
        m2 : array_like
            Masked alms for map2
        lmin : int
            The minimum ell bin to include in the output Cls.  All ell
            bins below this are nulled out.
        lmax : int
            The maximum ell bin to compute.  If None, this is set to the
            lmax value with which the class was initialized.
        symmetric : bool
            If True, the average cross spectrum of (m1-x-m2 + m2-x-m1) / 2.
            is computed.

        Returns
        -------
        cls : array_like
            Cross-spectrum of m1-x-m2.
        """
        if lmax is None:
            lmax = self.lmax
        cls = np.asarray(hp.alm2cl(m1, alms2=m2, lmax=lmax))
        if symmetric:
            cls_T = np.asarray(hp.alm2cl(m2, alms2=m1, lmax=lmax))
            cls = (cls + cls_T) / 2.0
        if lmin:
            cls[..., :lmin] = 0
        return cls

    def get_mask_weights(self, apply_gcorr=False, reload_gcorr=False):
        """
        Compute cross spectra of the masks for each data map.

        Mode counting matrices are also computed and stored for each mask.

        Arguments
        ---------
        apply_gcorr : bool
            If True, a correction factor is applied to the g (mode counting)
            matrix.  The correction factor should have been pre-computed for
            each map tag.
        reload_gcorr : bool
            If True, reload the gcorr file from the masks directory. Useful when
            iteratively solving for the correction terms.

        Data Handling
        -------------
        This method is called at the 'masks' checkpoint, loads or saves a
        data dictionary with the following keys:

           wls : (num_map_corr, num_pol_mask_corr, lmax + 1)
               mask1-x-mask2 mask cross spectra for every mask pair
           fsky, w1, w2, w4 : (num_map_corr, num_pol_mask_corr)
               sky fraction and weighted modes per mask product
           gmat : (num_maps * num_pol_mask_corr, ) * 2
               mode-counting matrix, computed from
                   g = fsky * w2 ** 2 / w4

        Where the dimensions of each item are determined from:

           num_map_corr : Nmap * (Nmap + 1) / 2
               number of map-map correlations
           num_pol_mask_corr : 3 (TT, TP, PP) if pol else 1 (TT)
               number of mask spectrum correlations
        """

        mask_files = self.mask_files
        num_maps = self.num_maps
        mask_shape = self.mask_shape
        save_attrs = ["wls", "fsky", "w1", "w2", "w4", "gmat", "nside", "npix", "gcorr"]
        save_name = "masks_xcorr"
        ret = self.load_data(
            save_name,
            "masks",
            fields=save_attrs,
            optional=["gcorr"],
            to_attrs=True,
            shape=mask_shape,
            shape_ref="wls",
            alt_name="data_xcorr",
        )

        def process_gcorr():
            if not hasattr(self, "gcorr"):
                self.gcorr = None
            if apply_gcorr and self.gcorr is None:
                self.gcorr = OrderedDict()

            for tag, mfile in zip(self.map_tags, self.mask_files):
                if not apply_gcorr:
                    continue

                if not reload_gcorr and tag in self.gcorr:
                    continue

                gcorr_file = mfile.replace(".fits", "_gcorr.npz")
                if not os.path.exists(gcorr_file):
                    warnings.warn("G correction file {} not found".format(gcorr_file))
                    continue
                gdata = pt.load_and_parse(gcorr_file)
                gcorr = gdata["gcorr"]
                for k, g in gcorr.items():
                    # check bins match g
                    bd0 = self.bin_def["cmb_{}".format(k)]
                    bd = gdata["bin_def"]["cmb_{}".format(k)]
                    if len(bd0) < len(bd):
                        bd = bd[: len(bd0)]
                        gcorr[k] = g[: len(bd)]
                    if not np.all(bd0 == bd):
                        warnings.warn(
                            "G correction for map {} has incompatible bin def".format(
                                tag
                            )
                        )
                        break
                else:
                    self.log(
                        "Found g correction for map {}: {}".format(tag, gcorr), "detail"
                    )
                    self.gcorr[tag] = gcorr

            # compute ell-by-ell mode counting factor
            gmat_ell = OrderedDict()
            ell = np.arange(self.lmax + 1)
            ellfac = 2.0 * ell + 1.0

            for xname, (m0, m1) in self.map_pairs.items():
                gmat_ell[xname] = OrderedDict()
                if apply_gcorr:
                    gcorr0 = self.gcorr[m0]
                    gcorr1 = self.gcorr[m1]

                for spec in self.specs:
                    gmat_ell[xname][spec] = self.gmat[xname][spec] * ellfac
                    if apply_gcorr:
                        gcorr = np.sqrt(gcorr0[spec] * gcorr1[spec])
                        bd = self.bin_def["cmb_{}".format(spec)]
                        for gc, (start, stop) in zip(gcorr, bd):
                            gmat_ell[xname][spec][start:stop] *= gc

            self.apply_gcorr = apply_gcorr
            self.gmat_ell = gmat_ell

        if ret is not None:
            process_gcorr()
            if apply_gcorr and (reload_gcorr or ret.get("gcorr", None) is None):
                return self.save_data(save_name, from_attrs=save_attrs)
            ret["gcorr"] = self.gcorr
            return ret

        # mask spectra
        wls = OrderedDict()

        pol_dim = 3 if self.pol else 1

        # moments
        fsky = OrderedDict()
        fsky_eff = OrderedDict()
        w1 = OrderedDict()
        w2 = OrderedDict()
        w4 = OrderedDict()
        gmat = OrderedDict()

        cache = dict()

        def process_index(idx):
            if idx in cache:
                return cache[idx]

            self.log("Computing Alms for mask {}/{}".format(idx + 1, num_maps), "all")

            mask = self.get_mask(mask_files[idx])
            mask_alms = self.map2alm(mask, False)

            cache[idx] = (mask_alms, mask)
            return cache[idx]

        spec_inds = {
            "tt": (0, 0),
            "ee": (1, 1),
            "bb": (2, 2),
            "te": (0, 1),
            "eb": (1, 2),
            "tb": (0, 2),
        }

        for xname, (idx, jdx) in pt.tag_pairs(self.map_tags, index=True).items():
            imask_alms, imask = process_index(idx)
            jmask_alms, jmask = process_index(jdx)

            self.log("Computing mask spectra {}x{}".format(idx, jdx), "detail")
            wls[xname] = self.alm2cl(imask_alms, jmask_alms, lmin=0)

            if self.pol_mask:
                # If there is a pol mask in addition to I, copy it to
                # U so no I masks is [I, Q, U] instead of [I, pol]
                imask = np.vstack([imask, imask[1]])
                jmask = np.vstack([jmask, jmask[1]])

            # calculate moments of cross masks
            # this is an array of shape (pol_dim, pol_dim, npix)
            # and contains all combinations of mask products
            mask = np.sqrt(np.einsum("i...,j...->ij...", imask, jmask))
            counts = [np.count_nonzero(x) for x in mask.reshape(-1, mask.shape[-1])]
            counts = np.array(counts).reshape(self.pol_dim, self.pol_dim).astype(float)

            # fsky is the fraction of pixels that are nonzero, independent
            # of weight
            fsky[xname] = counts / self.npix
            w1[xname] = np.sum(mask, axis=-1) / counts
            w2[xname] = np.sum(mask ** 2, axis=-1) / counts
            w4[xname] = np.sum(mask ** 4, axis=-1) / counts
            # effective fsky takes into account weights between 0 and 1
            fsky_eff[xname] = (
                fsky[xname]
                * w2[xname] ** 2
                / w4[xname]
                * (1.0 + 4.0 * fsky[xname])  # second order correction
            )

            # compute gmat as the average fsky_eff assuming symmetrically
            # computed cross spectra, e.g. TE = (T1 * E2 + T2 * E1) / 2
            gmat[xname] = OrderedDict()
            for spec in self.specs:
                si, sj = spec_inds[spec]
                f = (fsky_eff[xname][si, sj] + fsky_eff[xname][sj, si]) / 2.0
                gmat[xname][spec] = f

        if np.any(np.asarray([f for f in fsky.values()]) > 0.1):
            warnings.warn(
                "Some fsky are larger than 10% - second order "
                "correction may break down here {}".format(fsky)
            )

        # store and return
        self.wls = wls
        self.fsky = fsky
        self.w1 = w1
        self.w2 = w2
        self.w4 = w4
        self.gmat = gmat

        process_gcorr()

        self.log("Fsky: {}".format(self.fsky), "detail")
        self.log("Effective fsky: {}".format(fsky_eff), "detail")
        self.log("Mask moments 1: {}".format(self.w1), "detail")
        self.log("Mask moments 2: {}".format(self.w2), "detail")
        self.log("Mask moments 4: {}".format(self.w4), "detail")
        self.log("G matrix: {}".format(self.gmat), "detail")

        return self.save_data(save_name, from_attrs=save_attrs)

    def get_masked_xcorr(
        self,
        template_alpha90=None,
        template_alpha150=None,
        sub_planck=False,
        sub_hm_noise=True,
    ):
        """
        Compute cross spectra of the data maps.

        Map and mask files must have been loaded in by calling the `get_files`
        method with the appropriate file selection options.

        If only one dataset is selected, spectra are computed for every
        combination of pairs of data maps.  This results in N * (N + 1) / 2
        cross spectra for N maps.  A unique mask is used for each input map.

        If two datasets are selected, then sum and difference cross-spectra are
        computed by summing and differencing the two datasets.  A unique mask is
        used for each map in the first dataset, and the same mask is applied to
        the corresponding map in the second dataset, so that both halves are
        masked identically.

        If `template_alpha90` or `template_alpha150` is supplied, it is applied
        to an appropriate template, and the result is subtracted from the data
        alms with nominal frequencies 90 or 150 GHz.  Map alms are cached to
        speed up processing, if this method is called repeatedly with different
        values.

        Data Handling
        -------------
        This method is called at the 'data' checkpoint, loads or saves a data
        dictionary with the following keys:

           cls_data : OrderedDict
               map1-x-map2 cross spectra for every map pair. This contains the
               sum cross spectra if constructing a null test, or the
               template-subtracted cross spectra if `template_alpha90` or
               `template_alpha150` is supplied.
           cls_data_null : OrderedDict
               (map1a-map1b)-x-(map2a-map2b) difference cross spectra
               for every map pair, if computing a null test
        """

        map_tags = self.map_tags
        map_files = self.map_files
        mask_files = self.mask_files
        raw_files = self.raw_files
        num_maps = self.num_maps
        data_shape = self.data_shape
        null_run = self.null_run
        map_files2 = self.map_files2 if null_run else None

        # ignore unused template coefficients
        if not any([int(x) == 90 for x in self.nom_freqs.values()]):
            template_alpha90 = None
        if not any([int(x) == 150 for x in self.nom_freqs.values()]):
            template_alpha150 = None
        if null_run or self.template_type is None:
            template_alpha90 = None
            template_alpha150 = None

        # Check for output data on disk
        save_attrs = ["cls_data", "nside"]

        if self.clean_type == "raw":
            data_name = "data"
        else:
            data_name = "data_{}".format(self.clean_type)
        save_name = "{}_xcorr".format(data_name)
        template_fit = False

        if null_run:
            save_attrs += ["cls_data_null"]
            if sub_planck:
                save_attrs += ["cls_data_sub_null"]
                save_attrs += ["cls_planck_null"]

        elif template_alpha90 is not None or template_alpha150 is not None:
            template_fit = True
            save_name = "{}_clean_{}_xcorr".format(data_name, self.template_type)
            save_attrs += [
                "cls_data_clean",
                "cls_template",
                "template_alpha90",
                "template_alpha150",
            ]

        if sub_planck:
            save_attrs += ["cls_data_sub"]
            save_attrs += ["cls_planck"]

        def apply_template():
            cls_clean = getattr(self, "cls_data_clean", OrderedDict())
            adict = {"90": template_alpha90, "150": template_alpha150}

            for spec in self.specs:
                cls_clean[spec] = copy.deepcopy(self.cls_data[spec])
                if spec not in self.cls_template:
                    continue
                for xname, d in cls_clean[spec].items():
                    if xname not in self.cls_template[spec]:
                        continue
                    m0, m1 = self.map_pairs[xname]
                    alphas = [adict.get(self.nom_freqs[m], None) for m in (m0, m1)]

                    t1, t2, t3 = self.cls_template[spec][xname]

                    if alphas[0] is not None:
                        d -= alphas[0] * t1
                    if alphas[1] is not None:
                        d -= alphas[1] * t2
                        if alphas[0] is not None:
                            d += alphas[0] * alphas[1] * t3
                            # subtract average template noise spectrum to debias
                            if sub_hm_noise:
                                d -= (
                                    alphas[0]
                                    * alphas[1]
                                    * self.cls_tnoise_hm1xhm2[spec][xname]
                                )

            self.cls_data_clean = cls_clean
            self.template_alpha90 = template_alpha90
            self.template_alpha150 = template_alpha150
            self.template_cleaned = True

        def subtract_planck_maps():
            cls_data_sub = getattr(self, "cls_data_sub", OrderedDict())
            cls_data_sub_null = getattr(self, "cls_data_sub_null", OrderedDict())

            for spec in self.specs:
                cls_data_sub[spec] = copy.deepcopy(self.cls_data[spec])
                for xname, d in cls_data_sub[spec].items():
                    t1, t2, t3 = self.cls_planck[spec][xname]
                    d += -t1 - t2 + t3

                # do null specs
                cls_data_sub_null[spec] = copy.deepcopy(self.cls_data_null[spec])
                for xname, d in cls_data_sub_null[spec].items():
                    t1, t2, t3 = self.cls_planck_null[spec][xname]
                    d += -t1 - t2 + t3

            self.cls_data_sub = cls_data_sub
            self.cls_data_sub_null = cls_data_sub_null
            self.planck_sub = True

        # change template subtraction coefficients for pre-loaded data
        if all([hasattr(self, attr) for attr in save_attrs]):
            if not template_fit and not getattr(self, "template_cleaned", False):
                return {k: getattr(self, k) for k in save_attrs}

            if template_fit and getattr(self, "template_cleaned", False):
                if (
                    template_alpha90 == self.template_alpha90
                    and template_alpha150 == self.template_alpha150
                ):
                    return {k: getattr(self, k) for k in save_attrs}

                apply_template()
                return {k: getattr(self, k) for k in save_attrs}

        ret = self.load_data(
            save_name,
            "data",
            fields=save_attrs,
            to_attrs=True,
            shape=data_shape,
            shape_ref="cls_data",
        )
        if ret is not None:
            self.planck_sub = False
            self.template_cleaned = False
            if null_run:
                if sub_planck and not self.planck_sub:
                    subtract_planck_maps()
                return ret
            if template_alpha90 is None and template_alpha150 is None:
                self.template_cleaned = False
                return ret
            if (
                template_alpha90 == self.template_alpha90
                and template_alpha150 == self.template_alpha150
            ):
                self.template_cleaned = True
                return ret
            apply_template()
            return ret

        # map spectra
        cls = OrderedDict()
        cls_null = OrderedDict() if null_run else None

        # set up template subtraction
        cls_tmp = None
        if template_fit:
            cls_tmp = OrderedDict()
            template_files = list(zip(self.template_files, self.template_files2))
        template_cleaned = False

        # set up planck subtraction
        cls_planck = None
        if sub_planck:
            cls_planck = OrderedDict()
            cls_planck_null = OrderedDict() if null_run else None
            planck_files_split = list(
                zip(
                    self.planck_files1_hm1,
                    self.planck_files1_hm2,
                    self.planck_files2_hm1,
                    self.planck_files2_hm2,
                )
            )
        planck_subtracted = False

        cache = dict()

        # convenience function
        def process_index(idx):
            if idx in cache:
                return cache[idx]

            self.log("Computing Alms for map {}/{}".format(idx + 1, num_maps), "all")

            m = self.get_map(map_files[idx])
            mask = self.get_mask(mask_files[idx])
            self.apply_mask(m, mask)

            if null_run:
                m2 = self.get_map(map_files2[idx])
                self.apply_mask(m2, mask)

                # sum and diff spectra for null tests
                # XXX should not take average but sum here if we want to
                # compare power with sum...
                m_alms = self.map2alm((m + m2) / 2.0, self.pol)
                mn_alms = self.map2alm((m - m2) / 2.0, self.pol)
                if sub_planck:
                    # cache raw data alms and planck alms together
                    (p1hm1, p1hm2, p2hm1, p2hm2) = planck_files_split[idx]
                    mp1hm1 = self.get_map(p1hm1)
                    mp1hm2 = self.get_map(p1hm2)
                    mp2hm1 = self.get_map(p2hm1)
                    mp2hm2 = self.get_map(p2hm2)
                    self.apply_mask(mp1hm1, mask)
                    self.apply_mask(mp1hm2, mask)
                    self.apply_mask(mp2hm1, mask)
                    self.apply_mask(mp2hm2, mask)
                    m_alms_hm1 = self.map2alm((mp1hm1 + mp2hm1) / 2.0, self.pol)
                    m_alms_hm2 = self.map2alm((mp1hm2 + mp2hm2) / 2.0, self.pol)
                    mn_alms_hm1 = self.map2alm((mp1hm1 - mp2hm1) / 2.0, self.pol)
                    mn_alms_hm2 = self.map2alm((mp1hm2 - mp2hm2) / 2.0, self.pol)
                    m_alms = (m_alms, m_alms_hm1, m_alms_hm2)
                    mn_alms = (mn_alms, mn_alms_hm1, mn_alms_hm2)

            elif not template_fit:
                m_alms = self.map2alm(m, self.pol)
                mn_alms = None
            else:
                # cache raw data alms and template alms together
                mn_alms = None
                m_alms = [self.map2alm(m, self.pol)]
                for tf in template_files[idx]:
                    self.log("Loading template from {}".format(tf), "detail")
                    mt = self.get_map(tf)
                    self.apply_mask(mt, mask)
                    mt_alms = self.map2alm(mt, self.pol)
                    # null out T template
                    if self.pol:
                        mt_alms[0] *= 0
                    m_alms.append(mt_alms)
                m_alms = tuple(m_alms)

            cache[idx] = (m_alms, mn_alms)
            return cache[idx]

        map_pairs = pt.tag_pairs(map_tags, index=True)

        for xname, (idx, jdx) in map_pairs.items():

            imap_alms, inull_alms = process_index(idx)
            jmap_alms, jnull_alms = process_index(jdx)

            self.log("Computing spectra {}x{}".format(idx + 1, jdx + 1), "detail")

            # store cross spectra
            if isinstance(imap_alms, tuple) and len(imap_alms) == 3:
                if sub_planck:
                    sub_planck = True
                else:
                    template_cleaned = True
                # raw map spectrum component
                cls1 = self.alm2cl(imap_alms[0], jmap_alms[0])

                # average template alms
                imt = (imap_alms[1] + imap_alms[2]) / 2.0
                jmt = (jmap_alms[1] + jmap_alms[2]) / 2.0

                # compute maximally symmetric cross spectra
                t1 = self.alm2cl(imt, jmap_alms[0])  # multiplies alpha_i
                t2 = self.alm2cl(imap_alms[0], jmt)  # multiplies alpha_j
                t3 = (
                    self.alm2cl(imap_alms[1], jmap_alms[2])
                    + self.alm2cl(imap_alms[2], jmap_alms[1])
                ) / 2.0  # multiplies alpha_i * alpha_j

                for s, spec in enumerate(self.specs):
                    if sub_planck:
                        cls_planck.setdefault(spec, OrderedDict())[xname] = (
                            t1[s],
                            t2[s],
                            t3[s],
                        )
                    else:
                        # apply template to TE/TB but not TT
                        if spec == "tt":
                            continue
                        cls_tmp.setdefault(spec, OrderedDict())[xname] = (
                            t1[s],
                            t2[s],
                            t3[s],
                        )
                if null_run:
                    # do this again for the null maps
                    cls_null1 = self.alm2cl(inull_alms[0], jnull_alms[0])

                    # average template alms
                    imt = (inull_alms[1] + inull_alms[2]) / 2.0
                    jmt = (jnull_alms[1] + jnull_alms[2]) / 2.0

                    # compute maximally symmetric cross spectra
                    t1 = self.alm2cl(imt, jnull_alms[0])
                    t2 = self.alm2cl(inull_alms[0], jmt)
                    t3 = (
                        self.alm2cl(inull_alms[1], jnull_alms[2])
                        + self.alm2cl(inull_alms[2], jnull_alms[1])
                    ) / 2.0

                    for s, spec in enumerate(self.specs):
                        cls_planck_null.setdefault(spec, OrderedDict())[xname] = (
                            t1[s],
                            t2[s],
                            t3[s],
                        )
            else:
                cls1 = self.alm2cl(imap_alms, jmap_alms)
                if null_run:
                    cls_null1 = self.alm2cl(inull_alms, jnull_alms)
            for s, spec in enumerate(self.specs):
                cls.setdefault(spec, OrderedDict())[xname] = cls1[s]
                if null_run:
                    cls_null.setdefault(spec, OrderedDict())[xname] = cls_null1[s]

        # store and return
        self.cls_data = cls
        self.cls_data_null = cls_null

        if template_cleaned:
            self.cls_template = cls_tmp
            apply_template()
        else:
            self.template_cleaned = False

        if sub_planck:
            self.cls_planck = cls_planck
            self.cls_planck_null = cls_planck_null
            subtract_planck_maps()
        else:
            self.planck_sub = False

        return self.save_data(save_name, from_attrs=save_attrs)

    def get_masked_sims(
        self,
        ensemble_mean=False,
        ensemble_median=False,
        sim_index=None,
        transfer=False,
        do_noise=True,
        sims_add_alms=True,
        lmin=2,
        qb_file=None,
    ):
        """
        Compute average signal and noise spectra for a given
        ensemble of maps.  The same procedure that is used for computing
        data cross spectra is used for each realization in the sim
        ensemble, and only the average spectra for all realizations
        are stored.

        See `get_masked_xcorr` for more details on how cross spectra
        are computed.

        Arguments
        ---------
        ensemble_mean : bool
            If true, the mean signal + noise spectrum is used in place
            of input data.  This is useful for testing the behavior of
            the estimator and mapmaker independently of the data.
        ensemble_median : bool
            If true, the median signal + noise spectrum is used in place
            of input data.  This is useful for testing the behavior of
            the estimator and mapmaker independently of the data.
        sim_index : int
            If not None, substitute the sim_index S+N alms for the observed alms
        sims_add_alms : bool
            If True and sim_index is not None, add sim alms instead of sim Cls
            to include signal and noise correlations
        qb_file : string
            Pointer to a bandpowers.npz file in the output directory. If used
            in sim_index mode, the noise sim read from disk will be corrected
            by the residual qb values stored in qb_file.

        Data Handling
        -------------
        This method is called at the 'sims' checkpoint, and loads or saves
        a data dictionary with the following entries:

            cls_signal : <same shape as cls_data>
            cls_signal_null : <same as cls_data>
                Average signal spectra (sum and difference if computing
                a null test).
            cls_noise : <same shape as cls_data>
            cls_noise_null : <same as cls_data>
                Average null spectra (sum and difference if computing
                a null test).
        """
        mask_files = self.mask_files
        map_tags = self.map_tags
        map_pairs = pt.tag_pairs(map_tags, index=True)
        num_maps = self.num_maps
        num_corr = self.num_corr
        data_shape = self.data_shape

        sims_attr = {}

        if transfer:
            sims_attr["signal_files"] = self.signal_transfer_files
            sims_attr["num_signal"] = self.num_signal_transfer
        else:
            sims_attr["signal_files"] = self.signal_files
            sims_attr["num_signal"] = self.num_signal

        sims_attr["noise_files"] = self.noise_files
        sims_attr["num_noise"] = self.num_noise
        sims_attr["noise_files_sim"] = self.noise_files_sim
        sims_attr["signal_files_sim"] = self.signal_files_sim
        sims_attr["num_noise_sim"] = self.num_noise_sim
        sims_attr["num_signal_sim"] = self.num_signal_sim
        sims_attr["foreground_files"] = self.foreground_files
        sims_attr["num_foreground"] = self.num_foreground

        foreground_sims = sims_attr["foreground_files"] is not None

        if do_noise:
            do_noise = sims_attr["noise_files"] is not None
            # if qb file is not none, modify cls by residual in file
            if qb_file is not None:
                if not os.path.exists(qb_file):
                    qb_file = os.path.join(self.output_root, qb_file)
                qb_file = pt.load_and_parse(qb_file)

        null_run = self.null_run
        if transfer:
            sims_attr["signal_files2"] = (
                self.signal_transfer_files2 if null_run else None
            )
        else:
            sims_attr["signal_files2"] = self.signal_files2 if null_run else None

        sims_attr["noise_files2"] = self.noise_files2 if null_run else None
        sims_attr["noise_files_sim2"] = self.noise_files_sim2 if null_run else None
        sims_attr["signal_files_sim2"] = self.signal_files_sim2 if null_run else None
        sims_attr["foreground_files2"] = self.foreground_files2 if null_run else None

        # convenience functions
        def process_index(files, files2, idx, idx2=None, cache=None, qbf=None):
            """
            Compute alms of masked input map
            """
            if cache is None:
                cache = {}

            if idx in cache:
                return cache[idx]

            filename = files[idx]
            if idx2 is None:
                self.log(
                    "Computing Alms for map {}/{}".format(idx + 1, num_maps), "all"
                )
            else:
                self.log(
                    "Computing Alms for sim {} of map {}/{}".format(
                        idx2, idx + 1, num_maps
                    ),
                    "all",
                )
                filename = filename[idx2]

            m = self.get_map(filename)
            mask = self.get_mask(mask_files[idx])
            self.apply_mask(m, mask)
            if null_run:
                # second null half
                filename2 = files2[idx]
                if idx2 is not None:
                    filename2 = filename2[idx2]
                m2 = self.get_map(filename2)
                self.apply_mask(m2, mask)

                m_alms = self.map2alm((m + m2) / 2.0, self.pol)
                mn_alms = self.map2alm((m - m2) / 2.0, self.pol)
            else:
                m_alms = self.map2alm(m, self.pol)
                mn_alms = None
            if qbf is not None:
                # if qb file is not none, modify alms by residual in file
                rbins = dict(filter(lambda x: "res" in x[0], qbf["bin_def"].items()))
                rfields = {"tt": [0], "ee": [1], "bb": [2], "eebb": [1, 2]}
                for rb, rb0 in rbins.items():
                    srb = rb.split("_", 2)
                    mod = np.zeros(np.max(rb0))
                    for ib, (left, right) in enumerate(rb0):
                        il = slice(left, right)
                        mod[il] = np.sqrt(1 + qbf["qb"][rb][ib])
                        if np.any(np.isnan(mod[il])):
                            warnings.warn(
                                "Unphysical residuals fit, "
                                + "setting to zero {} bin {}".format(rb, ib)
                            )
                            mod[il][np.isnan(mod[il])] = 1

                    for rf in rfields[srb[1]]:
                        if self.map_tags[idx] == srb[2]:
                            m_alms[rf] = hp.almxfl(m_alms[rf], mod)
                            if null_run:
                                mn_alms[rf] = hp.almxfl(mn_alms[rf], mod)

            cache[idx] = (m_alms, mn_alms)
            return cache[idx]

        def process_files():
            """
            Compute cross spectra
            """
            sig_field = "cls_signal"
            sig_null_field = "cls_signal_null"
            noise_field = "cls_noise"
            noise_null_field = "cls_noise_null"
            tot_field = "cls_sim"
            tot_null_field = "cls_sim_null"
            med_field = "cls_med"
            med_null_field = "cls_med_null"

            ### These fields are needed to iteratate on noise from res fits
            noise0_field = "cls_noise0"
            noise0_null_field = "cls_noise0_null"
            noise1_field = "cls_noise1"
            noise1_null_field = "cls_noise1_null"
            sxn0_field = "cls_sxn0"
            sxn0_null_field = "cls_sxn0_null"
            nxs0_field = "cls_nxs0"
            nxs0_null_field = "cls_nxs0_null"
            sxn1_field = "cls_sxn1"
            sxn1_null_field = "cls_sxn1_null"
            nxs1_field = "cls_nxs1"
            nxs1_null_field = "cls_nxs1_null"
            ###

            sig_files = sims_attr["signal_files"]
            sig_files2 = None
            noise_files = sims_attr["noise_files"]
            noise_files2 = None
            if null_run:
                sig_files2 = sims_attr["signal_files2"]
                noise_files2 = sims_attr["noise_files2"]

            nsim_sig = sims_attr["num_signal"]
            nsim_noise = sims_attr["num_noise"]

            cls_sig = OrderedDict()
            cls_null_sig = OrderedDict() if null_run else None
            cls_noise = OrderedDict() if do_noise else None
            cls_null_noise = OrderedDict() if null_run and do_noise else None
            cls_tot = OrderedDict()
            cls_null_tot = OrderedDict() if null_run else None
            cls_med = OrderedDict()
            cls_null_med = OrderedDict() if null_run else None

            ### Noise iteration from res fit fields
            cls_noise0 = OrderedDict() if do_noise else None
            cls_null_noise0 = OrderedDict() if null_run and do_noise else None
            cls_noise1 = OrderedDict() if do_noise else None
            cls_null_noise1 = OrderedDict() if null_run and do_noise else None
            cls_sxn0 = OrderedDict()
            cls_null_sxn0 = OrderedDict() if null_run else None
            cls_nxs0 = OrderedDict()
            cls_null_nxs0 = OrderedDict() if null_run else None
            cls_sxn1 = OrderedDict()
            cls_null_sxn1 = OrderedDict() if null_run else None
            cls_nxs1 = OrderedDict()
            cls_null_nxs1 = OrderedDict() if null_run else None

            sig_cache = dict()
            noise_cache = dict()
            if nsim_noise != 0:
                nsim_min = min([nsim_sig, nsim_noise])
            else:
                nsim_min = nsim_sig
            nsim_max = max([nsim_sig, nsim_noise])
            cls_all = np.zeros(
                [nsim_max, len(map_pairs.items()), len(self.specs), self.lmax + 1]
            )
            if null_run:
                cls_all_null = np.zeros(
                    [nsim_max, len(map_pairs.items()), len(self.specs), self.lmax + 1]
                )

            for isim in range(nsim_max):
                sig_cache.clear()
                noise_cache.clear()
                for xind, (xname, (idx, jdx)) in enumerate(map_pairs.items()):
                    self.log(
                        "Computing spectra {} for signal{} sim {}".format(
                            xname, "+noise" if do_noise else "", isim
                        ),
                        "detail",
                    )
                    if isim < nsim_sig:
                        simap_alms, sinull_alms = process_index(
                            sig_files, sig_files2, idx, isim, sig_cache
                        )
                        sjmap_alms, sjnull_alms = process_index(
                            sig_files, sig_files2, jdx, isim, sig_cache
                        )

                        cls1_sig = self.alm2cl(simap_alms, sjmap_alms)
                        if null_run:
                            cls_null1_sig = self.alm2cl(sinull_alms, sjnull_alms)

                    if do_noise and isim < nsim_noise:
                        nimap_alms, ninull_alms = process_index(
                            noise_files,
                            noise_files2,
                            idx,
                            isim,
                            noise_cache,
                            qbf=qb_file,
                        )
                        njmap_alms, njnull_alms = process_index(
                            noise_files,
                            noise_files2,
                            jdx,
                            isim,
                            noise_cache,
                            qbf=qb_file,
                        )

                        # need non-symmetric since will potentially modify these
                        # with different residuals for T, E, B
                        cls1_noise0 = self.alm2cl(
                            nimap_alms, njmap_alms, symmetric=False
                        )
                        cls1_noise1 = self.alm2cl(
                            njmap_alms, nimap_alms, symmetric=False
                        )
                        cls1_noise = (cls1_noise0 + cls1_noise1) / 2
                        cls1_sxn0 = self.alm2cl(simap_alms, njmap_alms, symmetric=False)
                        cls1_nxs0 = self.alm2cl(nimap_alms, sjmap_alms, symmetric=False)
                        cls1_sxn1 = self.alm2cl(njmap_alms, simap_alms, symmetric=False)
                        cls1_nxs1 = self.alm2cl(sjmap_alms, nimap_alms, symmetric=False)
                        cls1_sxn = (cls1_sxn0 + cls1_sxn1) / 2
                        cls1_nxs = (cls1_nxs0 + cls1_nxs1) / 2
                        if null_run:
                            cls_null1_noise0 = self.alm2cl(
                                ninull_alms, njnull_alms, symmetric=False
                            )
                            cls_null1_noise1 = self.alm2cl(
                                njnull_alms, ninull_alms, symmetric=False
                            )
                            cls_null1_noise = (cls_null1_noise0 + cls_null1_noise1) / 2

                            cls_null1_sxn0 = self.alm2cl(
                                sinull_alms, njnull_alms, symmetric=False
                            )
                            cls_null1_nxs0 = self.alm2cl(
                                ninull_alms, sjnull_alms, symmetric=False
                            )
                            cls_null1_sxn1 = self.alm2cl(
                                njnull_alms, sinull_alms, symmetric=False
                            )
                            cls_null1_nxs1 = self.alm2cl(
                                sjnull_alms, ninull_alms, symmetric=False
                            )
                            cls_null1_sxn = (cls_null1_sxn0 + cls_null1_sxn1) / 2
                            cls_null1_nxs = (cls_null1_nxs0 + cls_null1_nxs1) / 2

                        if isim < nsim_min:
                            cls1t = cls1_sig + cls1_sxn + cls1_nxs + cls1_noise
                            if null_run:
                                cls_null1t = (
                                    cls_null1_sig
                                    + cls_null1_sxn
                                    + cls_null1_nxs
                                    + cls_null1_noise
                                )
                    else:
                        cls1t = np.copy(cls1_sig)
                        if null_run:
                            cls_null1t = np.copy(cls_null1_sig)

                    for s, spec in enumerate(self.specs):
                        quants = []
                        if isim < nsim_sig:
                            quants += [[cls_sig, cls1_sig]]
                            if null_run:
                                quants += [[cls_null_sig, cls_null1_sig]]

                        if do_noise and isim < nsim_noise:
                            quants += [
                                [cls_noise, cls1_noise],
                                [cls_noise0, cls1_noise0],
                                [cls_noise1, cls1_noise1],
                                [cls_sxn0, cls1_sxn0],
                                [cls_sxn1, cls1_sxn1],
                                [cls_nxs0, cls1_nxs0],
                                [cls_nxs1, cls1_nxs1],
                            ]
                            if null_run:
                                quants += [
                                    [cls_null_noise, cls_null1_noise],
                                    [cls_null_noise0, cls_null1_noise0],
                                    [cls_null_noise1, cls_null1_noise1],
                                    [cls_null_sxn0, cls_null1_sxn0],
                                    [cls_null_sxn1, cls_null1_sxn1],
                                    [cls_null_nxs0, cls_null1_nxs0],
                                    [cls_null_nxs1, cls_null1_nxs1],
                                ]
                        if isim < nsim_min:
                            quants += [[cls_tot, cls1t]]
                            if null_run:
                                quants += [[cls_null_tot, cls_null1t]]

                        if len(quants):
                            # running average
                            for quant0, quant1 in quants:
                                d = quant0.setdefault(spec, OrderedDict()).setdefault(
                                    xname, np.zeros_like(quant1[s])
                                )
                                d[:] += (quant1[s] - d) / float(isim + 1)  # in-place
                        cls_all[isim][xind][s] = cls_tot[spec][xname]
                        if null_run:
                            cls_all_null[isim][xind][s] = cls_null_tot[spec][xname]

            cls_med_arr = np.median(cls_all, axis=0)
            for s, spec in enumerate(self.specs):
                cls_med[spec] = OrderedDict()
                for xind, xname in enumerate(map_pairs.keys()):
                    cls_med[spec][xname] = cls_med_arr[xind][s]
            if null_run:
                cls_null_med_arr = np.median(cls_all_null, axis=0)
                for s, spec in enumerate(self.specs):
                    cls_null_med[spec] = OrderedDict()
                    for xind, xname in enumerate(map_pairs.keys()):
                        cls_null_med[spec][xname] = cls_null_med_arr[xind][s]

            setattr(self, sig_field, cls_sig)
            setattr(self, sig_null_field, cls_null_sig)
            setattr(self, noise_field, cls_noise)
            setattr(self, noise_null_field, cls_null_noise)
            setattr(self, tot_field, cls_tot)
            setattr(self, tot_null_field, cls_null_tot)
            setattr(self, med_field, cls_med)
            setattr(self, med_null_field, cls_null_med)

            setattr(self, noise0_field, cls_noise0)
            setattr(self, noise0_null_field, cls_null_noise0)
            setattr(self, noise1_field, cls_noise1)
            setattr(self, noise1_null_field, cls_null_noise1)
            setattr(self, sxn0_field, cls_sxn0)
            setattr(self, sxn0_null_field, cls_null_sxn0)
            setattr(self, sxn1_field, cls_sxn1)
            setattr(self, sxn1_null_field, cls_null_sxn1)
            setattr(self, nxs0_field, cls_nxs0)
            setattr(self, nxs0_null_field, cls_null_nxs0)
            setattr(self, nxs1_field, cls_nxs1)
            setattr(self, nxs1_null_field, cls_null_nxs1)

        def check_options():
            if ensemble_mean:
                self.log("Substitute signal + noise for observed spectrum", "info")
                for spec in self.specs:
                    for xname in self.cls_data[spec]:
                        if do_noise:
                            self.cls_data[spec][xname] = self.cls_sim[spec][xname]
                        else:
                            self.cls_data[spec][xname] = self.cls_signal[spec][xname]
                        if null_run:
                            if do_noise:
                                self.cls_data_null[spec][xname] = self.cls_sim_null[
                                    spec
                                ][xname]
                            else:
                                self.cls_data_null[spec][xname] = self.cls_signal_null[
                                    spec
                                ][xname]
            elif ensemble_median:
                self.log(
                    "Substitute signal + noise median for observed spectrum", "info"
                )
                for spec in self.specs:
                    for xname in self.cls_data[spec]:
                        self.cls_data[spec][xname] = self.cls_med[spec][xname]
                        if null_run:
                            self.cls_data_null[spec][xname] = self.cls_med_null[spec][
                                xname
                            ]

            elif sim_index is not None:
                msg = "Substitute #{} sim signal + noise for observed alms"
                self.log(msg.format(sim_index), "info")

                # find the sim file that matches the requested sim index
                # NB: this assumes that the sim files have the form
                # *_<sim_index>.fits, and will raise an error
                # if this is not the case, or if the index is not found
                file_indices = [
                    int(os.path.splitext(x.rsplit("_")[-1])[0])
                    for x in sims_attr["signal_files"][0]
                ]
                file_index = file_indices.index(sim_index)
                scache = {}
                ncache = {}
                fgcache = {}

                for xname, (idx, jdx) in map_pairs.items():
                    simap_alms, sinull_alms = process_index(
                        sims_attr["signal_files_sim"],
                        sims_attr["signal_files_sim2"],
                        idx,
                        file_index,
                        scache,
                    )
                    simap_alms = np.copy(simap_alms)
                    if null_run:
                        sinull_alms = np.copy(sinull_alms)
                    if do_noise:
                        nimap_alms, ninull_alms = process_index(
                            sims_attr["noise_files_sim"],
                            sims_attr["noise_files_sim2"],
                            idx,
                            file_index,
                            ncache,
                            qbf=qb_file,
                        )
                        if sims_add_alms:
                            simap_alms += nimap_alms
                            if null_run:
                                sinull_alms += ninull_alms
                    if foreground_sims:
                        fimap_alms, finull_alms = process_index(
                            sims_attr["foreground_files"],
                            sims_attr["foreground_files2"],
                            idx,
                            file_index,
                            fgcache,
                        )
                        if sims_add_alms:
                            simap_alms += fimap_alms
                            if null_run:
                                sinull_alms += finull_alms

                    sjmap_alms, sjnull_alms = process_index(
                        sims_attr["signal_files_sim"],
                        sims_attr["signal_files_sim2"],
                        jdx,
                        file_index,
                        scache,
                    )
                    sjmap_alms = np.copy(sjmap_alms)
                    if null_run:
                        sjnull_alms = np.copy(sjnull_alms)
                    if do_noise:
                        njmap_alms, njnull_alms = process_index(
                            sims_attr["noise_files_sim"],
                            sims_attr["noise_files_sim2"],
                            jdx,
                            file_index,
                            ncache,
                            qbf=qb_file,
                        )
                        if sims_add_alms:
                            sjmap_alms += njmap_alms
                            if null_run:
                                sjnull_alms += njnull_alms
                    if foreground_sims:
                        fjmap_alms, fjnull_alms = process_index(
                            sims_attr["foreground_files"],
                            sims_attr["foreground_files2"],
                            jdx,
                            file_index,
                            fgcache,
                        )
                        if sims_add_alms:
                            sjmap_alms += fjmap_alms
                            if null_run:
                                sjnull_alms += fjnull_alms

                    cls = self.alm2cl(simap_alms, sjmap_alms)
                    if not sims_add_alms:
                        if do_noise:
                            cls += self.alm2cl(nimap_alms, njmap_alms)
                        if foreground_sims:
                            cls += self.alm2cl(fimap_alms, fjmap_alms)
                    for s, spec in enumerate(self.specs):
                        self.cls_data[spec][xname] = cls[s]

                    if null_run:
                        cls = self.alm2cl(sinull_alms, sjnull_alms)
                        if not sims_add_alms:
                            if do_noise:
                                cls += self.alm2cl(ninull_alms, njnull_alms)
                            if foreground_sims:
                                cls += self.alm2cl(finull_alms, fjnull_alms)

                        for s, spec in enumerate(self.specs):
                            self.cls_data_null[spec][xname] = cls[s]

            self.ensemble_mean = ensemble_mean
            self.ensemble_median = ensemble_median
            self.sim_index = sim_index

        save_attrs = [
            "cls_signal",
            "cls_noise",
            "cls_sim",
            "cls_med",
            "cls_noise0",
            "cls_noise1",
            "cls_sxn0",
            "cls_sxn1",
            "cls_nxs0",
            "cls_nxs1",
        ]
        if null_run:
            save_attrs += [
                "cls_signal_null",
                "cls_noise_null",
                "cls_sim_null",
                "cls_med_null",
                "cls_noise0_null",
                "cls_noise1_null",
                "cls_sxn0_null",
                "cls_sxn1_null",
                "cls_nxs0_null",
                "cls_nxs1_null",
            ]

        if transfer:
            save_name = "sims_xcorr_{}".format(self.signal_transfer_type)
            cp = "sims_transfer"
            if self.signal_transfer_type == self.signal_type:
                self.force_rerun["sims"] = False
        else:
            save_name = "sims_xcorr_{}".format(self.signal_type)
            cp = "sims"

        ret = self.load_data(
            save_name,
            cp,
            fields=save_attrs,
            to_attrs=True,
            shape=data_shape,
            shape_ref="cls_signal",
        )
        if ret is not None:
            if do_noise and self.cls_noise is None:
                process_files()
                check_options()
                return self.save_data(save_name, from_attrs=save_attrs)
            elif not do_noise and self.cls_noise is not None:
                self.cls_noise = None
                self.cls_noise_null = None
                ret["cls_noise"] = None
                ret["cls_noise_null"] = None
                check_options()
                return ret
            else:
                check_options()
                return ret

        # process signal, noise, and S+N
        process_files()

        if not do_noise:
            self.cls_noise = None
            self.cls_noise_null = None

        # save and return
        check_options()
        return self.save_data(save_name, from_attrs=save_attrs)

    def get_masked_fake_data(
        self,
        fake_data_r=None,
        fake_data_template=None,
        sim_index=None,
        template_alpha90=None,
        template_alpha150=None,
        noise_type=None,
        do_noise=True,
        do_signal=True,
        save_data=False,
        sub_hm_noise=True,
    ):
        """
        In memory, make a fake data map with signal, noise, and
        foregrounds.
        Signal maps are signal_scalar + fake_data_r * signal_tensor
        where scalar maps are assumed to be in signal_r0 directory
        and tensor maps are assumed to be in signal_r0tens directory.
        sim_index is used to determine which sims. Noise maps taken
        from usual noise directory. Templates read read from
        templates_fake_data_template/halfmission-1.

        This function doesn't write anything to disk. It just constructs
        the maps and computes the Cls and replaces data cls with them
        """
        map_tags = self.map_tags
        map_files = self.map_files
        map_root = self.map_root
        mask_files = self.mask_files
        raw_files = self.raw_files
        num_maps = self.num_maps
        data_shape = self.data_shape
        data_root = self.data_root

        scalar_root = os.path.join(data_root, "signal_r0")
        tensor_root = os.path.join(data_root, "signal_r1tens")
        noise_root = os.path.join(data_root, "noise_{}".format(noise_type))
        # ignore unused template coefficients
        if not any([int(x) == 90 for x in self.nom_freqs.values()]):
            template_alpha90 = None
        if not any([int(x) == 150 for x in self.nom_freqs.values()]):
            template_alpha150 = None

        template_fit = fake_data_template is not None
        if template_fit:
            template_files = list(zip(self.template_files, self.template_files2))
            template_root = os.path.join(
                data_root, "templates_{}/halfmission-1".format(fake_data_template)
            )

        cache = dict()
        adict = {"90": template_alpha90, "150": template_alpha150}
        self.log("fake data r: {}".format(fake_data_r))

        def process_index(idx):
            # create the fake map for each map in map_files,
            # compute alms for that and templates
            if idx in cache:
                return cache[idx]
            self.log("Computing Alms for fake data map {}/{}".format(idx, num_maps))
            mfile = map_files[idx]
            freq = self.nom_freqs[map_tags[idx]]
            if do_signal:
                scalar = self.get_map(
                    mfile.replace(map_root, scalar_root).replace(
                        ".fits", "_{:04}.fits".format(sim_index)
                    )
                )
                tensor = self.get_map(
                    mfile.replace(map_root, tensor_root).replace(
                        ".fits", "_{:04}.fits".format(sim_index)
                    )
                )
            else:
                self.log("Using signal 0", "detail")
                scalar = self.get_map(
                    mfile.replace(map_root, scalar_root).replace(".fits", "_0000.fits")
                )
                tensor = self.get_map(
                    mfile.replace(map_root, tensor_root).replace(".fits", "_0000.fits")
                )
            if do_noise:
                noise = self.get_map(
                    mfile.replace(map_root, noise_root).replace(
                        ".fits", "_{:04}.fits".format(sim_index)
                    )
                )
            else:
                self.log("Using noise 0", "detail")
                noise = self.get_map(
                    mfile.replace(map_root, noise_root).replace(".fits", "_0000.fits")
                )

            if template_fit:
                template = self.get_map(mfile.replace(map_root, template_root))
            else:
                template = 0

            m_tot = (
                scalar
                + np.sqrt(np.abs(fake_data_r)) * tensor
                + noise
                + adict[freq] * template
            )
            mask = self.get_mask(mask_files[idx])
            self.apply_mask(m_tot, mask)
            m_alms = self.map2alm(m_tot, self.pol)

            if fake_data_r < 0:
                m_totn = (
                    scalar
                    - np.sqrt(np.abs(fake_data_r)) * tensor
                    + noise
                    + adict[freq] * template
                )
                self.apply_mask(m_totn, mask)
                mn_alms = self.map2alm(m_totn, self.pol)

            if template_fit:
                m_alms = [m_alms]
                for tf in template_files[idx]:
                    self.log("Loading template from {}".format(tf), "detail")
                    mt = self.get_map(tf)
                    self.apply_mask(mt, mask)
                    mt_alms = self.map2alm(mt, self.pol)
                    # null out T template
                    if self.pol:
                        mt_alms[0] *= 0
                    m_alms.append(mt_alms)
                m_alms = tuple(m_alms)
            if fake_data_r < 0:
                cache[idx] = tuple([m_alms, mn_alms])
            else:
                cache[idx] = m_alms
            return cache[idx]

        map_pairs = pt.tag_pairs(map_tags, index=True)
        for xname, (idx, jdx) in map_pairs.items():
            imap_alms = process_index(idx)
            jmap_alms = process_index(jdx)

            self.log(
                "Computing fake data spectra {}x{}".format(idx + 1, jdx + 1), "detail"
            )

            # store cross spectra
            if isinstance(imap_alms, tuple) and len(imap_alms) in [2, 3]:
                if fake_data_r >= 0:
                    cls1 = self.alm2cl(imap_alms[0], jmap_alms[0])

                    # average template alms
                    imt = (imap_alms[1] + imap_alms[2]) / 2.0
                    jmt = (jmap_alms[1] + jmap_alms[2]) / 2.0

                    # compute maximally symmetric cross spectra
                    t1 = self.alm2cl(imt, jmap_alms[0])  # multiplies alpha_i
                    t2 = self.alm2cl(imap_alms[0], jmt)  # multiplies alpha_j
                    t3 = (
                        self.alm2cl(imap_alms[1], jmap_alms[2])
                        + self.alm2cl(imap_alms[2], jmap_alms[1])
                    ) / 2.0  # multiplies alpha_i * alpha_j

                    for s, spec in enumerate(self.specs):
                        # apply template to TE/TB but not TT
                        if spec == "tt":
                            continue
                        self.cls_template[spec][xname] = (t1[s], t2[s], t3[s])

                else:
                    mp_i = imap_alms[0][0]
                    mn_i = imap_alms[1]
                    mp_j = jmap_alms[0][0]
                    mn_j = jmap_alms[1]
                    t1_i = imap_alms[0][1]
                    t2_i = imap_alms[0][2]
                    t1_j = jmap_alms[0][1]
                    t2_j = jmap_alms[0][2]

                    cls1 = (
                        0.5 * (self.alm2cl(mp_i, mn_j) + self.alm2cl(mn_i, mp_j))
                        + 0.5 * self.alm2cl(mn_i, mn_j)
                        - 0.5 * self.alm2cl(mp_i, mp_j)
                    )

                    # average template alms
                    imt = (t1_i + t2_i) / 2.0
                    jmt = (t1_j + t2_j) / 2.0

                    # compute maximally symmetric cross spectra
                    t1 = 1.5 * self.alm2cl(mn_j, imt) - 0.5 * self.alm2cl(mp_j, imt)
                    t2 = 1.5 * self.alm2cl(mn_i, jmt) - 0.5 * self.alm2cl(mp_i, jmt)
                    t3 = 0.5 * (self.alm2cl(t1_i, t2_j) + self.alm2cl(t2_i, t1_j))

                    for s, spec in enumerate(self.specs):
                        # apply template to TE/TB but not TT
                        if spec == "tt":
                            continue
                        self.cls_template[spec][xname] = (t1[s], t2[s], t3[s])
            else:
                cls1 = self.alm2cl(imap_alms, jmap_alms)
            for s, spec in enumerate(self.specs):
                self.cls_data[spec][xname] = cls1[s]

        def apply_template():
            cls_clean = getattr(self, "cls_data_clean", OrderedDict())
            adict = {"90": template_alpha90, "150": template_alpha150}

            for spec in self.specs:
                cls_clean[spec] = copy.deepcopy(self.cls_data[spec])
                if spec not in self.cls_template:
                    continue
                for xname, d in cls_clean[spec].items():
                    if xname not in self.cls_template[spec]:
                        continue
                    m0, m1 = self.map_pairs[xname]
                    alphas = [adict.get(self.nom_freqs[m], None) for m in (m0, m1)]

                    t1, t2, t3 = self.cls_template[spec][xname]

                    if alphas[0] is not None:
                        d -= alphas[0] * t1
                    if alphas[1] is not None:
                        d -= alphas[1] * t2
                        if alphas[0] is not None:
                            d += alphas[0] * alphas[1] * t3
                            # subtract average template noise spectrum to debias
                            if sub_hm_noise:
                                d -= (
                                    alphas[0]
                                    * alphas[1]
                                    * self.cls_tnoise_hm1xhm2[spec][xname]
                                )

            self.cls_data_clean = cls_clean
            self.template_alpha90 = template_alpha90
            self.template_alpha150 = template_alpha150
            self.template_cleaned = True

        if template_fit:
            apply_template()
        if save_data:
            save_attrs = [
                "cls_data",
                "cls_data_clean",
                "cls_template",
                "template_alpha90",
                "template_alpha150",
                "nside",
            ]
            if fake_data_r < 0:
                rname = "rmp{:03}".format(int(np.abs(fake_data_r) * 1000))
            else:
                rname = "rp{:03}".format(int(np.abs(fake_data_r) * 1000))
            data_name = "data_{}_clean_{}_sim{:03}".format(
                rname, fake_data_template, sim_index
            )
            save_name = "{}_xcorr".format(data_name)
            self.save_data(save_name, from_attrs=save_attrs)

    def get_masked_template_noise(self, template_type):
        """
        Compute hm1, hm2, and hm1xhm2 template noise spectra from
        sim ensemble

        Data Handling
        -------------
        This method is called at the 'template_noise' checkpoint,
        and loads or saves a data dictionary with the following entries:

            cls_tnoise_hm1 : <same shape as cls_data>
            cls_tnoise_hm2 : <same shape as cls_data>
            cls_tnoise_hm1xhm2 : <same shape as cls_data>
        """
        mask_files = self.mask_files
        map_tags = self.map_tags
        map_pairs = pt.tag_pairs(map_tags, index=True)
        num_maps = self.num_maps
        num_corr = self.num_corr
        data_shape = self.data_shape

        sims_attr = {}

        sims_attr["template_noise_files"] = self.template_noise_files
        sims_attr["template_noise_files2"] = self.template_noise_files2
        sims_attr["num_template_noise"] = self.num_template_noise

        # convenience functions
        def process_index(files, idx, idx2=None, cache=None):
            """
            Compute alms of masked input map
            """
            if cache is None:
                cache = {}

            if idx in cache:
                return cache[idx]

            filename = files[idx]
            if idx2 is None:
                self.log(
                    "Computing Alms for map {}/{}".format(idx + 1, num_maps), "all"
                )
            else:
                self.log(
                    "Computing Alms for sim {} of map {}/{}".format(
                        idx2, idx + 1, num_maps
                    ),
                    "all",
                )
                filename = filename[idx2]

            m = self.get_map(filename)
            mask = self.get_mask(mask_files[idx])
            self.apply_mask(m, mask)
            m_alms = self.map2alm(m, self.pol)

            cache[idx] = m_alms
            return cache[idx]

        def process_files():
            """
            Compute cross spectra
            """
            hm1_field = "cls_tnoise_hm1"
            hm2_field = "cls_tnoise_hm2"
            hm1xhm2_field = "cls_tnoise_hm1xhm2"

            hm1_files = sims_attr["template_noise_files"]
            hm2_files = sims_attr["template_noise_files2"]

            nsim_tnoise = sims_attr["num_template_noise"]

            cls_hm1 = OrderedDict()
            cls_hm2 = OrderedDict()
            cls_hm1xhm2 = OrderedDict()

            hm1_cache = dict()
            hm2_cache = dict()
            for isim in range(nsim_tnoise):
                hm1_cache.clear()
                hm2_cache.clear()
                for xind, (xname, (idx, jdx)) in enumerate(map_pairs.items()):
                    self.log(
                        "Computing spectra {} for template noise sim {}".format(
                            xname, isim
                        ),
                        "detail",
                    )
                    hm1imap_alms = process_index(hm1_files, idx, isim, hm1_cache)
                    hm1jmap_alms = process_index(hm1_files, jdx, isim, hm1_cache)
                    hm2imap_alms = process_index(hm2_files, idx, isim, hm2_cache)
                    hm2jmap_alms = process_index(hm2_files, jdx, isim, hm2_cache)

                    cls1_hm1 = self.alm2cl(hm1imap_alms, hm1jmap_alms)
                    cls1_hm2 = self.alm2cl(hm2imap_alms, hm2jmap_alms)
                    cls1_hm1xhm2 = 0.5 * (
                        self.alm2cl(hm1imap_alms, hm2jmap_alms)
                        + self.alm2cl(hm2imap_alms, hm1jmap_alms)
                    )

                    for s, spec in enumerate(self.specs):
                        quants = [
                            [cls_hm1, cls1_hm1],
                            [cls_hm2, cls1_hm2],
                            [cls_hm1xhm2, cls1_hm1xhm2],
                        ]
                        # running average
                        for quant0, quant1 in quants:
                            d = quant0.setdefault(spec, OrderedDict()).setdefault(
                                xname, np.zeros_like(quant1[s])
                            )
                            d[:] += (quant1[s] - d) / float(isim + 1)  # in-place

            setattr(self, hm1_field, cls_hm1)
            setattr(self, hm2_field, cls_hm2)
            setattr(self, hm1xhm2_field, cls_hm1xhm2)

        save_attrs = ["cls_tnoise_hm1", "cls_tnoise_hm2", "cls_tnoise_hm1xhm2"]

        ### this block is so sims with template type like
        # 353_100_gauss_003 can use ensemble in 353_100_gauss
        tp = template_type.split("_")
        ttype = template_type
        if tp[-1].isdigit():
            if ttype[-7:] not in ["353_100", "217_100"]:
                ttype = "_".join(tp[:-1])
        save_name = "template_noise_{}".format(ttype)
        cp = "template_noise"

        ret = self.load_data(
            save_name,
            cp,
            fields=save_attrs,
            to_attrs=True,
            shape=data_shape,
            shape_ref="cls_tnoise_hm1",
        )
        if ret is not None:
            return ret

        # process template noise
        process_files()

        return self.save_data(save_name, from_attrs=save_attrs)

    def get_marg_table(
        self,
        tt_marg=False,
        auto_marg_table=None,
        cross_marg_table=None,
        marg_value=1e10,
    ):
        """
        Construct a table of coefficients for marginalizing over particular spectra.

        Arguments
        ---------
        tt_marg : bool
            If True, marginalize over all but one map's TT spectrum, by
            artificially inflating the appropriate TT noise terms.
            This avoids singular matrices in the case where all input maps
            have high SNR in T.
        auto_marg_table : string
            A filename for JSON file specifying autos to marginalise over. This
            overrides tt_marg. Set this to avoid degeneracy in high SNR
            identitcal maps (ie. TT in Spider) and also to marginalize over
            auto biases. Generalisation of tt_marg
        cross_marg_table : string
            A filename for JSON file specifying cross spectra to marginalise
            over. Allows for, eg, ignoring cross spectra with the same
            Planck half missions used.
        marg_value : float
            Multiplicative factor to use for marginalization. This value
            multiplies each of the spectra that are included in the table.
        """
        self.marg_table = None
        marg_table = OrderedDict()

        def update_marg_table(spec, xname):
            if xname not in self.map_pairs:
                return
            self.log("Marginalizing {}".format(xname), "detail")
            stable = marg_table.setdefault(spec, OrderedDict())
            stable.setdefault(xname, marg_value)
            # remove residual bins for marginalized autospectra
            m0, m1 = self.map_pairs[xname]
            if m0 == m1:
                res_tag = "res_{}_{}".format(spec, m0)
                if res_tag not in self.bin_def and spec in ["ee", "bb"]:
                    res_tag = "res_eebb_{}".format(m0)
                if res_tag in self.bin_def:
                    self.nbins_res -= len(self.bin_def[res_tag])
                    del self.bin_def[res_tag]

        # These two methods are obsolete now
        if tt_marg and auto_marg_table is None:
            # Marginalize over all but one of the TT for same mask
            # this is because TT is very high S/N
            for xname in list(self.map_pairs)[1:]:
                update_marg_table("tt", xname)
            self.log("Marginalizing over all but first TT auto.", "detail")
        elif auto_marg_table is not None:
            if not os.path.exists(auto_marg_table):
                rp = os.path.join(os.getenv("XFASTER_PATH", "config"), auto_marg_table)
                if os.path.exists(rp):
                    auto_marg_table = rp
                else:
                    raise OSError(
                        "Missing auto marginalization JSON file {}".format(
                            auto_marg_table
                        )
                    )
            with open(auto_marg_table, "r") as marg_file:
                marg = marg_file.read()
            # Remove comments
            marg_cleaned = re.sub(r"#.*\n", "", marg)
            marg_opts = json.loads(marg_cleaned)

            for tag in self.map_tags:
                for spec in ["tt", "ee", "bb"]:
                    if marg_opts.get(tag, {}).get(spec.upper(), None):
                        self.log(
                            "Marginalizing auto {} of {}.".format(spec, tag), "detail"
                        )
                        update_marg_table(spec, "{0}:{0}".format(tag))

        if cross_marg_table is not None:
            if not os.path.exists(cross_marg_table):
                rp = os.path.join(os.getenv("XFASTER_PATH"), "config", cross_marg_table)
                if os.path.exists(rp):
                    cross_marg_table = rp
                else:
                    raise OSError(
                        "Missing cross marginalization JSON file {}".format(
                            cross_marg_table
                        )
                    )
            with open(cross_marg_table, "r") as marg_file:
                marg = marg_file.read()
            # Remove comments
            marg_cleaned = re.sub(r"#.*\n", "", marg)
            marg_opts = json.loads(marg_cleaned)
            for m0, m1s in marg_opts.items():
                for m1 in m1s:
                    xname = "{}:{}".format(m0, m1)
                    if xname not in self.map_pairs:
                        # name is the reverse
                        xname = "{}:{}".format(m1, m0)
                    if xname not in self.map_pairs:
                        warnings.warn("No such pair with maps {}:{}".format(m0, m1))
                    for spec in self.specs:
                        update_marg_table(spec, xname)

        self.marg_table = marg_table
        return marg_table

    def get_kernels(self, window_lmax=None):
        """
        Compute kernels using the mask cross-spectra.  This follows
        the polspice azimuthal approximation for the kernel computation.

        Arguments
        ---------
        window_lmax : int
            The window within which the kernel is computed about
            each ell bin.

        Data Handling
        -------------
        This method is called at the 'kernels' checkpoint and loads or saves
        the following data keys to disk:

            kern, pkern, mkern, xkern : (num_mask_corr, lmax+1, 2*lmax+1)
                Temperature and polarization kernels
        """

        if window_lmax is None:
            window_lmax = self.lmax

        save_name = "kernels"
        save_attrs = ["kern", "pkern", "mkern", "xkern", "window_lmax"]
        ret = self.load_data(
            save_name,
            "kernels",
            fields=save_attrs,
            to_attrs=True,
            shape=self.kern_shape,
            shape_ref="kern",
            value_ref={"window_lmax": window_lmax},
        )
        if ret is not None:
            return ret

        kern = OrderedDict()
        if self.pol:
            pkern = OrderedDict()
            mkern = OrderedDict()
            xkern = OrderedDict()
        else:
            pkern = None
            mkern = None
            xkern = None

        lmax = self.lmax
        pol = self.pol
        wls = self.wls

        all_ells = np.arange(2 * lmax + 1)
        for xname in self.map_pairs:
            kern[xname] = np.zeros((lmax + 1, 2 * lmax + 1))
            if pol:
                pkern[xname] = np.zeros((lmax + 1, 2 * lmax + 1))
                mkern[xname] = np.zeros((lmax + 1, 2 * lmax + 1))
                xkern[xname] = np.zeros((lmax + 1, 2 * lmax + 1))

        for l in all_ells[2 : lmax + 1]:
            if np.mod(l, 50) == 0:
                self.log("Computing kernels for ell {}/{}".format(l, lmax), "detail")
            l2 = np.min([2 * lmax + 1, l + lmax + 1])
            # populate upper triangle
            for ll in all_ells[l:l2]:
                j0, j0_lmin, j0_lmax = xft.ThreeJC_2(l, 0, ll, 0)
                if pol:
                    j2, j2_lmin, j2_lmax = xft.ThreeJC_2(l, 2, ll, -2)

                # only go up to window lmax
                j0_lmax = np.minimum(j0_lmax, window_lmax)

                # computed as in https://arxiv.org/abs/1909.09375
                # equations 128 - 136
                l3 = np.arange(j0_lmin, j0_lmax + 1)
                dl3 = 2.0 * l3 + 1.0
                vk = j0[l3] ** 2 * dl3
                if pol:
                    sign = ((-1.0) ** (l + ll + l3)).astype(int)
                    v = j2[l3] ** 2 * dl3
                    vp = v * (1.0 + sign) ** 2
                    vm = v * (1.0 - sign) ** 2
                    vx = j2[l3] * j0[l3] * dl3 * (1.0 + sign)
                for xname in self.map_pairs:
                    wls1 = wls[xname][:, l3]
                    kern[xname][l, ll] += (vk * wls1[0]).sum(axis=-1)
                    if pol:
                        pkern[xname][l, ll] += (vp * wls1[1]).sum(axis=-1)
                        mkern[xname][l, ll] += (vm * wls1[1]).sum(axis=-1)
                        xkern[xname][l, ll] += (vx * wls1[2]).sum(axis=-1)

        # apply symmetry relation
        for l in all_ells[2 : lmax + 1]:
            ll = np.arange(2 * lmax + 1)
            dll = (2.0 * ll + 1.0) / 4.0 / np.pi
            sll = slice(l, lmax + 1)
            for xname in self.map_pairs:
                # populate lower triangle (wigners are symmetric in l and ll)
                kern[xname][sll, l] = kern[xname][l, sll]
                if pol:
                    pkern[xname][sll, l] = pkern[xname][l, sll]
                    mkern[xname][sll, l] = mkern[xname][l, sll]
                    xkern[xname][sll, l] = xkern[xname][l, sll]
                # apply ell scaling along the axis that we bin over
                kern[xname][l, :] *= dll
                if pol:
                    pkern[xname][l, :] *= dll / 4.0
                    mkern[xname][l, :] *= dll / 4.0
                    xkern[xname][l, :] *= dll / 2.0

        # save and return
        self.kern = kern
        self.pkern = pkern
        self.mkern = mkern
        self.xkern = xkern
        self.window_lmax = window_lmax

        return self.save_data(save_name, from_attrs=save_attrs)

    def get_signal_shape(
        self,
        filename=None,
        r=None,
        component=None,
        flat=None,
        tbeb=False,
        foreground_fit=False,
        signal_mask=None,
        transfer=False,
        save=True,
    ):
        """
        Load a shape spectrum for input to the Fisher iteration algorithm.

        If the spectrum is used as input to `get_transfer`, it must match
        the spectrum used to generate the simulations, in order to compute
        the correct transfer function.

        Alternatively, the spectrum can be computed using CAMB for arbitrary
        values of `r`, typically used to compute the `r` likelihood once
        the bandpowers have been computed.

        Finally, the spectrum can be flat in ell^2 Cl.  This is typically
        used as the input shape for computing bandpowers for a null test.

        Arguments
        ---------
        filename : string
            Filename for a spectrum on disk.  If None, and `r` is None and
            `flat` is False, this will search for a spectrum stored in
            `signal_<signal_type>/spec_signal_<signal_type>.dat`.
            Otherwise, if the filename is a relative path and not found,
            the $XFASTER_PATH/data directory is searched.
        r : float
            If supplied and `flat` is False, a spectrum is computed using
            CAMB for the given `r` value.  Overrides `filename`.
        component : 'scalar', 'tensor', 'fg'
            If 'scalar', and `r` is not None, return just the r=0 scalar terms
            in the signal model.  If 'tensor', return just the tensor component
            scaled by the input `r` value. If 'fg', return just fg term
        flat : float
            If given, a spectrum that is flat in ell^2 Cl is returned, with
            amplitude given by the supplied value. Overrides all other options.
        tbeb : bool
            Include TB EB shape
        foreground_fit : bool
            Include a foreground shape in cls_shape
        signal_mask: str array
            Include only these spectra, others set to zero.
            Options: TT, EE, BB, TE, EB, TB
        transfer : bool
            If True, this is a transfer function run. If `filename` is None
            and `r` is None and `flat` is False, will search for a spectrum
            stored in
            `signal_<signal_transfer_type>/spec_signal_<signal_transfer_type>.dat`.

        Returns
        -------
        cls : OrderedDict
            Dictionary keyed by spectrum (cmb_tt, cmb_ee, ... , fg), each
            entry containing a vector of length 2 * lmax + 1
        """

        lmax_kern = 2 * self.lmax

        specs = ["cmb_tt"]
        nspecs = 1
        if self.pol:
            for s in ["ee", "bb", "te"]:
                specs += ["cmb_{}".format(s)]
            if tbeb:
                specs += ["cmb_eb", "cmb_tb"]
            nspecs += 5
        if foreground_fit:
            specs += ["fg"]
            nspecs += 1

        if save:
            shape = (nspecs, lmax_kern + 1)
            save_name = "shape_transfer" if transfer else "shape"

            opts = dict(
                filename=filename,
                r=r,
                flat=flat,
                tbeb=tbeb,
                foreground_fit=foreground_fit,
                signal_mask=signal_mask,
            )
            ret = self.load_data(
                save_name, save_name, shape_ref="cls_shape", shape=shape, value_ref=opts
            )
            if ret is not None:
                if r is not None:
                    self.r_model = ret["r_model"]
                return ret["cls_shape"]

        ell = np.arange(lmax_kern + 1)
        ellfac = ell * (ell + 1) / 2.0 / np.pi
        cls_shape = OrderedDict()

        if flat is not None and flat is not False:
            if flat is True:
                flat = 2e-5
            # flat spectrum for null tests
            for spec in specs:
                cls_shape[spec] = flat * np.ones_like(ell)

        elif r is not None:
            # cache model components
            if not hasattr(self, "r_model") or self.r_model is None:
                # scalar CAMB spectrum
                scal = xft.get_camb_cl(r=0, lmax=lmax_kern)
                # tensor CAMB spectrum for r=1, scales linearly with r
                tens = xft.get_camb_cl(r=1, lmax=lmax_kern, nt=0, spec="tensor")
                self.r_model = {"scalar": scal, "tensor": tens}
                if save:
                    opts["r_model"] = self.r_model
            # CAMB spectrum for given r value
            component = str(component).lower()
            if component == "scalar":
                cls_camb = self.r_model["scalar"]
            elif component == "tensor":
                cls_camb = r * self.r_model["tensor"]
            else:
                cls_camb = self.r_model["scalar"] + r * self.r_model["tensor"]
            ns, _ = cls_camb.shape
            for s, spec in enumerate(specs[:ns]):
                cls_shape[spec] = cls_camb[s]
        else:
            # signal sim model or custom filename
            if filename is None:
                signal_root = (
                    self.signal_transfer_root if transfer else self.signal_root
                )
                filename = "spec_{}.dat".format(os.path.basename(signal_root))
                filename = os.path.join(signal_root, filename)
            if not os.path.exists(filename) and not os.path.isabs(filename):
                filename = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../data", filename)
                )
            if not os.path.exists(filename):
                raise OSError("Missing model file {}".format(filename))

            tmp = np.loadtxt(filename, unpack=True)

            ltmp = tmp.shape[-1]
            if lmax_kern + 1 < ltmp:
                ltmp = lmax_kern + 1
            else:
                raise ValueError(
                    "Require at least lmax={} in model file, found {}".format(
                        lmax_kern, ltmp
                    )
                )

            # camb starts at l=2, so set first 2 ells to be 0
            cls_shape["cmb_tt"] = np.append([0, 0], tmp[1, : ltmp - 2])
            if self.pol:
                if np.any(tmp[2, : ltmp - 2] < 0):
                    # this is true if TE is the third index instead of EE
                    # (ell is 0th index for CAMB)
                    self.log(
                        "Old CAMB format in model file {}. Re-indexing.".format(
                            filename
                        ),
                        "detail",
                    )
                    pol_specs = [3, 4, 2]
                else:
                    pol_specs = [2, 3, 4]
                for spec, d in zip(specs[1:4], tmp[pol_specs]):
                    cls_shape[spec] = np.append([0, 0], d[: ltmp - 2])

        if self.pol:
            # EB and TB flat l^2 * C_l
            if tbeb and (flat is None or flat is False):
                tbeb_flat = np.abs(cls_shape["cmb_bb"][100]) * ellfac[100] * 1e-4
                tbeb_flat = np.ones_like(cls_shape["cmb_bb"]) * tbeb_flat
                tbeb_flat[:2] = 0
                cls_shape["cmb_eb"] = np.copy(tbeb_flat)
                cls_shape["cmb_tb"] = np.copy(tbeb_flat)
            elif not tbeb:
                cls_shape["cmb_eb"] = np.zeros_like(ell, dtype=float)
                cls_shape["cmb_tb"] = np.zeros_like(ell, dtype=float)

        if foreground_fit:
            # From Planck LIV EE dust
            cls_dust = 34.0 * (ell[2:] / 80.0) ** (-2.28 + 2.0)
            cls_shape["fg"] = np.append([0, 0], cls_dust)
            self.log(
                "Added foreground to cls shape {}".format(list(cls_shape)), "detail"
            )

        # divide out l^2/2pi
        for spec in specs:
            cls_shape[spec][2:] /= ellfac[2:]
            cls_shape[spec][:2] = 0.0

        if signal_mask is not None:
            self.log("Masking {} spectra".format(signal_mask), "detail")
            for csk in cls_shape:
                masked = False
                for smk in signal_mask:
                    if smk.lower() in csk:
                        masked = True
                if not masked:
                    cls_shape[csk] *= 1.0e-12

        if component == "fg":
            cls_shape = {"fg": cls_shape["fg"]}

        if save:
            opts["cls_shape"] = cls_shape
            self.save_data(save_name, **opts)

        return cls_shape

    def get_beams(self, pixwin=True):
        """
        Return beam window functions for all input map tags.

        Arguments
        ---------
        pixwin : bool
            If True, the pixel window function for the map `nside` is
            applied to the Gaussian beams.

        Returns
        -------
        beam_windows : dict
            A dictionary of beam window function arrays
            (3 x lmax+1 if pol, 1 x lmax+1 if not) for each map tag

        Data Handling
        -------------
        This method is called at the 'beams' checkpoint and loads or saves
        a dictionary containing just the `beam_windows` key to disk.
        """

        lsize = 2 * self.lmax + 1
        nspec = 6 if self.pol else 1
        beam_shape = (self.num_maps * nspec, lsize)

        save_name = "beams"
        cp = "beams"

        ret = self.load_data(
            save_name, cp, fields=["beam_windows"], to_attrs=True, shape=beam_shape
        )
        if ret is not None:
            return ret["beam_windows"]

        beam_windows = OrderedDict()
        beam_windows["tt"] = OrderedDict()
        if self.pol:
            for s in ["ee", "bb", "te", "eb", "tb"]:
                beam_windows[s] = OrderedDict()

        pwl = np.ones((3 if self.pol else 1, lsize))
        if pixwin:
            pwl *= 0.0
            pixT, pixP = hp.pixwin(self.nside, pol=self.pol)
            end = min(len(pixT), lsize)
            pwl[0, :end] = pixT[:end]
            pwl[1, :end] = pixP[:end]
            pwl[2, :end] = np.sqrt(pixT[:end] * pixP[:end])

        if self.beam_product not in ["None", None]:
            beam_prod = np.load(self.beam_product)
        else:
            beam_prod = {}

        for tag, otag in self.map_reobs_freqs.items():
            if tag in beam_prod:
                bl = np.atleast_2d(beam_prod[otag])[:, :lsize]
            elif self.fwhm[otag] not in ["None", None]:
                bl = hp.gauss_beam(float(self.fwhm[otag]), lsize - 1, self.pol)
                if self.pol:
                    bl = bl.T[[0, 1, 3]]
            else:
                raise ValueError("No beam in config for {}".format(tag))

            blpw = np.atleast_2d(bl * pwl)[:, :lsize]
            beam_windows["tt"][tag] = np.copy(blpw[0])
            if self.pol:
                for s in ["ee", "bb", "eb"]:
                    beam_windows[s][tag] = np.copy(blpw[1])
                for s in ["te", "tb"]:
                    beam_windows[s][tag] = np.copy(blpw[2])

        # save and return
        self.beam_windows = beam_windows
        self.save_data(save_name, from_attrs=["beam_windows"])
        return self.beam_windows

    def get_beam_err(self, tag, lsize):
        """
        Get error envelope to multiply beam by (so, to get beam + 2 sigma error,
        do beam * (1+2*get_beam_err(tag)))
        """
        if getattr(self, "beam_err", None) is not None:
            if tag in self.beam_err:
                return self.beam_err[tag]
        else:
            self.beam_err = {}
        # get original tag if it's not a single frequency or fpu
        mind = self.map_tags == tag
        tag = np.atleast_1d(self.map_tags_orig[mind])[0]
        try:
            err_envelope = pickle.load(
                open(
                    os.path.join(
                        os.getenv("XFASTER_PATH"), "data", "beam_error_model.pkl"
                    ),
                    "rb",
                ),
                encoding="latin1",
            )
        except TypeError:
            err_envelope = pickle.load(
                open(
                    os.path.join(
                        os.getenv("XFASTER_PATH"), "data", "beam_error_model.pkl"
                    ),
                    "rb",
                )
            )

        # add envelope to beam for marginalizing over error
        etag = "{}_dbl_data".format(tag)
        if etag in err_envelope:
            self.beam_err[tag] = err_envelope[etag][:lsize]
            return self.beam_err[tag]
        elif tag == "90":
            tot_off = np.zeros(lsize)
            for fpu in [2, 4, 6]:
                etag = "X{}_dbl_data".format(fpu)
                tot_off += err_envelope[etag][:lsize]
            self.beam_err[tag] = tot_off / 3.0
            return self.beam_err[tag]
        elif tag == "150":
            tot_off = np.zeros(lsize)
            for fpu in [1, 3, 5]:
                etag = "X{}_dbl_data".format(fpu)
                tot_off += err_envelope[etag][:lsize]
            self.beam_err[tag] = tot_off / 3.0
            return self.beam_err[tag]
        elif tag == "150a":
            tot_off = np.zeros(lsize)
            for fpu in [1, 5]:
                etag = "X{}_dbl_data".format(fpu)
                tot_off += err_envelope[etag][:lsize]
            self.beam_err[tag] = tot_off / 2.0
            return self.beam_err[tag]
        else:
            warnings.warn(
                "No beam error field found for {}.".format(tag)
                + " Ignoring and moving on."
            )

    def get_bin_def(
        self,
        bin_width=25,
        lmin=2,
        tbeb=False,
        foreground_fit=False,
        bin_width_fg=25,
        residual_fit=False,
        res_specs=None,
        bin_width_res=25,
        weighted_bins=False,
    ):
        """
        Construct the bin definition array that defines the bins for each output
        spectrum.

        Arguments
        ---------
        bin_width : int or list of ints
            Width of each spectrum bin.  If a scalar, the same width is applied
            to all cross spectra.  Otherwise, must be a list of up to six
            elements, listing bin widths for the spectra in the order (TT, EE,
            BB, TE, EB, TB).
        lmin : int
            Minimum ell for binned spectra
        tbeb : bool
            If True, EB and TB bins are constructed so that these spectra are
            computed by the estimator.  Otherwise, these spectra are fixed at
            zero.
        foreground_fit : bool
            If True, construct bin definitions for foreground components as
            well.
        bin_width_fg : int or list of ints
            Width of each foreground spectrum bin.  If a scalar, the same width
            is applied to all cross spectra.  Otherwise, must be a list of up to
            six elements, listing bin widths for the spectra in the order (TT,
            EE, BB, TE, EB, TB).
        residual_fit : bool
            If True, fit for (compute bandpower amplitudes for) several wide
            bins of excess noise.
        res_specs : list of strings
            List of spectra which are to be included in the residual fit.  Can
            be individual spectra ('tt', 'ee', 'bb'), or 'eebb' to fit for EE
            and BB residuals simultaneously.  If not supplied, this defaults to
            ['eebb'] for polarized maps, and ['tt'] for unpolarized maps.
        bin_width_res : int or list of ints
            Width of each residual spectrum bin.  If a scalar, the same width
            is applied to all spectra for all cross spectra.  Otherwise, must
            be a list of up to nspec * nmaps elements, listing bin widths for
            each of the spectra in `res_specs` in order, then ordered by map.
        weighted_bins : bool
            If True, use an lfac-weighted binning operator to construct Cbls.
            By default, a flat binning operator is used.

        Returns
        -------
        bin_def : dict
            The bin definition dictionary.  Each key contains a Nx2 array
            that defines the left and right edges for each bin of the
            corresponding spectrum.
        """
        npol = (6 if tbeb else 4) if self.pol else 1
        specs = ["tt", "ee", "bb", "te", "eb", "tb"][:npol]

        if np.isscalar(bin_width):
            bin_width = [bin_width] * npol
        bin_width = np.asarray(bin_width)[:npol]

        bwerr = "EE and BB must have the same bin width (for mixing)"
        if self.pol and bin_width[1] != bin_width[2]:
            raise ValueError(bwerr)

        # Define bins
        nbins_cmb = 0
        bin_def = OrderedDict()
        for spec, bw in zip(specs, bin_width):
            bins = np.arange(lmin, self.lmax, bw)
            bins = np.append(bins, self.lmax + 1)
            bin_def["cmb_{}".format(spec)] = np.column_stack((bins[:-1], bins[1:]))
            nbins_cmb += len(bins) - 1
        self.log("Added {} CMB bins to bin_def".format(nbins_cmb), "detail")

        # Do the same for foreground bins
        nbins_fg = 0
        if foreground_fit:
            if np.isscalar(bin_width_fg):
                bin_width_fg = [bin_width_fg] * npol
            bin_width_fg = np.asarray(bin_width_fg)[:npol]

            if self.pol and bin_width_fg[1] != bin_width_fg[2]:
                raise ValueError("Foreground {}".format(bwerr))

            for spec, bw in zip(specs, bin_width_fg):
                bins = np.arange(lmin, self.lmax, bw)
                bins = np.append(bins, self.lmax + 1)
                bin_def["fg_{}".format(spec)] = np.column_stack((bins[:-1], bins[1:]))
                nbins_fg += len(bins) - 1
            bin_def["delta_beta"] = np.array([[0, 0]])
            self.log(
                "Added {} foreground bins to bin_def".format(nbins_fg + 1), "detail"
            )

        # Do the same for residual bins
        nbins_res = 0
        if residual_fit:
            if res_specs is None:
                res_specs = ["eebb"] if self.pol else ["tt"]
            res_specs = [s.lower() for s in res_specs]
            nmap = len(self.map_tags)
            nspecr = len(res_specs) * nmap
            if np.isscalar(bin_width_res):
                bin_width_res = [bin_width_res] * nspecr
            elif len(bin_width_res) == nspecr // nmap:
                bin_width_res = np.array([bin_width_res] * nmap).ravel()
            bin_width_res = np.asarray(bin_width_res)[:nspecr].reshape(nmap, -1)

            for tag, bws in zip(self.map_tags, bin_width_res):
                if self.pol and "ee" in res_specs and "bb" in res_specs:
                    if bws[res_specs.index("ee")] != bws[res_specs.index("bb")]:
                        raise ValueError("Residual {} {}".format(tag, bwerr))

                for spec, bw in zip(res_specs, bws):
                    bins = np.arange(lmin, self.lmax, bw)
                    bins = np.append(bins, self.lmax + 1)
                    btag = "res_{}_{}".format(spec, tag)
                    bin_def[btag] = np.column_stack((bins[:-1], bins[1:]))
                    nbins_res += len(bins) - 1

            self.log("Added {} residual bins to bin_def".format(nbins_res), "detail")

        self.lmin = lmin
        self.nbins_cmb = nbins_cmb
        self.nbins_fg = nbins_fg
        self.nbins_res = nbins_res
        self.bin_def = bin_def
        self.specs = specs
        self.weighted_bins = weighted_bins

        return self.bin_def

    def bin_cl_template(
        self,
        cls_shape,
        map_tag=None,
        transfer_run=False,
        beam_error=False,
        fg_ell_ind=0,
    ):
        """
        Compute the Cbl matrix from the input shape spectrum.

        This method requires beam windows, kernels and transfer functions
        (if `transfer_run` is False) to have been precomputed.

        Arguments
        ---------
        cls_shape : array_like
            The shape spectrum to use.  This can be computed using
            `get_signal_shape` or otherwise.
        map_tag : str
            If supplied, the Cbl is computed only for the given map tag
            (or cross if map_tag is map_tag1:map_tag2).
            Otherwise, it is computed for all maps and crosses.
        transfer_run : bool
            If True, this assumes a unity transfer function for all bins, and
            the output Cbl is used to compute the transfer functions that are
            then loaded when this method is called with `transfer_run = False`.
        beam_error : bool
            If True, use beam error envelope instead of beam to get cbls that
            are 1 sigma beam error envelope offset of signal terms.
        fg_ell_ind : float
            If binning foreground shape, offset the ell index from the reference
            by this amount.

        Returns
        -------
        cbl : dict of arrays (num_bins, 2, lmax + 1)
            The Cbl matrix, indexed by component and spectrum, then by map
            cross, e.g. `cbl['cmb_tt']['map1:map2']`
            E/B mixing terms are stored in elements `cbl[:, 1, :]`,
            and unmixed terms are stored in elements `cbl[:, 0, :]`.
        """
        map_pairs = None
        if map_tag is not None:
            if map_tag in self.map_pairs:
                map_pairs = {map_tag: self.map_pairs[map_tag]}
                map_tags = list(set(self.map_pairs[map_tag]))
            else:
                map_tags = [map_tag]
        else:
            map_tags = self.map_tags

        if map_pairs is None:
            map_pairs = pt.tag_pairs(map_tags)

        specs = list(self.specs)
        if transfer_run:
            if "eb" in specs:
                specs.remove("eb")
            if "tb" in specs:
                specs.remove("tb")

        lmax = self.lmax
        lmax_kern = lmax  # 2 * self.lmax

        # populate transfer function terms
        transfer = OrderedDict()
        for spec in specs:
            transfer[spec] = OrderedDict()
            stag = "cmb_{}".format(spec)
            for tag in map_tags:
                # if computing transfer function, set transfer to 1 everywhere
                transfer[spec][tag] = np.full(lmax_kern + 1, float(transfer_run))
                if not transfer_run:
                    # set each l equal to the transfer function computed in its bin
                    for ib, (left, right) in enumerate(self.bin_def[stag]):
                        il = slice(left, right)
                        transfer[spec][tag][il] = self.qb_transfer[stag][tag][ib]

        ls = slice(2, lmax + 1)
        cbl = OrderedDict()

        comps = []
        if "cmb_tt" in cls_shape or "cmb_ee" in cls_shape:
            comps += ["cmb"]
        if "fg" in cls_shape and not transfer_run:
            comps += ["fg"]
        if self.nbins_res > 0 and not transfer_run:
            comps += ["res"]
            cls_noise = self.cls_noise_null if self.null_run else self.cls_noise
            cls_noise0 = self.cls_noise0_null if self.null_run else self.cls_noise0
            cls_noise1 = self.cls_noise1_null if self.null_run else self.cls_noise1
            cls_sxn0 = self.cls_sxn0_null if self.null_run else self.cls_sxn0
            cls_sxn1 = self.cls_sxn1_null if self.null_run else self.cls_sxn1
            cls_nxs0 = self.cls_nxs0_null if self.null_run else self.cls_nxs0
            cls_nxs1 = self.cls_nxs1_null if self.null_run else self.cls_nxs1

        ell = np.arange(lmax_kern + 1)
        lfac = ell * (ell + 1) / 2.0 / np.pi

        if self.weighted_bins:

            def binup(d, left, right):
                w = lfac[left:right]
                # normalize
                w = w / w.sum() * len(w)
                return (d[..., left:right] * w).sum(axis=-1)

        else:

            def binup(d, left, right):
                return d[..., left:right].sum(axis=-1)

        def bin_things(comp, d, md, d_b1, d_b2, d_b3, md_b1, md_b2, md_b3):
            if "res" in comp:
                return
            for si, spec in enumerate(specs):
                stag = "{}_{}".format(comp, spec)
                cbl.setdefault(stag, OrderedDict())
                mstag = None
                if spec in ["ee", "bb"]:
                    mstag = stag + "_mix"
                    cbl.setdefault(mstag, OrderedDict())
                bd = self.bin_def[stag]
                for xi, (xname, (tag1, tag2)) in enumerate(map_pairs.items()):
                    if beam_error:
                        cbl[stag].setdefault(xname, OrderedDict())
                        cbl[stag][xname]["b1"] = np.zeros((len(bd), lmax + 1))
                        cbl[stag][xname]["b2"] = np.zeros((len(bd), lmax + 1))
                        cbl[stag][xname]["b3"] = np.zeros((len(bd), lmax + 1))
                    else:
                        cbl[stag][xname] = np.zeros((len(bd), lmax + 1))
                    if spec in ["ee", "bb"]:
                        if beam_error:
                            cbl[mstag].setdefault(xname, OrderedDict())
                            cbl[mstag][xname]["b1"] = np.zeros((len(bd), lmax + 1))
                            cbl[mstag][xname]["b2"] = np.zeros((len(bd), lmax + 1))
                            cbl[mstag][xname]["b3"] = np.zeros((len(bd), lmax + 1))
                        else:
                            cbl[mstag][xname] = np.zeros((len(bd), lmax + 1))

                    # integrate per bin
                    for idx, (left, right) in enumerate(bd):
                        if beam_error:
                            cbl[stag][xname]["b1"][idx, ls] = binup(
                                d_b1[:, si, xi], left, right
                            )
                            cbl[stag][xname]["b2"][idx, ls] = binup(
                                d_b2[:, si, xi], left, right
                            )
                            cbl[stag][xname]["b3"][idx, ls] = binup(
                                d_b3[:, si, xi], left, right
                            )
                        else:
                            cbl[stag][xname][idx, ls] = binup(d[:, si, xi], left, right)
                        if spec in ["ee", "bb"]:
                            if beam_error:
                                cbl[mstag][xname]["b1"][idx, ls] = binup(
                                    md_b1[:, si - 1, xi], left, right
                                )
                                cbl[mstag][xname]["b2"][idx, ls] = binup(
                                    md_b2[:, si - 1, xi], left, right
                                )
                                cbl[mstag][xname]["b3"][idx, ls] = binup(
                                    md_b3[:, si - 1, xi], left, right
                                )
                            else:
                                cbl[mstag][xname][idx, ls] = binup(
                                    md[:, si - 1, xi], left, right
                                )

        for comp in comps:
            # convert to matrices to do multiplication to speed things up,
            # except for res is weird so don't do it for that.
            # need n_xname x n_spec x ell
            nspec = len(specs)
            nxmap = len(map_pairs.items())
            if comp == "fg" and fg_ell_ind != 0:
                s_arr = (ell / 80.0) ** fg_ell_ind
                s_arr[0] = 0
                if not beam_error:
                    # don't create a new object in memory each time
                    # use last one's space to save runtime
                    self.d = np.multiply(self.d_fg, s_arr, out=getattr(self, "d", None))
                    self.md = np.multiply(
                        self.md_fg, s_arr, out=getattr(self, "md", None)
                    )
                    bin_things(
                        comp, self.d, self.md, None, None, None, None, None, None
                    )
                else:
                    self.d_b1 = np.multiply(
                        self.d_fg_b1, s_arr, out=getattr(self, "d_b1", None)
                    )
                    self.d_b2 = np.multiply(
                        self.d_fg_b2, s_arr, out=getattr(self, "d_b2", None)
                    )
                    self.d_b3 = np.multiply(
                        self.d_fg_b3, s_arr, out=getattr(self, "d_b3", None)
                    )
                    self.md_b1 = np.multiply(
                        self.md_fg_b1, s_arr, out=getattr(self, "md_b1", None)
                    )
                    self.md_b2 = np.multiply(
                        self.md_fg_b2, s_arr, out=getattr(self, "md_b2", None)
                    )
                    self.md_b3 = np.multiply(
                        self.md_fg_b3, s_arr, out=getattr(self, "md_b3", None)
                    )
                    bin_things(
                        comp,
                        None,
                        None,
                        self.d_b1,
                        self.d_b2,
                        self.d_b3,
                        self.md_b1,
                        self.md_b2,
                        self.md_b3,
                    )
            else:
                k_arr = np.zeros([nspec, nxmap, self.lmax - 1, lmax_kern + 1])
                mk_arr = np.zeros([2, nxmap, self.lmax - 1, lmax_kern + 1])
                f_arr = np.zeros([nspec, nxmap, lmax_kern + 1])
                b_arr = np.zeros([nspec, nxmap, lmax_kern + 1])
                s_arr = np.zeros([nspec, nxmap, lmax_kern + 1])
                if beam_error:
                    b1_arr = np.zeros([nspec, nxmap, lmax_kern + 1])
                    b2_arr = np.zeros([nspec, nxmap, lmax_kern + 1])
                    b3_arr = np.zeros([nspec, nxmap, lmax_kern + 1])

                for si, spec in enumerate(specs):
                    stag = "{}_{}".format(comp, spec)
                    mstag = None
                    if comp != "res" and spec in ["ee", "bb"]:
                        mstag = stag + "_mix"

                    for xi, (xname, (tag1, tag2)) in enumerate(map_pairs.items()):
                        if "res" in comp:
                            s0, s1 = spec
                            res_tags = {
                                "s0m0": "res_{}_{}".format(s0 * 2, tag1),
                                "s0m1": "res_{}_{}".format(s0 * 2, tag2),
                                "s1m0": "res_{}_{}".format(s1 * 2, tag1),
                                "s1m1": "res_{}_{}".format(s1 * 2, tag2),
                            }
                            bd = [[0, lmax + 1]]
                            # if any component of XY spec is in residual bin
                            # def, use that bin def
                            for k, v in res_tags.items():
                                spec0 = v.split("_")[1]
                                if v not in self.bin_def:
                                    if spec0 in ["ee", "bb"]:
                                        v = v.replace(spec0, "eebb")
                                        if v in self.bin_def:
                                            bd = self.bin_def[v]
                                else:
                                    bd = self.bin_def[v]
                            for comp in [
                                "res0_nxn",
                                "res1_nxn",
                                "res0_sxn",
                                "res1_sxn",
                                "res0_nxs",
                                "res1_nxs",
                                "res",
                            ]:
                                stag = "{}_{}".format(comp, spec)
                                cbl.setdefault(stag, OrderedDict())
                                cbl[stag][xname] = np.zeros((len(bd), lmax + 1))
                                if comp == "res0_nxn":
                                    cl1 = cls_noise0[spec][xname]
                                elif comp == "res1_nxn":
                                    cl1 = cls_noise1[spec][xname]
                                elif comp == "res0_sxn":
                                    cl1 = cls_sxn0[spec][xname]
                                elif comp == "res1_sxn":
                                    cl1 = cls_sxn1[spec][xname]
                                elif comp == "res0_nxs":
                                    cl1 = cls_nxs0[spec][xname]
                                elif comp == "res1_nxs":
                                    cl1 = cls_sxn1[spec][xname]
                                elif comp == "res":
                                    cl1 = cls_noise[spec][xname]
                                for idx, (left, right) in enumerate(bd):
                                    lls = slice(left, right)
                                    cbl[stag][xname][idx, lls] = np.copy(cl1[lls])

                            continue

                        # get cross spectrum transfer function
                        if tag1 == tag2:
                            f_arr[si, xi] = transfer[spec][tag1]
                        else:
                            f_arr[si, xi] = np.sqrt(
                                transfer[spec][tag1] * transfer[spec][tag2]
                            )
                        # get cross spectrum beam window function
                        b_arr[si, xi] = (
                            self.beam_windows[spec][tag1]
                            * self.beam_windows[spec][tag2]
                        )[: lmax_kern + 1]

                        if beam_error:
                            # beam term with error needs to include cross terms
                            # since it's squared, so bsig1, bsig2 sigma added is
                            # c_model = mean_model +
                            #     (bsig1 * berr1 * bl_2 + bsig2 * berr2 * bl_1 +
                            #      bsig1 * bisg2 * berr1 * berr2) * Kll'*Fl*Cl_sky
                            b1_err = (
                                self.get_beam_err(tag1, len(b_arr[si, xi]))
                                * self.beam_windows[spec][tag1][: lmax_kern + 1]
                            )
                            b2_err = (
                                self.get_beam_err(tag2, len(b_arr[si, xi]))
                                * self.beam_windows[spec][tag2][: lmax_kern + 1]
                            )
                            b1_arr[si, xi] = (
                                b1_err * self.beam_windows[spec][tag2][: lmax_kern + 1]
                            )
                            b2_arr[si, xi] = (
                                b2_err * self.beam_windows[spec][tag1][: lmax_kern + 1]
                            )
                            b3_arr[si, xi] = b1_err * b2_err

                        # use correct shape spectrum
                        if comp == "fg":
                            # single foreground spectrum
                            s_arr = (
                                cls_shape["fg"][: lmax_kern + 1]
                                * (ell / 80.0) ** fg_ell_ind
                            )
                            s_arr[0] = 0
                        else:
                            s_arr[si, xi] = cls_shape["cmb_{}".format(spec)][
                                : lmax_kern + 1
                            ]
                        # get cross spectrum kernel terms
                        if spec == "tt":
                            k_arr[si, xi] = self.kern[xname][ls, : lmax_kern + 1]
                        elif spec in ["ee", "bb"]:
                            k_arr[si, xi] = self.pkern[xname][ls, : lmax_kern + 1]
                            mk_arr[si - 1, xi] = self.mkern[xname][ls, : lmax_kern + 1]
                        elif spec in ["te", "tb"]:
                            k_arr[si, xi] = self.xkern[xname][ls, : lmax_kern + 1]
                        elif spec == "eb":
                            k_arr[si, xi] = (
                                self.pkern[xname][ls] - self.mkern[xname][ls]
                            )[:, : lmax_kern + 1]
                # need last 3 dims of kernel to match other arrays
                k_arr = np.transpose(k_arr, axes=[2, 0, 1, 3])
                mk_arr = np.transpose(mk_arr, axes=[2, 0, 1, 3])
                if s_arr.ndim == 1:
                    s_arr_md = s_arr
                else:
                    s_arr_md = s_arr[1:3]
                if not beam_error:
                    d = k_arr * b_arr * f_arr * s_arr
                    md = mk_arr * b_arr[1:3] * f_arr[1:3] * s_arr_md
                    if comp == "fg":
                        self.d_fg = np.copy(d)
                        self.md_fg = np.copy(md)
                    d_b1 = None
                    d_b2 = None
                    d_b3 = None
                    md_b1 = None
                    md_b2 = None
                    md_b3 = None
                else:
                    d = None
                    md = None
                    d_b1 = k_arr * b1_arr * f_arr * s_arr
                    d_b2 = k_arr * b2_arr * f_arr * s_arr
                    d_b3 = k_arr * b3_arr * f_arr * s_arr
                    md_b1 = mk_arr * b1_arr[1:3] * f_arr[1:3] * s_arr_md
                    md_b2 = mk_arr * b2_arr[1:3] * f_arr[1:3] * s_arr_md
                    md_b3 = mk_arr * b3_arr[1:3] * f_arr[1:3] * s_arr_md
                    if comp == "fg":
                        self.d_fg_b1 = d_b1
                        self.d_fg_b2 = d_b2
                        self.d_fg_b3 = d_b3
                        self.md_fg_b1 = md_b1
                        self.md_fg_b2 = md_b2
                        self.md_fg_b3 = md_b3
                bin_things(comp, d, md, d_b1, d_b2, d_b3, md_b1, md_b2, md_b3)
        return cbl


    def get_model_spectra(
        self, qb, cbl, delta=True, res=True, cls_noise=None, cond_noise=None
    ):
        """
        Compute unbinned model spectra from qb amplitudes and a Cbl matrix.
        Requires pre-loaded bin definitions using `get_bin_def` or
        `get_transfer`.

        This method is used internally by `fisher_calc`.

        Arguments
        ---------
        qb : dict of arrays
            Array of bandpowers for every spectrum bin.
        cbl : dict
            Cbl dict as computed by `bin_cl_template`.
        delta : bool
            If True, evaluate the foreground model at the spectral
            index offset by qb['delta_beta']
        res : bool
            If True, include the residual noise model terms.
        cls_noise : OrderedDict
            If supplied, the noise spectrum is applied to the model spectrum.
        cond_noise : float
            Conditioning noise amplitude to add to TT, EE and BB autospectra,
            to improve convergence of the fisher iterations.  The noise model
            is constant cond_noise for EE, BB and 10x that for TT.

        Returns
        -------
        cls : dict of arrays
            Model spectra.  Keyed by spectrum type, e.g. 'total_xx' for the
            total model spectrom, 'fg_xx' for the foreground terms, 'res_xx' for
            the residual (noise) terms, where 'xx' is one of the six power
            spectrum components (tt, ee, bb, te, eb, tb).  Each entry in the
            dictionary is itself a dictionary keyed by map cross, e.g.
            'map1:map1' for an autospectrum term, 'map1:map2' for a cross
            spectrum, etc, and the map names are the same as those in the
            `map_tags` attribute.  Each individual spectrum is an array of
            length `lmax + 1`.
        """
        comps = []

        if any([k.startswith("cmb_") for k in qb]):
            comps = ["cmb"]

        delta_beta = 0.0
        if "delta_beta" in qb:
            # Evaluate fg at spectral index pivot for derivative
            # in Fisher matrix, unless delta is True
            if delta:
                delta_beta = qb["delta_beta"][0]
            comps += ["fg"]

        if res and any([k.startswith("res_") for k in qb]):
            comps += ["res"]

        if cls_noise is not None:
            comps += ["noise"]

        if cond_noise is not None:
            cls_cond = np.ones(self.lmax + 1) * cond_noise
            cls_cond[:2] = 0
            comps += ["cond"]

        if not len(comps):
            raise ValueError("Must specify at least one model component")

        cls = OrderedDict()

        specs = []
        for spec in self.specs:
            if "cmb_{}".format(spec) in cbl:
                # Don't add entries that won't be filled in later
                cls["total_{}".format(spec)] = OrderedDict()
            elif "fg_{}".format(spec) in cbl:
                cls["total_{}".format(spec)] = OrderedDict()
            specs.append(spec)

        for comp in comps:
            for spec in specs:
                stag = "{}_{}".format(comp, spec)
                if spec in ["ee", "bb"]:
                    mstag = "{}_{}".format(comp, "bb" if spec == "ee" else "ee")

                if comp == "noise":
                    pairs = list(cls_noise[spec])
                elif comp == "cond":
                    pairs = list(cls["total_{}".format(spec)])
                else:
                    if "res" not in comp and stag not in qb:
                        continue
                    if "res" not in comp and stag not in cbl:
                        continue
                    pairs = self.map_pairs.keys()  # list(cbl[stag])

                for xname in pairs:
                    tag1, tag2 = self.map_pairs[xname]

                    # extract qb's for the component spectrum
                    if comp == "cmb":
                        qbs = qb[stag]
                        if spec in ["ee", "bb"]:
                            qbm = qb[mstag]

                    elif comp == "fg":
                        # frequency scaling for foreground model
                        # I don't remember why delta beta was done this way.
                        # For likelihood, it makes sense to just use beta_ref+db
                        freq_scale = xft.scale_dust(
                            self.map_freqs[tag1],
                            self.map_freqs[tag2],
                            ref_freq=self.ref_freq,
                            beta=self.beta_ref,
                            delta_beta=delta_beta,
                        )
                        qbs = freq_scale * qb[stag]
                        if spec in ["ee", "bb"]:
                            qbm = freq_scale * qb[mstag]

                    elif comp == "res":
                        # if 'res_{}_{}'.format(spec,
                        # modify model by previously fit res, including
                        # off diagonals and SXN/NXS for nulls
                        s0, s1 = spec  # separate qbs for, eg, TE resTT and resEE
                        res_tags = {
                            "s0m0": "res_{}_{}".format(s0 * 2, tag1),
                            "s0m1": "res_{}_{}".format(s0 * 2, tag2),
                            "s1m0": "res_{}_{}".format(s1 * 2, tag1),
                            "s1m1": "res_{}_{}".format(s1 * 2, tag2),
                        }
                        qb_fac = {"s0m0": 1, "s0m1": 1, "s1m0": 1, "s1m1": 1}

                        for k, v in res_tags.items():
                            spec0 = v.split("_")[1]
                            if v not in qb:
                                if spec0 in ["ee", "bb"]:
                                    res_tags[k] = v.replace(spec0, "eebb")
                                    if res_tags[k] not in qb:
                                        # if not fitting EE/BB resids
                                        qb_fac[k] = 1
                                    else:
                                        qb_fac[k] = np.sqrt(1 + qb[res_tags[k]])[
                                            :, None
                                        ]
                                else:
                                    # this will happen for specs with T
                                    # with no T resids fit
                                    qb_fac[k] = 1
                            else:
                                qb_fac[k] = np.sqrt(1 + qb[v])[:, None]

                            if np.any(np.isnan(qb_fac[k])):
                                warnings.warn(
                                    "Unphysical residuals fit, "
                                    + "setting to zero {} {}".format(
                                        spec, np.where(np.isnan(qb_fac[k]))
                                    )
                                )
                                qb_fac[k][np.isnan(qb_fac[k])] = 1

                        # N_s0_map0 x N_s1_map1
                        cl1 = (
                            (qb_fac["s0m0"] * qb_fac["s1m1"] - 1)
                            * cbl["res0_nxn_{}".format(spec)][xname]
                        ).sum(axis=0)
                        # N_s1_map0 x N_s0_map1
                        cl1 += (
                            (qb_fac["s1m0"] * qb_fac["s0m1"] - 1)
                            * cbl["res1_nxn_{}".format(spec)][xname]
                        ).sum(axis=0)
                        if self.null_run:
                            # S_s0_map0 x N_s1_map1
                            cl1 += (
                                (qb_fac["s1m1"] - 1)
                                * cbl["res0_sxn_{}".format(spec)][xname]
                            ).sum(axis=0)
                            # S_s1_map0 x N_s0_map1
                            cl1 += (
                                (qb_fac["s0m1"] - 1)
                                * cbl["res1_sxn_{}".format(spec)][xname]
                            ).sum(axis=0)

                            # N_s0_map0 x S_s1_map1
                            cl1 += (
                                (qb_fac["s0m0"] - 1)
                                * cbl["res0_nxs_{}".format(spec)][xname]
                            ).sum(axis=0)
                            # N_s1_map0 x S_s0_map1
                            cl1 += (
                                (qb_fac["s1m0"] - 1)
                                * cbl["res1_nxs_{}".format(spec)][xname]
                            ).sum(axis=0)

                        # all of these were asymmetric specs, divide by 2 for mean
                        cl1 /= 2.0

                    # compute model spectra
                    if comp in ["cmb", "fg"]:
                        if xname not in cbl[stag]:
                            continue
                        cbl1 = cbl[stag][xname]
                        if isinstance(cbl1, dict):
                            # has beam error terms. deal with them individually
                            cl1 = OrderedDict()
                            cl1["b1"] = (qbs[:, None] * cbl1["b1"]).sum(axis=0)
                            cl1["b2"] = (qbs[:, None] * cbl1["b2"]).sum(axis=0)
                            cl1["b3"] = (qbs[:, None] * cbl1["b3"]).sum(axis=0)
                        else:
                            cl1 = (qbs[:, None] * cbl1).sum(axis=0)
                        if spec in ["ee", "bb"]:
                            # mixing terms, add in-place
                            if qbm is not None and mstag + "_mix" in cbl:
                                cbl1_mix = cbl[mstag + "_mix"][xname]
                                if isinstance(cbl1_mix, dict):
                                    cl1["b1"] += (qbm[:, None] * cbl1_mix["b1"]).sum(
                                        axis=0
                                    )
                                    cl1["b2"] += (qbm[:, None] * cbl1_mix["b2"]).sum(
                                        axis=0
                                    )
                                    cl1["b3"] += (qbm[:, None] * cbl1_mix["b3"]).sum(
                                        axis=0
                                    )
                                else:
                                    cl1 += (qbm[:, None] * cbl1_mix).sum(axis=0)

                    elif comp == "noise":
                        cl1 = cls_noise[spec][xname][: self.lmax + 1]

                    elif comp == "cond":
                        # add conditioner along diagonal
                        if tag1 != tag2:
                            continue
                        if spec == "tt":
                            cl1 = 10 * cls_cond
                        elif spec in ["ee", "bb"]:
                            cl1 = cls_cond
                        else:
                            continue

                    # store
                    cls.setdefault(stag, OrderedDict())[xname] = cl1

                    # add to total model
                    if not isinstance(cl1, dict):
                        ttag = "total_{}".format(spec)
                        cls[ttag].setdefault(xname, np.zeros_like(cl1))
                        cls[ttag][xname] += cl1
        return cls

    def get_data_spectra(self, map_tag=None, transfer_run=False, do_noise=True):
        """
        Return data and noise spectra for the given map tag(s).  Data spectra
        and signal/noise sim spectra must have been precomputed or loaded from
        disk.

        Arguments
        ---------
        map_tag : str
            If None, all map-map cross-spectra are included in the outputs.
            Otherwise, only the autospectra of the given map are included.
        transfer_run : bool
            If True, the data cls are the average of the signal simulations, and
            noise cls are ignored.  If False, the data cls are either
            `cls_data_null` (for null tests) or `cls_data`.  See
            `get_masked_xcorr` for how these are computed.  The input noise is
            similarly either `cls_noise_null` or `cls_noise`.
        do_noise : bool
            If True, return noise spectra along with data.

        Returns
        -------
        obs : OrderedDict
            Dictionary of data cross spectra
        nell : OrderedDict
            Dictionary of noise cross spectra, or None if transfer_run is True.
        """
        # select map pairs
        if map_tag is not None:
            map_tags = [map_tag]
        else:
            map_tags = self.map_tags
        map_pairs = pt.tag_pairs(map_tags)

        # select spectra
        tbeb = "cmb_tb" in self.bin_def
        if transfer_run or not tbeb:
            specs = self.specs[:4]
        else:
            specs = self.specs

        # obs depends on what you're computing
        if transfer_run:
            obs_quant = self.cls_signal
        elif self.null_run:
            if self.planck_sub:
                obs_quant = self.cls_data_sub_null
            else:
                obs_quant = self.cls_data_null
        elif self.template_cleaned:
            obs_quant = self.cls_data_clean
        else:
            obs_quant = self.cls_data

        # in case we're excluding some spectra or maps, repopulate obs dict
        obs = OrderedDict()
        for spec in specs:
            obs[spec] = OrderedDict()
            for xname in map_pairs:
                obs[spec][xname] = obs_quant[spec][xname]

        if not do_noise:
            return obs

        nell = None
        debias = None
        # Nulls are debiased by average of S+N sims
        if self.null_run and not transfer_run:
            if self.cls_noise is not None:
                nell = OrderedDict()
            debias = OrderedDict()
            for spec in specs:
                if self.cls_noise is not None:
                    nell[spec] = OrderedDict()
                debias[spec] = OrderedDict()
                for xname, (m0, m1) in map_pairs.items():
                    if m0 != m1:
                        if self.cls_noise is not None:
                            nell[spec][xname] = np.copy(self.cls_sim_null[spec][xname])
                        if self.planck_sub:
                            debias[spec][xname] = np.copy(
                                self.cls_noise_null[spec][xname]
                            )
                        else:
                            debias[spec][xname] = np.copy(
                                self.cls_sim_null[spec][xname]
                            )

                    else:
                        if self.cls_noise is not None:
                            nell[spec][xname] = np.copy(self.cls_sim_null[spec][xname])
                        if self.planck_sub:
                            debias[spec][xname] = np.copy(
                                self.cls_noise_null[spec][xname]
                            )
                        else:
                            debias[spec][xname] = np.copy(
                                self.cls_sim_null[spec][xname]
                            )

        # Non-nulls are debiased by average of N sims
        elif not transfer_run and self.cls_noise is not None:
            nell = OrderedDict()
            debias = OrderedDict()
            for spec in specs:
                nell[spec] = OrderedDict()
                debias[spec] = OrderedDict()
                for xname, (m0, m1) in map_pairs.items():
                    if m0 != m1:
                        # set non-auto noise to 0-- don't care to fit cross
                        # spectrum noise
                        nell[spec][xname] = np.zeros_like(self.cls_noise[spec][xname])
                    else:
                        nell[spec][xname] = np.copy(self.cls_noise[spec][xname])
                    debias[spec][xname] = np.copy(nell[spec][xname])

        return obs, nell, debias

    def do_qb2cb(
        self, qb, cls_shape, inv_fish=None, tophat_bins=False, return_cls=False
    ):
        """
        Compute binned output spectra and covariances by averaging
        the shape spectrum over each bin, and applying the appropriate
        `qb` bandpower amplitude.

        This method is used internally by `fisher_calc`, and requires
        bin definitions to have been pre-loaded using `get_bin_def`
        or `get_transfer`.

        Arguments
        ---------
        qb : dict
            Bandpower amplitudes for each spectrum bin.
        cls_shape : dict
            Shape spectrum
        inv_fish : array_like, (nbins, nbins)
            Inverse fisher matrix for computing the bin errors
            and covariance.  If not supplied, these are not computed.
        tophat_bins : bool
            If True, compute binned bandpowers using a tophat weight.
            Otherwise a logarithmic ell-space weighting is applied.
        return_cls : bool
            If True, return binned C_l spectrum rather than the default D_l

        Returns
        -------
        cb : dict of arrays
            Binned spectrum
        dcb : dict of arrays
            Binned spectrum error, if `inv_fish` is not None
        ellb : dict of arrays
            Average bin center
        cov : array_like, (nbins, nbins)
            Binned spectrum covariance, if `inv_fish` is not None
        qb2cb : dict
            The conversion factor from `qb` to `cb`, computed by
            averaging over the input shape spectrum.
        """
        return xft.bin_spec(
            qb,
            cls_shape,
            bin_def=self.bin_def,
            inv_fish=inv_fish,
            tophat=tophat_bins,
            lfac=not return_cls,
        )

    def fisher_precalc(self, cbl, cls_input, cls_noise=None, likelihood=False):
        """
        Pre-compute the D_ell and signal derivative matrices necessary for
        `fisher_calc` from the input data spectra.  This method requires bin
        definitions precomputed by `get_bin_def` or `get_transfer`.

        Arguments
        ---------
        cbl : OrderedDict
            Cbl dict computed by `bin_cl_template` for a given
            shape spectrum.
        cls_input : OrderedDict
            Input spectra.  If computing a transfer function, this is the
            average `cls_signal`.  If computing a null test, this is
            `cls_data_null`, and otherwise it is `cls_data`, for a single map or
            several input maps.
        cls_noise : OrderedDict
            If supplied, the noise spectrum is subtracted from the input.
        likelihood : bool
            If True, compute just Dmat_obs_b.  Otherwise, Dmat_obs and
            dSdqb_mat1 are also computed.

        Returns
        -------
        Dmat_obs : OrderedDict
            De-biased D_ell matrix from `cls_input`
        Dmat_obs_b : OrderedDict
            Biased D_ell matrix from `cls_input` (for likelihood)
        dSdqb_mat1 : OrderedDict
            Signal derivative matrix from Cbl

        NB: the output arrays are also stored as attributes of the
        parent object to avoid repeating the computation in fisher_calc
        """
        num_maps = self.num_maps
        pol_dim = 3 if self.pol else 1
        dim1 = pol_dim * num_maps

        comps = ["cmb"]
        if "fg_tt" in cbl:
            comps += ["fg"]
        if "res_tt" in cbl or "res_ee" in cbl:
            comps += ["res"]

        specs = list(cls_input)

        if likelihood:
            Dmat_obs_b = OrderedDict()
            Dmat_obs = None
            dSdqb = None
        else:
            Dmat_obs_b = None
            Dmat_obs = OrderedDict()
            dSdqb = OrderedDict()

        for xname, (m0, m1) in self.map_pairs.items():
            # transfer function doesn't have all the crosses
            if xname not in cls_input[specs[0]]:
                continue

            if likelihood:
                Dmat_obs_b[xname] = OrderedDict()
            else:
                Dmat_obs[xname] = OrderedDict()

            for spec in specs:
                if likelihood:
                    # without bias subtraction for likelihood
                    Dmat_obs_b[xname][spec] = cls_input[spec][xname]
                else:
                    if cls_noise is not None:
                        Dmat_obs[xname][spec] = (
                            cls_input[spec][xname] - cls_noise[spec][xname]
                        )
                    else:
                        Dmat_obs[xname][spec] = cls_input[spec][xname]

            if likelihood:
                continue

            for comp in comps:
                for spec in specs:
                    stag = "{}_{}".format(comp, spec)
                    if stag not in cbl:
                        continue
                    if xname not in cbl[stag]:
                        continue

                    dSdqb.setdefault(comp, OrderedDict()).setdefault(
                        xname, OrderedDict()
                    ).setdefault(spec, OrderedDict())
                    dSdqb[comp][xname][spec][spec] = cbl[stag][xname]

                    if spec in ["ee", "bb"]:
                        if stag + "_mix" not in cbl:
                            continue
                        mspec = "bb" if spec == "ee" else "ee"
                        mix_cbl = cbl[stag + "_mix"][xname]
                        dSdqb[comp][xname][spec][mspec] = mix_cbl

                if comp == "fg":
                    # add delta beta bin for spectral index
                    dSdqb.setdefault("delta_beta", OrderedDict()).setdefault(
                        xname, OrderedDict()
                    )
                    for spec in specs:
                        # this will be filled in in fisher_calc
                        dSdqb["delta_beta"][xname][spec] = OrderedDict()
        return Dmat_obs, Dmat_obs_b, dSdqb

    def clear_precalc(self):
        """
        Clear variables pre-computed with `fisher_precalc`.
        """
        self.Dmat_obs = None
        self.Dmat_obs_b = None
        self.dSdqb_mat1 = None

    def fisher_calc(
        self,
        qb,
        cbl,
        cls_input,
        cls_noise=None,
        cls_debias=None,
        cls_model=None,
        cond_noise=None,
        cond_criteria=None,
        likelihood=False,
        like_lmin=2,
        like_lmax=None,
        delta_beta_prior=None,
        null_first_cmb=False,
        use_precalc=True,
    ):
        """
        Re-compute the Fisher matrix and qb amplitudes based on
        input data spectra.  This method is called iteratively
        by `fisher_iterate`, and requires bin definitions precomputed
        by `get_bin_def` or `get_transfer`.

        Arguments
        ---------
        qb : OrderedDict
            Bandpower amplitudes, typically computed in a previous call
            to this method.
        cbl : OrderedDict
            Cbl matrix computed by `bin_cl_template` for a given
            shape spectrum.
        cls_input : OrderedDict
            Input spectra.  If computing a transfer function,
            this is the average `cls_signal`.  If computing a null
            test, this is `cls_data_null`, and otherwise it is
            `cls_data`, for a single map or several input maps.
        cls_noise : OrderedDict
            If supplied, the noise spectrum is applied to the model spectrum.
        cls_debias : OrderedDict
            If supplied, the debias spectrum is subtracted from the input.
        cond_criteria : float
            The maximum condition number allowed for Dmat1 to be acceptable
            for taking its inverse.
        likelihood : bool
            If True, return the likelihood for the given input bandpowers, shapes
            and data spectra.  Otherwise, computes output bandpowers and the fisher
            covariance for a NR iteration.
        use_precalc : bool
            If True, load pre-calculated terms stored from a previous iteration,
            and store for a future iteration.  Otherwise, all calculations are
            repeated.

        Returns
        -------
        qb : OrderedDict
            New bandpower amplitudes
        inv_fish : array_like
            Inverse Fisher correlation matrix over all bins
        -- or --
        likelihood : scalar
            Likelihood of the given input parameters.
        """
        if cond_criteria is None:
            cond_criteria = np.inf
        well_cond = False

        pol_dim = 3 if self.pol else 1
        do_fg = "fg_tt" in cbl

        dkey = "Dmat_obs_b" if likelihood else "Dmat_obs"

        if getattr(self, dkey, None) is None or not use_precalc:
            Dmat_obs, Dmat_obs_b, dSdqb_mat1 = self.fisher_precalc(
                cbl,
                cls_input,
                cls_noise=cls_debias if not likelihood else None,
                likelihood=likelihood,
            )
            if use_precalc:
                self.Dmat_obs = Dmat_obs
                self.Dmat_obs_b = Dmat_obs_b
                self.dSdqb_mat1 = dSdqb_mat1
        else:
            if likelihood:
                Dmat_obs_b = self.Dmat_obs_b
            else:
                Dmat_obs = self.Dmat_obs
                dSdqb_mat1 = self.dSdqb_mat1

        delta_beta = 0.0
        if "delta_beta" in qb:
            delta_beta = qb["delta_beta"][0]

        if not likelihood:
            dSdqb_mat1_freq = copy.deepcopy(dSdqb_mat1)

        if likelihood or not cond_noise:
            well_cond = True
            cond_noise = None

        gmat_ell = OrderedDict()
        Dmat1 = OrderedDict()

        if cls_model is None:
            cls_model = self.get_model_spectra(
                qb, cbl, delta=True, cls_noise=cls_noise, cond_noise=cond_noise
            )

        mkeys = list(cls_model)
        for xname, (m0, m1) in self.map_pairs.items():
            # transfer function does not have crosses
            if xname not in cls_model[mkeys[0]]:
                continue
            gmat_ell[xname] = self.gmat_ell[xname]

            if well_cond:
                for spec in self.specs:
                    # transfer function doesn't have eb/tb
                    if "total_{}".format(spec) not in cls_model:
                        continue
                    Dmat1.setdefault(xname, OrderedDict())
                    Dmat1[xname][spec] = cls_model["total_{}".format(spec)][xname]

        if well_cond:
            Dmat1_mat = pt.dict_to_dmat(Dmat1)

        # Set up dSdqb frequency dependence
        for xname, (m0, m1) in self.map_pairs.items():
            # transfer function does not have crosses
            if xname not in cls_model[mkeys[0]]:
                continue

            if do_fg and not likelihood:
                # Add spectral index dependence
                # dSdqb now depends on qb (spec index) because
                # model is non-linear so cannot be precomputed.

                # get foreground at pivot point spectral index
                # and first derivative
                freq_scale0, freq_scale_deriv = xft.scale_dust(
                    self.map_freqs[m0],
                    self.map_freqs[m1],
                    ref_freq=self.ref_freq,
                    beta=self.beta_ref,
                    deriv=True,
                )
                freq_scale = freq_scale0 + delta_beta * freq_scale_deriv
                freq_scale_ratio = freq_scale_deriv / freq_scale

                # scale foreground model by frequency scaling adjusted for beta
                for s1, sdat in dSdqb_mat1_freq["fg"][xname].items():
                    for s2, sdat2 in sdat.items():
                        dSdqb_mat1_freq["fg"][xname][s1][s2] *= freq_scale

                # build delta_beta term from frequency scaled model,
                # divide out frequeny scaling and apply derivative term
                for s1, sdat in dSdqb_mat1_freq["delta_beta"][xname].items():
                    sdat[s1] = cls_model["fg_{}".format(s1)][xname] * freq_scale_ratio

        # Set up Dmat -- if it's not well conditioned, add noise to the
        # diagonal until it is.
        cond_iter = 0
        while not well_cond:
            cls_model = self.get_model_spectra(
                qb, cbl, delta=True, cls_noise=cls_noise, cond_noise=cond_noise
            )

            for xname, (m0, m1) in self.map_pairs.items():
                # transfer function does not have crosses
                if xname not in cls_model[mkeys[0]]:
                    continue

                for spec in self.specs:
                    # transfer function doesn't have eb/tb
                    if "total_{}".format(spec) not in cls_model:
                        continue
                    Dmat1.setdefault(xname, OrderedDict())
                    Dmat1[xname][spec] = cls_model["total_{}".format(spec)][xname]

            Dmat1_mat = pt.dict_to_dmat(Dmat1)

            npv = np.__version__.split(".")[:3]
            npv = [int(x) for x in npv]
            if npv > [1, 10, 0]:
                cond = np.abs(
                    np.linalg.cond(Dmat1_mat[:, :, self.lmin :].swapaxes(0, -1))
                )
            else:
                # old numpy can't do multi-dimensional arrays
                cond = np.asarray(
                    [
                        np.abs(np.linalg.cond(x))
                        for x in Dmat1_mat[:, :, self.lmin :].swapaxes(0, -1)
                    ]
                )

            cond = np.max(cond)
            if cond > cond_criteria and cond_noise:
                cond_iter += 1
                # cond_noise iteration factor found through trial and error
                cond_noise *= 10 ** (cond_iter / 100.0)
                if cond_iter == 1:
                    self.log(
                        "Condition criteria not met. "
                        "Max Cond={:.0f}, Thresh={:.0f}".format(cond, cond_criteria),
                        "detail",
                    )
            else:
                well_cond = True
                self.log(
                    "Condition criteria met. "
                    "Max Cond={:.0f}, Thresh={:.0f}, Iter={:d}".format(
                        cond, cond_criteria, cond_iter
                    ),
                    "detail",
                )
                if cond_noise is not None:
                    self.log("Cond_noise = {:.3e}".format(cond_noise), "detail")

        # construct arrays from dictionaries
        Dmat1 = Dmat1_mat
        gmat = pt.dict_to_dmat(gmat_ell)
        if likelihood:
            Dmat_obs_b = pt.dict_to_dmat(Dmat_obs_b)
        else:
            Dmat_obs = pt.dict_to_dmat(Dmat_obs)
            dSdqb_mat1_freq = pt.dict_to_dsdqb_mat(dSdqb_mat1_freq, self.bin_def)
        # apply ell limits
        if likelihood:
            ell = slice(
                self.lmin if like_lmin is None else like_lmin,
                self.lmax + 1 if like_lmax is None else like_lmax + 1,
            )
        else:
            ell = slice(self.lmin, self.lmax + 1)
        Dmat1 = Dmat1[..., ell]
        if likelihood:
            Dmat_obs_b = Dmat_obs_b[..., ell]
        else:
            Dmat_obs = Dmat_obs[..., ell]
            dSdqb_mat1_freq = dSdqb_mat1_freq[..., ell]
        gmat = gmat[..., ell]

        self.Dmat1 = Dmat1

        lam, R = np.linalg.eigh(Dmat1.swapaxes(0, -1))
        bad = (lam <= 0).sum(axis=-1).astype(bool)
        if bad.sum():
            # exclude any ell's with ill-conditioned D matrix
            # this should happen only far from max like
            bad_idx = np.unique(np.where(bad)[0])
            bad_ells = np.arange(ell.start, ell.stop)[bad_idx]
            self.log("Found negative eigenvalues at ells {}".format(bad_ells))
            gmat[..., bad_idx] = 0
        inv_lam = 1.0 / lam
        Dinv = np.einsum("...ij,...j,...kj->...ik", R, inv_lam, R).swapaxes(0, -1)

        if likelihood:
            # log(det(D)) = tr(log(D)), latter is numerically stable
            # compute log(D) by eigenvalue decomposition per ell
            log_lam = np.log(lam)
            Dlog = np.einsum("...ij,...j,...kj->...ik", R, log_lam, R).swapaxes(0, -1)

        else:
            # compute ell-by-ell inverse
            # Dinv = np.linalg.inv(Dmat1.swapaxes(0, -1)).swapaxes(0, -1)

            # optimized matrix multiplication
            # there is something super weird about this whole matrix operation
            # that causes the computation of mats to take four times as long
            # if mat1 is not computed.
            eye = np.eye(len(gmat))
            mat1 = np.einsum("ij...,jk...->ik...", eye, Dinv)
            mat2 = np.einsum("klm...,ln...->knm...", dSdqb_mat1_freq, Dinv)
            mat = np.einsum("ik...,knm...->inm...", mat1, mat2)

        if getattr(self, "marg_table", None) is not None:
            marg_table = OrderedDict()
            for xname, (m0, m1) in self.map_pairs.items():
                if xname not in cls_model[mkeys[0]]:
                    continue
                marg_table.setdefault(xname, OrderedDict())
                for spec in self.specs:
                    marg_fac = self.marg_table.get(spec, {}).get(xname, 0)
                    marg_table[xname][spec] = marg_fac

            marg_mask = np.logical_not(pt.dict_to_dmat(marg_table).astype(bool))
            if not likelihood:
                Dmat_obs *= marg_mask[..., None]
                dSdqb_mat1_freq *= marg_mask[..., None, None]
            gmat *= marg_mask[..., None]

        if likelihood:
            # compute log likelihood as tr(g * (D^-1 * Dobs + log(D)))
            arg = np.einsum("ij...,jk...->ik...", Dinv, Dmat_obs_b) + Dlog
            like = -np.einsum("iij,iij->", gmat, arg) / 2.0

            # include priors in likelihood
            if "delta_beta" in qb and delta_beta_prior is not None:
                chi = (qb["delta_beta"] - self.delta_beta_fix) / delta_beta_prior
                like -= chi ** 2 / 2.0

            if null_first_cmb:
                for spec in self.specs:
                    stag = "cmb_{}".format(spec)
                    if stag not in qb:
                        continue
                    chi = (qb[stag][0] - 1) / np.sqrt(1e-10)
                    like -= chi ** 2 / 2.0

            return like

        # construct matrices for the qb and fisher terms,
        # and take the trace and sum over ell simultaneously
        qb_vec = np.einsum("iil,ijkl,jil->k", gmat, mat, Dmat_obs) / 2.0
        fisher = np.einsum("iil,ijkl,jiml->km", gmat, mat, dSdqb_mat1_freq) / 2

        bin_index = pt.dict_to_index(qb)

        if "delta_beta" in qb and delta_beta_prior is not None:
            # XXX need documentation for what happens here
            # for imposing the delta_beta prior
            sl = slice(*(bin_index["delta_beta"]))
            d = 1.0 / delta_beta_prior ** 2
            qb_vec[sl] += d * self.delta_beta_fix
            fisher[sl, sl] += d

        if null_first_cmb:
            # blow up the fisher matrix for first bin
            # by implementing a tight prior
            for spec in self.specs:
                stag = "cmb_{}".format(spec)
                if stag in bin_index:  # check for transfer function
                    b0 = bin_index[stag][0]
                    fisher[b0, b0] += 1e10
                    qb_vec[b0] += 1e10

        # invert
        qb_vec = np.linalg.solve(fisher, qb_vec)
        inv_fish = np.linalg.solve(fisher, np.eye(len(qb_vec)))
        qb_vec = pt.arr_to_dict(qb_vec, qb)

        return qb_vec, inv_fish

    def fisher_iterate(
        self,
        cbl,
        cls_shape,
        map_tag=None,
        iter_max=200,
        converge_criteria=0.005,
        qb_start=None,
        transfer_run=False,
        save_iters=False,
        tophat_bins=False,
        return_cls=False,
        null_first_cmb=False,
        delta_beta_prior=None,
        cond_noise=None,
        cond_criteria=None,
        like_profiles=False,
        like_profile_sigma=3.0,
        like_profile_points=100,
        file_tag=None,
    ):
        """
        Iterate over the Fisher calculation to compute bandpower estimates
        assuming an input shape spectrum.

        Arguments
        ---------
        cbl : OrderedDict
            Cbl matrix computed from an input shape spectrum
        cls_shape : OrderedDict
            Input shape spectrum
        map_tag : str
            If not None, then iteration is performed over the spectra
            corresponding to the given map, rather over all possible
            combinations of map-map cross-spectra. In this case, the first
            dimension of the input cbl must be of size 1 (this is done
            automatically by calling `bin_cl_template(..., map_tag=<map_tag>)`.
        iter_max : int
            Maximum number of iterations to perform.  if this limit is
            reached, a warning is issued.
        converge_criteria : float
            Maximum fractional change in qb that indicates convergence and
            stops iteration.
        qb_start : OrderedDict
            Initial guess at `qb` bandpower amplitudes.  If None, unity is
            assumed for all bins.
        transfer_run : bool
            If True, the input Cls passed to `fisher_calc` are the average
            of the signal simulations, and noise cls are ignored.
            If False, the input Cls are either `cls_data_null`
            (for null tests) or `cls_data`.  See `get_masked_xcorr` for
            how these are computed.  The input noise is similarly either
            `cls_noise_null` or `cls_noise`.
        save_iters : bool
            If True, the output data from each Fisher iteration are stored
            in an individual npz file.
        tophat_bins : bool
            If True, use uniform weights within each bin rather than any
            ell-based weighting.
        return_cls : bool
            If True, return C_l spectrum rather than D_l spectrum
        cond_criteria : float
            The maximum condition number allowed for Dmat1 to be acceptable
            for taking its inverse.
        like_profiles : bool
            If True, compute profile likelihoods for each qb, leaving all
            others fixed at their maximum likelihood values.  Profiles are
            computed over a range +/--sigma as estimated from the diagonals
            of the inverse Fisher matrix.
        like_profile_sigma : float
            Range in units of 1sigma over which to compute profile likelihoods
        like_profile_points : int
            Number of points to sample along the likelihood profile
        file_tag : string
            If supplied, appended to the bandpowers filename.

        Returns
        -------
        data : dict
            The results of the Fisher iteration process.  This dictionary
            contains the fields:
                qb : converged bandpower amplitudes
                cb : output binned spectrum
                dcb : binned spectrum errors
                ellb : bin centers
                cov : spectrum covariance
                inv_fish : inverse fisher matrix
                fqb : fractional change in qb for the last iteration
                qb2cb : conversion factor
                cbl : Cbl matrix
                cls_model : unbinned model spectrum computed from Cbl
                bin_def : bin definition array
                cls_obs : observed input spectra
                cls_noise : noise spectra
                cls_shape : shape spectrum
                iters : number of iterations completed
            If `transfer_run` is False, this dictionary also contains:
                qb_transfer : transfer function amplitudes
                nbins_res : number of residual bins
            If `sim_index` is not None, this dictionary also contains:
                simerr_qb : error of qb for each simulation

        Data Handling
        -------------
        This method stores outputs to files with name 'transfer' for transfer
        function runs (if `transfer_run = True`), otherwise with name
        'bandpowers'.  Outputs from each individual iteration, containing
        only the quantities that change with each step, are stored in
        separate files with the same name (and different index).
        """

        save_name = "transfer" if transfer_run else "bandpowers"

        if transfer_run:
            null_first_cmb = False

        # previous fqb iterations to monitor convergence and adjust conditioning
        prev_fqb = []
        cond_adjusted = False

        if qb_start is None:
            qb = OrderedDict()
            for k, v in self.bin_def.items():
                if transfer_run:
                    if "cmb" not in k or "eb" in k or "tb" in k:
                        continue
                if k == "delta_beta":
                    # qb_delta beta is a coefficient on the change from beta,
                    # so expect that it should be small if beta_ref is close
                    # (zeroes cause singular matrix problems)
                    qb[k] = [self.delta_beta_fix]
                elif k.startswith("res_") or k.startswith("fg_"):
                    # res qb=0 means noise model is 100% accurate.
                    qb[k] = 1e-5 * np.ones(len(v))
                else:
                    # start by assuming model is 100% accurate
                    qb[k] = np.ones(len(v))
        else:
            qb = qb_start

        obs, nell, debias = self.get_data_spectra(
            map_tag=map_tag, transfer_run=transfer_run
        )

        # initialize matrices for precomputation
        self.clear_precalc()

        bin_index = pt.dict_to_index(self.bin_def)

        success = False
        for iter_idx in range(iter_max):
            self.log(
                "Doing Fisher step {}/{}...".format(iter_idx + 1, iter_max), "part"
            )

            qb_new, inv_fish = self.fisher_calc(
                qb,
                cbl,
                obs,
                cls_noise=nell,
                cls_debias=debias,
                cond_noise=cond_noise,
                delta_beta_prior=delta_beta_prior,
                cond_criteria=cond_criteria,
                null_first_cmb=null_first_cmb,
            )

            qb_arr = pt.dict_to_arr(qb, flatten=True)
            qb_new_arr = pt.dict_to_arr(qb_new, flatten=True)
            dqb = qb_new_arr - qb_arr
            fqb = dqb / qb_arr
            max_fqb = np.nanmax(np.abs(fqb))

            prev_fqb.append(max_fqb)

            fnan = np.isnan(fqb)
            if fnan.any():
                (nanidx,) = np.where(fnan)
                self.log(
                    "Ignoring {} bins with fqb=nan: bins={}, qb_new={}, "
                    "qb={}".format(
                        len(nanidx), nanidx, qb_new_arr[nanidx], qb_arr[nanidx]
                    )
                )

            self.log("Max fractional change in qb: {}".format(max_fqb), "part")

            # put qb_new in original dict
            qb = copy.deepcopy(qb_new)
            cls_model = self.get_model_spectra(
                qb, cbl, delta=True, cls_noise=nell, cond_noise=None
            )

            cb, dcb, ellb, cov, qb2cb = self.do_qb2cb(
                qb, cls_shape, inv_fish, tophat_bins=tophat_bins, return_cls=return_cls
            )

            if "delta_beta" in qb:
                # get beta fit and beta error
                beta_fit = qb["delta_beta"][0] + self.beta_ref
                db_idx = slice(*bin_index["delta_beta"])
                beta_err = np.sqrt(np.diag(inv_fish[db_idx, db_idx]))[0]
            else:
                beta_fit = None
                beta_err = None

            if save_iters:
                # save only the quantities that change with each iteration
                self.save_data(
                    save_name,
                    bp_opts=not transfer_run,
                    map_tag=map_tag,
                    map_tags=self.map_tags,
                    iter_index=iter_idx,
                    bin_def=self.bin_def,
                    cls_shape=cls_shape,
                    cls_obs=obs,
                    ellb=ellb,
                    qb=qb,
                    fqb=fqb,
                    cb=cb,
                    dcb=dcb,
                    qb2cb=qb2cb,
                    inv_fish=inv_fish,
                    cov=cov,
                    cls_model=cls_model,
                    cbl=cbl,
                    beta_fit=beta_fit,
                    beta_err=beta_err,
                    ref_freq=self.ref_freq,
                    beta_ref=self.beta_ref,
                    map_freqs=self.map_freqs,
                    cls_signal=self.cls_signal,
                    cls_noise=self.cls_noise,
                    Dmat_obs=self.Dmat_obs,
                    gmat_ell=self.gmat_ell,
                    marg_table=getattr(self, "marg_table", None),
                    extra_tag=file_tag,
                )

            (nans,) = np.where(np.isnan(qb_new_arr))
            if len(nans):
                msg = "Found NaN values in qb bins {} at iter {}".format(nans, iter_idx)
                break

            if fnan.all():
                msg = "All bins have fqb=NaN, something has gone horribly wrong."
                break

            negs = np.where(np.diag(inv_fish) < 0)[0]
            if len(negs):
                self.log(
                    "Found negatives in inv_fish diagonal at locations "
                    "{}".format(negs)
                )

            if np.nanmax(np.abs(fqb)) < converge_criteria:
                if not transfer_run:
                    # Calculate final fisher matrix without conditioning
                    self.log("Calculating final Fisher matrix.")
                    _, inv_fish = self.fisher_calc(
                        qb,
                        cbl,
                        obs,
                        cls_noise=nell,
                        cls_debias=debias,
                        cond_noise=None,
                        delta_beta_prior=delta_beta_prior,
                        null_first_cmb=null_first_cmb,
                    )

                    cb, dcb, ellb, cov, qb2cb = self.do_qb2cb(
                        qb,
                        cls_shape,
                        inv_fish,
                        tophat_bins=tophat_bins,
                        return_cls=return_cls,
                    )

                # If any diagonals of inv_fisher are negative, something went wrong
                negs = np.where(np.diag(inv_fish) < 0)[0]
                if len(negs):
                    msg = (
                        "Found negatives in inv_fish diagonal at locations "
                        "{}".format(negs)
                    )
                    self.log(msg)
                    # break

                success = True
                break

            else:
                msg = "{} {} did not converge in {} iterations".format(
                    "Multi-map" if map_tag is None else "Map {}".format(map_tag),
                    "transfer function" if transfer_run else "spectrum",
                    iter_max,
                )
                # Check the slope of the last ten fqb_maxpoints.
                # If there's not a downward trend, adjust conditioning
                # criteria to help convergence.
                if len(prev_fqb) <= 10 or transfer_run:
                    continue
                m, b = np.polyfit(np.arange(10), prev_fqb[-10:], 1)
                if m > 0:  # Not converging
                    # First, start from very little conditioning
                    if not cond_adjusted:
                        cond_criteria = 5e3
                        cond_adjusted = True
                        self.log(
                            "Not converging. Setting cond_criteria={}".format(
                                cond_criteria
                            ),
                            "part",
                        )

                    elif cond_criteria > 100:
                        cond_criteria /= 2.0
                        self.log(
                            "Tightening condition criteria to help convergence. "
                            "cond_criteria={}".format(cond_criteria),
                            "part",
                        )
                    else:
                        self.log("Can't reduce cond_criteria any more.")
                    # give it ten tries to start converging
                    prev_fqb = []

        # save and return
        out = dict(
            qb=qb,
            cb=cb,
            dcb=dcb,
            ellb=ellb,
            cov=cov,
            inv_fish=inv_fish,
            fqb=fqb,
            qb2cb=qb2cb,
            bin_def=self.bin_def,
            iters=iter_idx,
            success=success,
            beta_fit=beta_fit,
            beta_err=beta_err,
            map_tags=self.map_tags,
            ref_freq=self.ref_freq,
            beta_ref=self.beta_ref,
            map_freqs=self.map_freqs,
            marg_table=getattr(self, "marg_table", None),
            converge_criteria=converge_criteria,
            delta_beta_prior=delta_beta_prior,
            cond_noise=cond_noise,
            cond_criteria=cond_criteria,
            null_first_cmb=null_first_cmb,
            apply_gcorr=self.apply_gcorr,
            weighted_bins=self.weighted_bins,
        )

        if self.debug:
            out.update(
                cbl=cbl,
                cls_obs=obs,
                cls_signal=self.cls_signal,
                cls_noise=self.cls_noise,
                cls_model=cls_model,
                cls_shape=cls_shape,
                cond_noise=cond_noise,
                Dmat_obs=self.Dmat_obs,
            )

        if not transfer_run:
            out.update(qb_transfer=self.qb_transfer)
            if self.template_cleaned:
                out.update(
                    template_alpha90=self.template_alpha90,
                    template_alpha150=self.template_alpha150,
                )

        if success and not transfer_run:
            # do one more fisher calc that doesn't include sample variance
            # set qb=very close to 0. 0 causes singular matrix problems.
            # don't do this for noise residual bins
            self.log("Calculating final Fisher matrix without sample variance.")
            qb_zeroed = copy.deepcopy(qb)
            qb_new_ns = copy.deepcopy(qb)
            for comp in ["cmb", "fg"]:
                for spec in self.specs:
                    stag = "{}_{}".format(comp, spec)
                    if stag not in qb_zeroed:
                        continue
                    qb_zeroed[stag][:] = 1e-20
                    qb_new_ns[stag][:] = 1.0
            if "delta_beta" in qb:
                qb_zeroed["delta_beta"][:] = 1e-20
                qb_new_ns["delta_beta"][:] = 0

            _, inv_fish_ns = self.fisher_calc(
                qb_zeroed,
                cbl,
                obs,
                cls_noise=nell,
                cls_debias=debias,
                cond_noise=None,
                delta_beta_prior=None,
                null_first_cmb=null_first_cmb,
            )

            _, dcb_nosampvar, _, cov_nosampvar, _ = self.do_qb2cb(
                qb_new_ns,
                cls_shape,
                inv_fish_ns,
                tophat_bins=tophat_bins,
                return_cls=return_cls,
            )

            out.update(
                dcb_nosampvar=dcb_nosampvar,
                cov_nosampvar=cov_nosampvar,
                invfish_nosampvar=inv_fish_ns,
            )

            if like_profiles:
                # compute bandpower likelihoods
                max_like = self.fisher_calc(
                    qb,
                    cbl,
                    obs,
                    cls_noise=nell,
                    cond_noise=None,
                    delta_beta_prior=delta_beta_prior,
                    null_first_cmb=null_first_cmb,
                    likelihood=True,
                )

                dqb = pt.arr_to_dict(np.sqrt(np.abs(np.diag(inv_fish))), qb)
                qb_like = OrderedDict()

                for stag, qbs in qb.items():
                    qb_like[stag] = np.zeros(
                        (len(qbs), 2, like_profile_points), dtype=float
                    )

                    for ibin, q in enumerate(qbs):
                        qb1 = copy.deepcopy(qb)
                        dq = dqb[stag][ibin] * like_profile_sigma
                        q_arr = np.linspace(q - dq, q + dq, like_profile_points)
                        like_arr = np.zeros_like(q_arr)

                        for iq, q1 in enumerate(q_arr):
                            qb1[stag][ibin] = q1
                            try:
                                like = self.fisher_calc(
                                    qb1,
                                    cbl,
                                    obs,
                                    cls_noise=nell,
                                    cond_noise=None,
                                    delta_beta_prior=delta_beta_prior,
                                    null_first_cmb=null_first_cmb,
                                    likelihood=True,
                                )
                            except np.linalg.LinAlgError:
                                like = np.nan

                            like_arr[iq] = like

                            self.log(
                                "{} bin {} delta qb {} delta like: {}".format(
                                    stag, ibin, q1 - q, like - max_like
                                ),
                                "detail",
                            )

                        qb_like[stag][ibin] = np.vstack([q_arr, like_arr])

                out.update(max_like=max_like, qb_like=qb_like)

        if not success:
            save_name = "ERROR_{}".format(save_name)
            self.log("ERROR: {}".format(msg), "info")
            warnings.warn(msg)

        # cleanup
        self.clear_precalc()

        return self.save_data(
            save_name, map_tag=map_tag, bp_opts=True, extra_tag=file_tag, **out
        )

    def get_transfer(
        self,
        cls_shape,
        converge_criteria=0.005,
        iter_max=200,
        save_iters=False,
        fix_bb_xfer=False,
        tophat_bins=False,
        return_cls=False,
    ):
        """
        Compute the transfer function from signal simulations created using
        the same spectrum as the input shape.

        This raises a ValueError if a negative transfer function amplitude
        is found.

        Arguments
        ---------
        cls_shape : OrderedDict
            Input shape spectrum.  Must match the input spectrum used to
            generate the signal simulations.
        converge_criteria : float
            Maximum fractional change in qb that indicates convergence and
            stops iteration.
        iter_max : int
            Maximum number of iterations to perform.  if this limit is
            reached, a warning is issued.
        save_iters : bool
            If True, the output data from each Fisher iteration are stored
            in an individual npz file.
        fix_bb_xfer : bool
            If True, after transfer functions have been calculated, impose
            the BB xfer is exactly equal to the EE transfer.
        tophat_bins : bool
            If True, use uniform weights within each bin rather than any
            ell-based weighting.
        return_cls : bool
            If True, return C_l spectrum rather than D_l spectrum

        Returns
        -------
        qb_transfer : OrderedDict
            Binned transfer function for each map

        Data Handling
        -------------
        This method is called at the 'transfer' checkpoint, and loads or saves
        a data dictionary named 'transfer_all' with the following entries:

            nbins : int
                number of bins
            bin_def : (nbins, 3)
                bin definition array (see `get_bin_def`)
            qb_transfer : (num_maps, nbins)
                binned transfer function for each map

        Additionally the final output of `fisher_iterate` is stored
        in a dictionary called `transfer_map<idx>` for each map.
        """
        self.return_cls = return_cls

        transfer_shape = (
            self.num_maps * len(self.specs),
            self.nbins_cmb / len(self.specs),
        )

        opts = dict(
            converge_criteria=converge_criteria,
            fix_bb_xfer=fix_bb_xfer,
            apply_gcorr=self.apply_gcorr,
            weighted_bins=self.weighted_bins,
        )

        save_name = "transfer_all"
        if self.weighted_bins:
            save_name = "{}_wbins".format(save_name)

        ret = self.load_data(
            save_name,
            "transfer",
            to_attrs=False,
            shape_ref="qb_transfer",
            shape=transfer_shape,
            value_ref=opts,
        )

        if ret is not None:
            self.qb_transfer = ret["qb_transfer"]
            return ret["qb_transfer"]

        self.qb_transfer = OrderedDict()
        for spec in self.specs:
            self.qb_transfer["cmb_" + spec] = OrderedDict()

        success = False
        msg = ""

        for im0, m0 in enumerate(self.map_tags):
            if self.map_reobs_freqs[m0] in self.planck_freqs:
                # DEBUG: Should we let this float since we think transfer
                # function computation is correcting for non-ideal computation
                # of mask kernel for non-axial-symmetric mask?
                for spec in self.specs:
                    self.qb_transfer["cmb_{}".format(spec)][m0] = np.ones(
                        self.nbins_cmb // len(self.specs)
                    )
                self.log("Setting Planck {} map transfer to unity".format(m0))
                success = True
                continue

            self.log(
                "Computing transfer function for map {}/{}".format(
                    im0 + 1, self.num_maps
                ),
                "part",
            )
            cbl = self.bin_cl_template(cls_shape, m0, transfer_run=True)
            ret = self.fisher_iterate(
                cbl,
                cls_shape,
                m0,
                transfer_run=True,
                iter_max=iter_max,
                converge_criteria=converge_criteria,
                save_iters=save_iters,
                tophat_bins=tophat_bins,
                return_cls=return_cls,
            )
            qb = ret["qb"]

            success = ret["success"]
            if not success:
                msg = "Error in fisher_iterate for map {}".format(m0)

            # fix negative amplitude bins
            for k, v in qb.items():
                if np.any(v < 0):
                    (negbin,) = np.where(v < 0)
                    warnings.warn(
                        "Transfer function amplitude {}".format(v)
                        + "< 0 for {} bin {} of map {}".format(k, negbin, m0)
                    )
                    # XXX cludge
                    # This happens in first bin
                    # try linear interp between zero and next value
                    try:
                        qb[k][negbin] = qb[k][negbin + 1] / 2.0
                        warnings.warn(
                            "Setting Transfer function in negative bin to small "
                            "positive. This is probably due to choice of bins or "
                            "insufficient number of signal sims"
                        )
                    except Exception as e:
                        msg = "Unable to adjust negative bins for map {}: {}".format(
                            m0, str(e)
                        )
                        success = False

            # fix the BB transfer to EE, if desired
            if fix_bb_xfer:
                qb["cmb_bb"] = qb["cmb_ee"]

            # fix TB/EB transfer functions
            if len(self.specs) > 4:
                qb["cmb_eb"] = np.sqrt(np.abs(qb["cmb_ee"] * qb["cmb_bb"]))
                qb["cmb_tb"] = np.sqrt(np.abs(qb["cmb_tt"] * qb["cmb_bb"]))

            for stag, qbdat in qb.items():
                self.qb_transfer[stag][m0] = qbdat

        self.save_data(
            "{}{}".format("" if success else "ERROR_", save_name),
            from_attrs=["nbins", "bin_def", "qb_transfer", "map_tags"],
            cls_shape=cls_shape,
            success=success,
            **opts
        )

        if not success:
            raise RuntimeError("Error computing transfer function: {}".format(msg))

        return self.qb_transfer

    def get_bandpowers(
        self,
        cls_shape,
        map_tag=None,
        converge_criteria=0.005,
        iter_max=200,
        return_qb=False,
        save_iters=False,
        delta_beta_prior=None,
        cond_noise=None,
        cond_criteria=None,
        null_first_cmb=False,
        tophat_bins=False,
        return_cls=False,
        like_profiles=False,
        like_profile_sigma=3.0,
        like_profile_points=100,
        file_tag=None,
        force_recompute=True,
    ):
        """
        Compute the maximum likelihood bandpowers of the data, assuming
        a given input spectrum shape.  Requires the transfer function to
        have been computed and loaded using `get_transfer`.

        Arguments
        ---------
        cls_shape : array_like
            Input shape spectrum.  Can differ from that used to compute the
            transfer function.
        map_tag : string
            If not None, then iteration is performed over the spectra
            corresponding to the given map, rather over all possible
            combinations of map-map cross-spectra. In this case, the first
            dimension of the input cbl must be of size 1 (this is done
            automatically by calling `bin_cl_template(..., map_tag=<map_tag>)`.
        converge_criteria : float
            Maximum fractional change in qb that indicates convergence and
            stops iteration.
        iter_max : int
            Maximum number of iterations to perform.  if this limit is
            reached, a warning is issued.
        return_qb : bool
            If True, only the maximum likelihood `qb` values are returned.
            Otherwise, the complete output dictionary is returned.
        save_iters : bool
            If True, the output data from each Fisher iteration are stored
            in an individual npz file.
        tophat_bins : bool
            If True, use uniform weights within each bin rather than any
            ell-based weighting.
        return_cls : bool
            If True, return C_ls rather than D_ls
        cond_criteria : float
            The maximum condition number allowed for Dmat1 to be acceptable
            for taking its inverse.
        like_profiles : bool
            If True, compute profile likelihoods for each qb, leaving all
            others fixed at their maximum likelihood values.  Profiles are
            computed over a range +/--sigma as estimated from the diagonals
            of the inverse Fisher matrix.
        like_profile_sigma : float
            Range in units of 1sigma over which to compute profile likelihoods
        like_profile_points : int
            Number of points to sample along the likelihood profile
        file_tag : string
            If supplied, appended to the bandpowers filename.
        force_recompute : bool
            If True, recompute bandpowers even if already on disk. Necessary
            if building fake data_xcorr in memory with input r.

        Returns
        -------
        data : dict
            Dictionary of maximum likelihood quantities, as output by
            `fisher_iterate`.
        -- or --
        qb, inv_fish : array_like
            Maximum likelihood bandpower amplitudes and fisher covariance.

        Data Handling
        ------------
        This method is called at the 'bandpowers' checkpoint, and loads or
        saves a data dictionary named 'bandpowers' with the quantities
        returned by `fisher_iterate`.
        """

        save_name = "bandpowers"

        fish_shape = (len(pt.dict_to_arr(self.bin_def)),) * 2

        # check all options that require rerunning fisher iterations
        opts = dict(
            converge_criteria=converge_criteria,
            delta_beta_prior=delta_beta_prior,
            cond_noise=cond_noise,
            null_first_cmb=null_first_cmb,
            apply_gcorr=self.apply_gcorr,
            weighted_bins=self.weighted_bins,
        )

        if self.template_cleaned:
            opts.update(
                template_alpha90=self.template_alpha90,
                template_alpha150=self.template_alpha150,
            )
        self.return_cls = return_cls

        ret = self.load_data(
            save_name,
            "bandpowers",
            bp_opts=True,
            to_attrs=False,
            shape=fish_shape,
            shape_ref="inv_fish",
            map_tag=map_tag,
            value_ref=opts,
            extra_tag=file_tag,
        )
        if force_recompute:
            ret = None
        if ret is not None:
            if return_qb:
                return ret["qb"], ret["inv_fish"]
            return ret

        cbl = self.bin_cl_template(cls_shape, map_tag, transfer_run=False)

        ret = self.fisher_iterate(
            cbl,
            cls_shape,
            map_tag,
            transfer_run=False,
            iter_max=iter_max,
            converge_criteria=converge_criteria,
            save_iters=save_iters,
            cond_noise=cond_noise,
            cond_criteria=cond_criteria,
            null_first_cmb=null_first_cmb,
            delta_beta_prior=delta_beta_prior,
            tophat_bins=tophat_bins,
            return_cls=return_cls,
            like_profiles=like_profiles,
            like_profile_sigma=like_profile_sigma,
            like_profile_points=like_profile_points,
            file_tag=file_tag,
        )

        if not ret["success"]:
            raise RuntimeError("Error computing bandpowers")

        # return
        if return_qb:
            return ret["qb"], ret["inv_fish"]
        return ret

    def get_likelihood(
        self,
        qb,
        inv_fish,
        map_tag=None,
        cls_shape=None,
        null_first_cmb=False,
        lmin=33,
        lmax=250,
        mcmc=True,
        r_prior=[0, np.inf],
        alpha90_prior=[0, np.inf],
        alpha150_prior=[0, np.inf],
        res_prior=None,
        beam_prior=[0, 1],
        betad_prior=[0, 1],
        dust_amp_prior=[0, np.inf],
        dust_ellind_prior=[0, 1],
        num_walkers=50,
        num_steps=20000,
        converge_criteria=0.01,
        reset_backend=None,
        file_tag=None,
    ):
        """
        Explore the likelihood, optionally with an MCMC sampler.

        Arguments
        ---------
        qb : OrderedDict
            Bandpower parameters previously computed by Fisher iteration.
        inv_fish : array_like
            Inverse Fisher matrix computed with the input qb's.
        map_tag : string
            If not None, then the likelihood is sampled using the spectra
            corresponding to the given map, rather over all possible
            combinations of map-map cross-spectra.  The input qb's and inv_fish
            must have been computed with the same option.
        cls_shape : array_like
            Input shape spectrum used to compute the qb's.  Required if r_prior
            is None.
        mcmc : bool
            If True, sample the likelihood using an MCMC sampler.  Remaining options
            determine parameter space and sampler configuration.
        r_prior : 2-list or None
            Prior upper and lower bound on tensor to scalar ratio.  If None, the
            fiducial shape spectrum is assumed, and the r parameter space is not
            varied.
        alpha90_prior, alpha150_prior : 2-list or None
            Prior upper and lower bound on template coefficients.  If None, the
            alpha parameter space is not varied.
        res_prior : 2-list or none
            Prior upper and lower bound on residual qbs.  If None, the
            res parameter space is not varied.
        beam_prior : 2-list or none
            Prior mean and width of gaussian width on beam error (when
            multiplied by beam error envelope).  If None, the
            beam parameter space is not varied.
        betad_prior : 2-list or none
            Prior mean and width of gaussian width on dust spectral index.
            If None, the dust index parameter space is not varied.
        dust_amp_prior : 2-list or none
            Prior upper and lower bound on dust amplitude.
            If None, the dust amp parameter space is not varied.
        dust_ellind_prior : 2-list or none
            Prior mean and width of Gaussian prior on difference in dust ell
            power law index. If None, don't vary from reference if fitting dust
            power spectrum model.
        num_walkers : int
            Number of unique walkers with which to sample the parameter space.
        num_steps : int
            Number of steps each walker should take in sampling the parameter space.
        reset_backend : bool
            If True, clear the backend buffer before sampling.  If False,
            samples are appended to the existing buffer.  If not supplied,
            set to True if the checkpoint has been forced to be rerun.
        file_tag : string
            If supplied, appended to the likelihood filename.
        """

        for x in [
            r_prior,
            alpha90_prior,
            alpha150_prior,
            res_prior,
            beam_prior,
            betad_prior,
            dust_amp_prior,
            dust_ellind_prior,
        ]:
            if x is not None:
                x[:] = [float(x[0]), float(x[1])]

        save_name = "like_mcmc"
        if not mcmc:
            alpha90_prior = None
            alpha150_prior = None
            res_prior = None
            beam_prior = None
            betad_prior = None
            dust_amp_prior = None
            dust_ellind_prior = None

        # no template cleaning if there aren't any templates specified
        if not getattr(self, "template_cleaned", False):
            alpha90_prior = None
            alpha150_prior = None
        else:
            # null out unused priors
            if self.template_alpha90 is None:
                alpha90_prior = None
            if self.template_alpha150 is None:
                alpha150_prior = None

        if not any([k.startswith("res_") for k in qb]):
            res_prior = None

        if np.any(
            [
                betad_prior is not None,
                dust_amp_prior is not None,
                dust_ellind_prior is not None,
            ]
        ):
            dust_ell_fit = True
        else:
            dust_ell_fit = False

        # bookkeeping: ordered priors
        priors = {
            "r_prior": r_prior,
            "alpha90_prior": alpha90_prior,
            "alpha150_prior": alpha150_prior,
            "res_prior": res_prior,
            "beam_prior": beam_prior,
            "betad_prior": betad_prior,
            "dust_amp_prior": dust_amp_prior,
            "dust_ellind_prior": dust_ellind_prior,
        }
        # priors on quantities that affect Dmat_obs or gmat (precalculated)
        obs_priors = [alpha90_prior, alpha150_prior]

        # check parameter space
        if all([x is None for x in priors.values()]):
            raise RuntimeError("Empty parameter space")

        out = dict(
            r_prior=r_prior,
            alpha90_prior=alpha90_prior,
            alpha150_prior=alpha150_prior,
            res_prior=res_prior,
            beam_prior=beam_prior,
            betad_prior=betad_prior,
            dust_amp_prior=dust_amp_prior,
            dust_ellind_prior=dust_ellind_prior,
            num_walkers=num_walkers,
            null_first_cmb=null_first_cmb,
            apply_gcorr=self.apply_gcorr,
            weighted_bins=self.weighted_bins,
            lmin=lmin,
            lmax=lmax,
        )


        if mcmc and reset_backend is None:
            ret = self.load_data(
                save_name,
                "likelihood",
                bp_opts=True,
                to_attrs=False,
                map_tag=map_tag,
                value_ref=out,
                extra_tag=file_tag,
            )
            if ret is not None and ret.get("converged", False):
                if converge_criteria >= ret.get("converge_criteria", 0.01):
                    return ret
            if ret is not None:
                for pname, pval in priors.items():
                    if np.all(pval != ret.get(pname, None)):
                        ret = None
            # clear chain cache if rerunning, otherwise append to chain by default
            reset_backend = ret is None

        out.update(converge_criteria=converge_criteria)

        # save state
        if mcmc and reset_backend:
            self.save_data(
                save_name, map_tag=map_tag, extra_tag=file_tag, bp_opts=True, **out
            )

        # clear pre-computed quantities
        self.clear_precalc()
        use_precalc = all([x is None for x in obs_priors])

        cls_input, cls_noise, cls_debias = self.get_data_spectra()

        # extract residual bins, ignoring bins outside of lmin/lmax
        if res_prior is not None:
            bin_def_orig = copy.deepcopy(self.bin_def)
            nbins_res_orig = self.nbins_res
            qb_res = OrderedDict()
            num_res = 0
            for k in list(qb):
                if k.startswith("res_"):
                    bd = self.bin_def[k]
                    good = np.where((bd[:, 1] > lmin) & (bd[:, 0] < lmax))[0]
                    # use all qb res in range lmin, lmax
                    self.bin_def[k] = bd[good]
                    v = qb.pop(k)[good]
                    num_res += len(v)

                    # use average qb res in good range per map
                    # self.bin_def[k] = np.array([[lmin, lmax + 1]])
                    # v = np.array([(qb.pop(k)[good]).mean()])
                    # num_res += 1
                    qb_res[k] = v
            self.nbins_res = num_res

        # set CMB model bandpowers to unity, since we are computing
        # the likelihood of this model given the data
        if r_prior is None:
            self.log("Computing model spectrum", "detail")
            warnings.warn("Beam variation not implemented for case " "of no r fit")
            cbl = self.bin_cl_template(cls_shape, map_tag)
            cls_model = self.get_model_spectra(qb, cbl, delta=True, cls_noise=cls_noise)
        else:
            qb = copy.deepcopy(qb)
            for spec in self.specs:
                stags = ["cmb_{}".format(spec), "fg_{}".format(spec)]
                for stag in stags:
                    if stag not in qb:
                        continue
                    qb[stag] = np.ones_like(qb[stag])

            self.log("Computing r model spectrum", "detail")
            cls_shape_scalar = self.get_signal_shape(
                r=1.0, save=False, component="scalar"
            )

            cls_shape_tensor = self.get_signal_shape(
                r=1.0, save=False, component="tensor"
            )

            # load tensor and scalar terms separately
            cbl_scalar = self.bin_cl_template(cls_shape_scalar, map_tag)
            cls_model_scalar = self.get_model_spectra(
                qb, cbl_scalar, delta=True, cls_noise=cls_noise
            )
            cbl_tensor = self.bin_cl_template(cls_shape_tensor, map_tag)
            cls_model_tensor = self.get_model_spectra(
                qb, cbl_tensor, delta=False, res=False
            )
            if beam_prior is not None:
                # load beam error term for tensor and scalar
                cbl_scalar_beam = self.bin_cl_template(
                    cls_shape_scalar, map_tag, beam_error=True
                )
                cls_mod_scal_beam = self.get_model_spectra(
                    qb, cbl_scalar_beam, delta=True, res=False
                )
                cbl_tensor_beam = self.bin_cl_template(
                    cls_shape_tensor, map_tag, beam_error=True
                )
                cls_mod_tens_beam = self.get_model_spectra(
                    qb, cbl_tensor_beam, delta=False, res=False
                )

            # load foreground shape
            if dust_ell_fit:
                cls_shape_dust = self.get_signal_shape(
                    foreground_fit=True, save=False, component="fg"
                )
                # if dust_ellind_prior is None:
                #    # can preload shape since not varying ell index
                cbl_fg = self.bin_cl_template(cls_shape_dust, map_tag=map_tag)
                if beam_prior is not None:
                    cbl_fg_beam = self.bin_cl_template(
                        cls_shape_dust, map_tag, beam_error=True
                    )

            cbl = copy.deepcopy(cbl_scalar)
            cls_model = copy.deepcopy(cls_model_scalar)

        # TODO
        # include priors on noise residuals from inv_fish
        # include beam uncertainties
        # how to marginalize over the garbage bin?

        def parse_params(theta):
            """
            Parse array of parameters into a dict
            """
            params = {}
            if r_prior is not None:
                params["r"] = theta[0]
                theta = theta[1:]
            if alpha90_prior is not None:
                params["alpha90"] = theta[0]
                theta = theta[1:]
            if alpha150_prior is not None:
                params["alpha150"] = theta[0]
                theta = theta[1:]
            if res_prior is not None:
                params["res"] = theta[:num_res]
                theta = theta[num_res:]
            if beam_prior is not None:
                if len(theta) == 1:
                    params["beam"] = theta[0]
                    theta = theta[1:]
                else:
                    # param for 90 and 150
                    params["beam"] = theta[:2]
                    theta = theta[2:]
            if betad_prior is not None:
                params["betad"] = theta[0]
                theta = theta[1:]
            if dust_amp_prior is not None:
                # param for ee and bb
                params["dust_amp"] = theta[:2]
                theta = theta[2:]
            if dust_ellind_prior is not None:
                params["dust_ellind"] = theta[0]
                theta = theta[1:]
            if len(theta):
                raise ValueError("Too many parameters to parse")
            return params

        def log_prior(
            r=None,
            alpha90=None,
            alpha150=None,
            res=None,
            beam=None,
            betad=None,
            dust_amp=None,
            dust_ellind=None,
        ):
            """
            Log prior function constructed from input options
            """
            values = {
                "r_prior": r,
                "alpha90_prior": alpha90,
                "alpha150_prior": alpha150,
                "res_prior": res,
                "dust_amp_prior": dust_amp,
            }
            for v, pval in values.items():
                prior = priors[v]
                if pval is not None and prior is not None:
                    if np.any(pval < prior[0]) or np.any(pval > prior[1]):
                        return -np.inf

            values_gauss = {
                "beam_prior": beam,
                "betad_prior": betad,
                "dust_ellind_prior": dust_ellind,
            }
            # for beam and betad, use gaussian prior
            log_prob_tot = 0.0
            for v, pval in values_gauss.items():
                prior = priors[v]
                if pval is not None and prior is not None:
                    pval = np.atleast_1d(pval)
                    norm = np.log(1.0 / (prior[1] * np.sqrt(2 * np.pi)))
                    chi = (pval[0] - prior[0]) / prior[1]
                    log_prob = norm - chi ** 2 / 2.0
                    if len(pval) > 1:  # 90 and 150
                        chi2 = (pval[1] - prior[0]) / prior[1]
                        log_prob += norm - chi2 ** 2 / 2.0
                    log_prob_tot += log_prob

            return log_prob_tot

        def log_like(
            r=None,
            alpha90=None,
            alpha150=None,
            res=None,
            beam=None,
            betad=None,
            dust_amp=None,
            dust_ellind=None,
        ):
            """
            Log likelihood function constructed from input options
            """
            cls_model0 = copy.deepcopy(cls_model)
            # compute new template subtracted data spectra
            if alpha90 is None and alpha150 is None:
                clsi = cls_input
            else:
                self.get_masked_xcorr(
                    template_alpha90=alpha90, template_alpha150=alpha150
                )
                clsi = self.get_data_spectra(do_noise=False)
            if beam is not None:
                beam = np.atleast_1d(beam)
                if len(beam) > 1:
                    beam = {"90": beam[0], "150": beam[1]}
                else:
                    beam = {"90": beam[0], "150": beam[0]}

            # compute new signal shape by scaling tensor component by r
            if r is not None:
                for stag, d in cls_model0.items():
                    comp, spec = stag.split("_", 1)
                    if spec not in ["ee", "bb"] or comp not in ["cmb", "total"]:
                        continue
                    ctag = "cmb_{}".format(spec)
                    for xname, dd in d.items():
                        if beam is not None:
                            m0, m1 = self.map_pairs[xname]
                            beam_coeffs = {
                                "b1": beam[self.nom_freqs[m0]],
                                "b2": beam[self.nom_freqs[m1]],
                                "b3": (
                                    beam[self.nom_freqs[m0]] * beam[self.nom_freqs[m1]]
                                ),
                            }
                            beam_term = 0
                            for bn, bc in beam_coeffs.items():
                                beam_term += bc * (
                                    cls_mod_scal_beam[ctag][xname][bn]
                                    + r * cls_mod_tens_beam[ctag][xname][bn]
                                )
                        else:
                            beam_term = 0

                        dd[:] = (
                            cls_model_scalar[stag][xname]
                            + r * cls_model_tensor[ctag][xname]
                            + beam_term
                        )
            elif beam is not None:
                for stag, d in cls_model0.items():
                    comp, spec = stag.split("_", 1)
                    if spec not in ["ee", "bb"] or comp not in ["cmb", "total"]:
                        continue
                    ctag = "cmb_{}".format(spec)
                    for xname, dd in d.items():
                        m0, m1 = self.map_pairs[xname]
                        beam_coeffs = {
                            "b1": beam[self.nom_freqs[m0]],
                            "b2": beam[self.nom_freqs[m1]],
                            "b3": (beam[self.nom_freqs[m0]] * beam[self.nom_freqs[m1]]),
                        }
                        beam_term = 0
                        for bn, bc in beam_coeffs.items():
                            beam_term += bc * cls_mod_scal_beam[ctag][xname][bn]
                        dd[:] = cls_model_scalar[stag][xname] + beam_term

            # fg term, including beam modifications. Because mix terms are
            # dependent on dust amp, get model specs here.
            if dust_ell_fit:
                if dust_amp is None:
                    qb["fg_ee"][:] = 1
                    qb["fg_bb"][:] = 1
                else:
                    qb["fg_ee"][:] = dust_amp[0]
                    qb["fg_bb"][:] = dust_amp[1]
                if betad is None:
                    qb["delta_beta"][:] = 0
                else:
                    qb["delta_beta"][:] = betad
                if dust_ellind is not None:
                    cbl_fg0 = self.bin_cl_template(
                        cls_shape_dust, map_tag=map_tag, fg_ell_ind=dust_ellind
                    )
                    if beam is not None:
                        cbl_fg_beam0 = self.bin_cl_template(
                            cls_shape_dust,
                            map_tag,
                            fg_ell_ind=dust_ellind,
                            beam_error=True,
                        )
                else:
                    cbl_fg0 = cbl_fg
                    if beam is not None:
                        cbl_fg_beam0 = cbl_fg_beam

                cls_model_fg = self.get_model_spectra(
                    qb, cbl_fg0, delta=True, res=False
                )
                if beam is not None:
                    cls_mod_fg_beam = self.get_model_spectra(
                        qb, cbl_fg_beam0, delta=True, res=False
                    )
                # add fg field to model, and add fg to total model
                for stag, d in cls_model_fg.items():
                    comp, spec = stag.split("_", 1)
                    if spec not in ["ee", "bb"] or comp not in ["fg", "total"]:
                        continue
                    ftag = "fg_{}".format(spec)
                    if stag not in cls_model0:
                        cls_model0[stag] = OrderedDict()
                    for xname, dd in d.items():
                        if xname not in cls_model0[stag]:
                            cls_model0[stag][xname] = cls_model_fg[ftag][xname]
                        else:
                            cls_model0[stag][xname] += cls_model_fg[ftag][xname]

                        # add beam terms to fg and total fields
                        if beam is not None:
                            m0, m1 = self.map_pairs[xname]
                            beam_coeffs = {
                                "b1": beam[self.nom_freqs[m0]],
                                "b2": beam[self.nom_freqs[m1]],
                                "b3": (
                                    beam[self.nom_freqs[m0]] * beam[self.nom_freqs[m1]]
                                ),
                            }
                            beam_term = 0
                            for bn, bc in beam_coeffs.items():
                                beam_term += bc * cls_mod_fg_beam[ftag][xname][bn]
                            cls_model0[stag][xname] += beam_term

            # compute noise model terms
            if res is None:
                clsm = cls_model0
            else:
                res = pt.arr_to_dict(res, qb_res)
                clsm = copy.deepcopy(cls_model0)
                cls_res = self.get_model_spectra(res, cbl)
                for stag, d in cls_res.items():
                    if stag not in clsm:
                        clsm[stag] = OrderedDict()
                    for xname, dd in d.items():
                        if xname not in clsm[stag]:
                            clsm[stag][xname] = dd
                        else:
                            clsm[stag][xname] += dd
            # compute likelihood
            like = self.fisher_calc(
                qb,
                cbl,
                clsi,
                cls_noise=cls_noise,
                cls_debias=cls_debias,
                cls_model=clsm,
                null_first_cmb=null_first_cmb,
                likelihood=True,
                use_precalc=use_precalc,
                like_lmin=lmin,
                like_lmax=lmax,
            )
            return like

        def log_prob(theta):
            """
            Log posterior probability from prior and likelihood

            Returns log_prior with each step
            """
            params = parse_params(theta)
            prior = log_prior(**params)
            if not np.isfinite(prior):
                return -np.inf, -np.inf
            like = log_like(**params)
            if not np.isfinite(like):
                return -np.inf, prior
            return prior + like, prior

        # initial values
        x0 = []
        brute_force = True if not mcmc else False  # only vary r
        if r_prior is not None:
            x0 += [0.01]
        if alpha90_prior is not None:
            if self.template_alpha90 == 0:
                x0 += [0.01]
            else:
                x0 += [self.template_alpha90]
            brute_force = False
        if alpha150_prior is not None:
            if self.template_alpha150 == 0:
                x0 += [0.01]
            else:
                x0 += [self.template_alpha150]
            brute_force = False
        if res_prior is not None:
            x0 += list(pt.dict_to_arr(qb_res, flatten=True))
            brute_force = False
        if beam_prior is not None:
            # add a beam term for each frequency
            x0 += [0.01] * len(np.unique(list(self.nom_freqs.values())))
            brute_force = False
        if betad_prior is not None:
            x0 += [0.01]
        if dust_amp_prior is not None:
            x0 += [1, 1]
        if dust_ellind_prior is not None:
            x0 += [0.01]

        ndim = len(x0)
        if ndim * 2 > num_walkers:
            num_walkers = int(np.round(ndim / float(num_walkers)) * num_walkers * 2)
            self.log(
                "Found {} parameters, increasing number of MCMC walkers to {}".format(
                    ndim, num_walkers
                )
            )
        x0 = np.array(x0)
        ndim = x0.size
        x0 = np.array(x0)[None, :] * (1 + 1e-4 * np.random.randn(num_walkers, len(x0)))

        if brute_force:
            self.log("Computing brute-force r profile likelihood", "task")
            likefile = self.get_filename(
                save_name, ext=".txt", map_tag=map_tag, extra_tag=file_tag, bp_opts=True
            )
            rs = np.linspace(0, 1, 200)
            likes = np.zeros_like(rs)
            res = None
            if res_prior is not None:
                res = pt.dict_to_arr(qb_res, flatten=True)
            for idx, r in enumerate(rs):
                like = log_like(r=r, res=res)
                if idx % 20 == 0:
                    self.log("r = {:.3f}, loglike = {:.2f}".format(r, like), "detail")
                likes[idx] = like
            header = "{} r likelihood\nColumns: r, loglike".format(
                "Multi-map" if map_tag is None else "Map {}".format(map_tag)
            )
            np.savetxt(likefile, np.column_stack((rs, likes)), header=header)

        if not mcmc:
            return

        # run chains!
        import emcee

        # setup sampler output file
        filename = self.get_filename(
            save_name, ext=".h5", map_tag=map_tag, extra_tag=file_tag, bp_opts=True
        )
        backend_exists = os.path.exists(filename)
        backend = emcee.backends.HDFBackend(filename)
        if backend_exists and backend.shape != (num_walkers, ndim):
            self.log(
                "Expected backend of shape ({}, {}), found {}. Resetting".format(
                    num_walkers, ndim, backend.shape
                )
            )
            reset_backend = True
        if reset_backend:
            backend.reset(num_walkers, ndim)

        # initialize sampler
        self.log("Initializing sampler", "task")
        sampler = emcee.EnsembleSampler(num_walkers, ndim, log_prob, backend=backend)
        if not reset_backend and backend_exists:
            # grab the last sample if appending to an existing run
            x0 = sampler.run_mcmc(None, 1)

        # track autocorrelation time
        old_tau = np.inf
        converged = False

        self.log(
            "Starting {} iterations with {} parameters".format(num_steps, ndim), "task"
        )
        for sample in sampler.sample(x0, iterations=num_steps):
            if not sampler.iteration % 10:
                self.log("MCMC iteration {}".format(sampler.iteration), "detail")
            # check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # compute autocorrelation time
            tau = sampler.get_autocorr_time(tol=0)

            # check convergence
            converged = np.all(tau / converge_criteria < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < converge_criteria)
            self.log(
                "MCMC iteration {} autocorr time: mean {:.1f} min {:.1f} max {:.1f}".format(
                    sampler.iteration, np.mean(tau), np.min(tau), np.max(tau)
                ),
                "task",
            )
            if converged:
                break
            old_tau = tau

        out.update(converged=converged, num_steps=sampler.iteration)

        # converged posterior distribution
        if converged:
            self.log(
                "MCMC converged in {} iterations".format(sampler.iteration), "task"
            )
            tau = sampler.get_autocorr_time()
            burnin = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))
            samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
            out.update(tau=tau, burnin=burnin, thin=thin, samples=samples)
        else:
            self.log("MCMC not converged in {} iterations".format(num_steps), "task")

        if res_prior is not None:
            self.bin_def = bin_def_orig
            self.nbins_res = nbins_res_orig

        # save and return
        return self.save_data(
            save_name, map_tag=map_tag, extra_tag=file_tag, bp_opts=True, **out
        )
