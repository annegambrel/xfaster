from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import glob
import sys
import subprocess as sp
import numpy as np
import time
import warnings
from .xfaster_class import XFaster
from . import base
from . import batch_tools as bt

__all__ = ["xfaster_run", "xfaster_parse", "xfaster_submit", "XFasterJobGroup"]


def githash():
    """
    Returns the git hash of xfaster
    """
    path = os.path.dirname(__file__)
    cwd = os.getcwd()
    os.chdir(path)
    ghash = sp.check_output(["git", "rev-parse", "HEAD"]).decode()
    os.chdir(cwd)
    return ghash.strip("\n")


def xfaster_run(
    config="config_example.ini",
    pol=True,
    pol_mask=True,
    bin_width=25,
    lmin=2,
    lmax=500,
    multi_map=True,
    likelihood=False,
    mcmc=True,
    output_root=None,
    output_tag=None,
    data_root="all",
    data_subset="full/*0",
    signal_subset="*",
    noise_subset="*",
    mask_type="hitsmask_tailored",
    clean_type="raw",
    noise_type="stationary",
    signal_type="r0p03",
    signal_transfer_type=None,
    signal_spec=None,
    signal_transfer_spec=None,
    model_r=None,
    data_root2=None,
    data_subset2=None,
    residual_fit=True,
    foreground_fit=False,
    bin_width_fg=30,
    res_specs=None,
    bin_width_res=25,
    weighted_bins=False,
    tt_marg=False,
    auto_marg_table=None,
    cross_marg_table=None,
    ensemble_mean=False,
    ensemble_median=False,
    sim_index=None,
    sims_add_alms=True,
    converge_criteria=0.005,
    iter_max=200,
    tbeb=False,
    fix_bb_xfer=False,
    window_lmax=None,
    like_lmin=33,
    like_lmax=250,
    like_profiles=False,
    like_profile_sigma=3.0,
    like_profile_points=100,
    pixwin=True,
    save_iters=False,
    verbose="task",
    debug=False,
    checkpoint=None,
    add_log=False,
    cond_noise=1e-5,
    cond_criteria=5e3,
    null_first_cmb=False,
    ref_freq=359.7,
    beta_ref=1.54,
    delta_beta_prior=0.5,
    noise_type_sim=None,
    signal_type_sim=None,
    foreground_type=None,
    template_type=None,
    template_alpha90=0.015,
    template_alpha150=0.043,
    sub_hm_noise=True,
    tophat_bins=False,
    return_cls=False,
    apply_gcorr=False,
    reload_gcorr=False,
    qb_file=None,
    alpha90_prior=[-np.inf, np.inf],
    alpha150_prior=[-np.inf, np.inf],
    r_prior=[-np.inf, np.inf],
    res_prior=None,
    beam_prior=[0, 1],
    betad_prior=None,
    dust_amp_prior=None,
    dust_ellind_prior=None,
    mcmc_walkers=50,
    like_converge_criteria=0.01,
    bp_tag=None,
    like_tag=None,
    sub_planck=False,
    fake_data_r=None,
    fake_data_template=None,
    do_fake_signal=True,
    do_fake_noise=True,
    save_fake_data=False,
    planck_reobs=True,
):
    """
    Main function for running the XFaster algorithm.

    Arguments
    ---------
    config : string
        Configuration file. If path doesn't exist, assumed
        to be in xfaster/config/<config>
    pol : bool
        If True, polarization spectra are computed.
    pol_mask : bool
        If True, a separate mask is applied for Q/U maps.
    bin_width : int or array_like of 6 ints
        Width of each ell bin for each of the six output spectra
        (TT, EE, BB, TE, EB, TB).  EE/BB bins should be the same
        in order to handle mixing correctly.
    lmin : int
        Minimum ell for which to compute spectra.
    lmax : int
        Maximum ell for which to compute spectra.
    weighted_bins : bool
        If True, use an lfac-weighted binning operator to construct Cbls.
        By default, a flat binning operator is used.
    multi_map : bool
        If True, compute all cross-spectra between maps
    likelihood : bool
        If True, compute the r likelihood
    mcmc : bool
        If True, sample the r likelihood using an MCMC sampler
    output_root : string
        Directory in which to store output files
    output_tag : string
        File tag for output files
    data_root : string
        Root directory where all input files are stored
    data_subset : string
        The subset of the data maps to include from each data split
        Must be a glob-parseable string
    signal_subset : string
        The subset of the signal sims to include
        Must be a glob-parseable string
    noise_subset : string
        The subset of the noise sims to include
        Must be a glob-parseable string
    clean_type : string
        If None, un-cleaned data are used.
        If set, the foreground residuals are estimated and fit by
        differencing with the un-cleaned data.
    noise_type : string
        The variant of noise sims to use
    noise_type_sim : string
        The variant of noise sims to use for sim_index fake data map.
        This enables having a different noise sim ensemble to use for
        sim_index run than the ensemble from which the noise is computed.
    mask_type : string
        The variant of mask to use
    signal_type : string
        The variant of signal sims to use
    signal_type_sim : string
        The variant of signal sims to use for sim_index face data map.
        This enables having a different noise sim ensemble to use for
        sim_index run than the ensemble from which the signal is computed.
    signal_transfer_type : string
        The variant of signal sims to use for transfer function
    signal_spec : string
        The spectrum data file to use for estimating bandpowers.  If not
        supplied, will search for `spec_signal_<signal_type>.dat` in the signal
        sim directory.
    signal_transfer_spec : string
        The spectrum data file used to generate signal sims.  If not
        supplied, will search for `spec_signal_<signal_type>.dat` in the
        transfer signal sim directory. Used for computing transfer functions.
    model_r : float
        The `r` value to use to compute a spectrum for estimating bandpowers.
        Overrides `signal_spec`.
    data_root2, data_subset2 : string
        If either of these is set, XFaster performs a null test between these
        two data halves.
    residual_fit : bool
        True is residual shape is being fit to the power. The residual shape
        can be split into a number of bins -> nbins_res. The residual bins can
        be marginalised over in the final output.
    foreground_fit : bool
        Include foreground residuals in the residual bins to be fit
    res_specs : list of strings
        Spectra to include in noise residual fitting.  List values can be any of
        the cross spectra TT, EE, BB, TE, EB, TB, or EEBB for fitting EE and BB
        residuals simultaneously.  If not supplied, this defaults to EEBB for
        polarized maps, or TT for unpolarized maps.
    bin_width_res : int
        Width of each bin to use for residual noise fitting
    tt_marg : bool
        Marginalize over all but one map's TT spectrum. Required for convergence
        if the masks are very similar between maps (?) -- This is obsolete,
        use auto_marg_table
    auto_marg_table : string
        JSON file specifying which auto spectra to marginalise for each map.
        If not None will override tt_marg
    cross_marg_table : string
        JSON file specifying which cross spectra to marginalise for each
        map tag.
    ensemble_mean : bool
        If True, substitute S+N ensemble means for Cls to test biasing
    ensemble_median : bool
        If True, substitute S+N ensemble median for Cls to test biasing
    sim_index : int
        If not None, substitute the sim_index S+N map for observed Cls
    sims_add_alms : bool
        If True and sim_index is not None, add sim alms instead of sim Cls
        to include signal and noise correlations
    converge_criteria : float
        The maximum fractional change in qb to signal convergence and
        end iteration
    iter_max : int
        The maximum number of iterations
    tbeb : bool
        If True, compute TB/EB spectra.
    null_bb_xfer : bool
        If True, after transfer functions have been calculated, impose
        the BB xfer is exactly equal to the EE transfer.
    window_lmax : int
        The size of the window used in computing the mask kernels
    like_lmin : int
        The minimum ell value to be included in the likelihood calculation
    like_lmax : int
        The maximum ell value to be included in the likelihood calculation
    like_profiles : bool
        If True, compute profile likelihoods for each qb, leaving all
        others fixed at their maximum likelihood values.  Profiles are
        computed over a range +/--sigma as estimated from the diagonals
        of the inverse Fisher matrix.
    like_profile_sigma : float
        Range in units of 1sigma over which to compute profile likelihoods
    like_profile_points : int
        Number of points to sample along the likelihood profile
    pixwin : bool
        If True, apply pixel window functions to beam windows.
    save_iters : bool
        If True, store the output of each Fisher iteration, in addition to
        the end result.
    verbose : bool or string
        Logging verbosity level.  If True, defaults to 'task'.
    debug : bool
        Store extra data in output files for debugging.
    checkpoint : string
        If supplied, re-compute all steps of the algorithm from this point
        forward.  Valid checkpoints are {checkpoints}
    add_log : bool
        If True, write log output to a file instead of to STDOUT.
        The log will be in `<output_root>/run_<output_tag>.log`.
        This option is useful for logging to file for jobs that
        are run directly (rather than submitted).
    cond_noise : float
        The level of regularizing noise to add to EE and BB diagonals.
    cond_criteria : float
        Threshold on covariance condition number. Above this, regularizing noise
        will be added to covariance to condition it.
    null_first_cmb : bool
        Keep first CMB bandpowers fixed to input shape (qb=1).
    foreground_fit: bool
        Fit for a dust component. Only for multi(freq)-map runs
    bin_width_fg : int or array_like of 6 ints
        Width of each ell bin for each of the six output foreground spectra
        (TT, EE, BB, TE, EB, TB).  EE/BB bins should be the same
        in order to handle mixing correctly.
    ref_freq : float
        In GHz, reference frequency for dust model. Dust bandpowers output
        will be at this reference frequency.
    beta_ref : float
        The spectral index of the dust model. This is a fixed value, with
        an additive deviation from this value fit for in foreground fitting
        mode.
    delta_beta_prior : float
        The width of the prior on the additive change from beta_ref. If you
        don't want the code to fit for a spectral index different
        from beta_ref, set this to be a very small value (O(1e-10)).
    foreground_type : string
        Tag for directory (foreground_<foreground_type>) where foreground
        sims are that should be added to the signal and noise sims
        when running in sim_index mode. Note: the same foreground sim
        map is used for each sim_index, despite signal and noise sims
        changing.
    template_type : string
        Tag for directory (templates_<template_type>) containing templates
        (e.g. a foreground model) to be scaled by a scalar value per
        map tag and subtracted from the data. The directory is assumed
        to contain halfmission-1 and halfmission-2 subdirectories, each
        containing one template per map tag.
    template_alpha90 : float
        Scalar to be applied to template map for subtraction from 90 GHz data.
    template_alpha150 : float
        Scalar to be applied to template map for subtraction from 150 GHz data.
    sub_hm_noise : bool
        If True, subtract average of Planck ffp10 noise crosses to debias
        template-cleaned spectra
    tophat_bins : bool
        If True, use uniform weights within each bin rather than any
        ell-based weighting.
    return_cls : bool
        If True, return C_l spectrum rather than the D_l spectrum
    apply_gcorr : bool
        If True, a correction factor is applied to the g (mode counting)
        matrix.  The correction factor should have been pre-computed
        for each map tag.
    reload_gcorr : bool
        If True, reload the gcorr file from the masks directory. Useful when
        iteratively solving for the correction terms.
    qb_file : string
        Pointer to a bandpowers.npz file in the output directory. If used
        in sim_index mode, the noise sim read from disk will be corrected
        by the residual qb values stored in qb_file.
    sub_planck : bool
        If True, subtract reobserved Planck from maps. Properly uses half
        missions so no Planck autos are used. Useful for removing expected
        signal residuals from null tests.
    fake_data_r : float
        If not None, construct a fake data map from scalar + r* tensor
        signal maps + noise + alpha*template, where signal and noise maps
        use the sim_index options, and alpha and template options are given
        by their respective options
    fake_data_template : str
        If not None, add halfmission-1 template in this directory scaled by
        alpha to fake data maps
    do_fake_signal : bool
        If true, use sim_index to set signal seed. If false, always use seed 0.
    do_fake_noise : bool
        If true, use sim_index to set noise seed. If false, always use seed 0.
    save_fake_data : bool
        If true, save data_xcorr file to disk for fake data.
    planck_reobs : bool
        If True, maps at Planck frequencies have been reobserved using 150a
        beam/filter.
    """

    cpu_start = time.clock()
    time_start = time.time()

    if noise_type == "None":
        noise_type = None

    if foreground_type is not None and sim_index is None:
        warnings.warn(
            "Ignoring argument foreground_type={} for non sim index run".format(
                foreground_type
            )
        )

    # initialize config file
    config_vars = base.XFasterConfig(locals(), "XFaster General")
    config_vars.update(dict(git_hash=githash(), config=config))

    common_opts = dict(
        lmax=lmax,
        pol=pol,
        pol_mask=pol_mask,
        output_root=output_root,
        output_tag=output_tag,
        verbose=verbose,
        debug=debug,
        checkpoint=checkpoint,
        add_log=add_log,
        ref_freq=ref_freq,
        beta_ref=beta_ref,
    )
    config_vars.update(common_opts, "XFaster Common")

    # initialize class
    X = XFaster(config, **common_opts)

    # setup options
    file_opts = dict(
        data_root=data_root,
        data_subset=data_subset,
        signal_subset=signal_subset,
        noise_subset=noise_subset,
        clean_type=clean_type,
        noise_type=noise_type,
        noise_type_sim=noise_type_sim,
        mask_type=mask_type,
        signal_type=signal_type,
        signal_type_sim=signal_type_sim,
        signal_transfer_type=signal_transfer_type,
        data_root2=data_root2,
        data_subset2=data_subset2,
        foreground_type=foreground_type,
        template_type=template_type,
        sub_planck=sub_planck,
        planck_reobs=planck_reobs,
    )
    config_vars.update(file_opts, "File Options")

    file_vars = X.get_files(**file_opts)
    config_vars.update(file_vars, "File Settings")

    beam_opts = dict(pixwin=pixwin)
    config_vars.update(beam_opts, "Beam Options")

    spec_opts = dict(
        converge_criteria=converge_criteria,
        iter_max=iter_max,
        save_iters=save_iters,
        lmin=lmin,
        bin_width=bin_width,
        tbeb=tbeb,
        fix_bb_xfer=fix_bb_xfer,
        window_lmax=window_lmax,
        ensemble_mean=ensemble_mean,
        ensemble_median=ensemble_median,
        sim_index=sim_index,
        sims_add_alms=sims_add_alms,
        signal_spec=signal_spec,
        signal_transfer_spec=signal_transfer_spec,
        model_r=model_r,
        foreground_fit=foreground_fit,
        bin_width_fg=bin_width_fg,
        residual_fit=residual_fit,
        res_specs=res_specs,
        bin_width_res=bin_width_res,
        weighted_bins=weighted_bins,
        delta_beta_prior=delta_beta_prior,
        cond_noise=cond_noise,
        cond_criteria=cond_criteria,
        null_first_cmb=null_first_cmb,
        tophat_bins=tophat_bins,
        return_cls=return_cls,
        apply_gcorr=apply_gcorr,
        reload_gcorr=reload_gcorr,
        like_profiles=like_profiles,
        like_profile_sigma=like_profile_sigma,
        like_profile_points=like_profile_points,
        qb_file=qb_file,
        template_alpha90=template_alpha90,
        template_alpha150=template_alpha150,
        sub_hm_noise=sub_hm_noise,
        file_tag=bp_tag,
    )
    config_vars.update(spec_opts, "Spectrum Estimation Options")
    config_vars.remove_option("XFaster General", "like_profile_sigma")
    config_vars.remove_option("XFaster General", "like_profile_points")
    config_vars.remove_option("XFaster General", "bp_tag")
    spec_opts.pop("ensemble_mean")
    spec_opts.pop("ensemble_median")
    spec_opts.pop("sim_index")
    spec_opts.pop("sims_add_alms")
    spec_opts.pop("window_lmax")
    spec_opts.pop("lmin")
    spec_opts.pop("bin_width")
    spec_opts.pop("foreground_fit")
    spec_opts.pop("bin_width_fg")
    spec_opts.pop("residual_fit")
    spec_opts.pop("res_specs")
    spec_opts.pop("bin_width_res")
    spec_opts.pop("weighted_bins")
    spec_opts.pop("tbeb")
    spec_opts.pop("fix_bb_xfer")
    spec_opts.pop("signal_spec")
    spec_opts.pop("signal_transfer_spec")
    spec_opts.pop("model_r")
    spec_opts.pop("qb_file")
    spec_opts.pop("template_alpha90")
    spec_opts.pop("template_alpha150")
    spec_opts.pop("sub_hm_noise")
    spec_opts.pop("apply_gcorr")
    spec_opts.pop("reload_gcorr")
    bandpwr_opts = spec_opts.copy()
    spec_opts.pop("file_tag")

    fisher_opts = spec_opts.copy()
    fisher_opts.pop("cond_noise")
    fisher_opts.pop("cond_criteria")
    fisher_opts.pop("delta_beta_prior")
    fisher_opts.pop("null_first_cmb")
    fisher_opts.pop("like_profiles")
    fisher_opts.pop("like_profile_sigma")
    fisher_opts.pop("like_profile_points")

    # disable residual fitting in single map mode
    if X.num_maps == 1 or not multi_map:
        residual_fit = False

    marg_opts = dict(
        tt_marg=tt_marg,
        auto_marg_table=auto_marg_table,
        cross_marg_table=cross_marg_table,
    )
    config_vars.update(marg_opts, "Marginalization Options")

    like_opts = dict(
        mcmc=mcmc,
        lmin=like_lmin,
        lmax=like_lmax,
        null_first_cmb=null_first_cmb,
        alpha90_prior=alpha90_prior,
        alpha150_prior=alpha150_prior,
        r_prior=r_prior,
        res_prior=res_prior,
        beam_prior=beam_prior,
        betad_prior=betad_prior,
        dust_amp_prior=dust_amp_prior,
        dust_ellind_prior=dust_ellind_prior,
        num_walkers=mcmc_walkers,
        converge_criteria=like_converge_criteria,
        file_tag=like_tag,
    )
    config_vars.update(like_opts, "Likelihood Estimation Options")
    config_vars.remove_option("XFaster General", "like_lmin")
    config_vars.remove_option("XFaster General", "like_lmax")
    config_vars.remove_option("XFaster General", "mcmc_walkers")
    config_vars.remove_option("XFaster General", "like_converge_criteria")
    config_vars.remove_option("XFaster General", "like_tag")

    # store config
    X.save_config(config_vars)

    X.log("Setting up bin definitions...", "task")
    X.get_bin_def(
        bin_width=bin_width,
        lmin=lmin,
        tbeb=tbeb,
        foreground_fit=False,
        weighted_bins=weighted_bins,
    )

    X.log("Computing mask cross-spectra and weights...", "task")
    X.get_mask_weights(apply_gcorr=apply_gcorr, reload_gcorr=reload_gcorr)

    X.log("Computing kernels...", "task")
    X.get_kernels(window_lmax=window_lmax)

    X.log("Computing sim ensemble averages for transfer function...", "task")
    if signal_transfer_type in [signal_type, None]:
        # Do all the sims at once to also get the S+N sim ensemble average
        do_noise = True
    else:
        do_noise = False
    # for transfer
    X.get_masked_sims(transfer=True, do_noise=do_noise)

    X.log("Computing beam window functions...", "task")
    X.get_beams(**beam_opts)

    X.log("Loading spectrum shape for transfer function...", "task")
    cls_shape = X.get_signal_shape(
        filename=signal_transfer_spec, tbeb=False, transfer=True
    )

    X.log("Computing transfer functions...", "task")
    X.get_transfer(cls_shape, fix_bb_xfer=fix_bb_xfer, **fisher_opts)

    # Rerun to add bins for foreground and residuals, if requested
    X.log("Setting up bin definitions with foregrounds and residuals...", "task")
    X.get_bin_def(
        bin_width=bin_width,
        lmin=lmin,
        tbeb=tbeb,
        foreground_fit=foreground_fit,
        bin_width_fg=bin_width_fg,
        residual_fit=residual_fit,
        res_specs=res_specs,
        bin_width_res=bin_width_res,
        weighted_bins=weighted_bins,
    )

    if template_type is not None:
        X.log("Computing template noise ensemble averages...", "task")
        X.get_masked_template_noise(template_type)

    X.log("Computing masked data cross-spectra...", "task")
    X.get_masked_xcorr(
        template_alpha90=template_alpha90,
        template_alpha150=template_alpha150,
        sub_planck=sub_planck,
        sub_hm_noise=sub_hm_noise,
    )

    X.log("Computing sim ensemble averages...", "task")
    if fake_data_r is not None:
        sim_index_sim = None
    else:
        sim_index_sim = sim_index
    X.get_masked_sims(
        ensemble_mean=ensemble_mean,
        ensemble_median=ensemble_median,
        sim_index=sim_index_sim,
        sims_add_alms=sims_add_alms,
        qb_file=qb_file,
    )
    if fake_data_r is not None:
        X.log("Replacing data with fake data...", "task")
        X.get_masked_fake_data(
            fake_data_r=fake_data_r,
            fake_data_template=fake_data_template,
            sim_index=sim_index,
            template_alpha90=template_alpha90,
            template_alpha150=template_alpha150,
            noise_type=noise_type,
            do_signal=do_fake_signal,
            do_noise=do_fake_noise,
            save_data=save_fake_data,
            sub_hm_noise=sub_hm_noise,
        )

    X.log("Computing spectra...", "task")

    if X.null_run:
        X.log("Loading flat spectrum for null test...", "task")
        cls_shape = X.get_signal_shape(flat=True, tbeb=tbeb)
    else:
        X.log("Loading spectrum shape for bandpowers...", "task")
        cls_shape = X.get_signal_shape(
            filename=signal_spec, r=model_r, foreground_fit=foreground_fit, tbeb=tbeb
        )

    X.log("Constructing marginalization table...", "task")
    X.get_marg_table(**marg_opts)

    if multi_map:
        X.log("Computing multi-map bandpowers...", "task")
        qb, inv_fish = X.get_bandpowers(
            cls_shape,
            return_qb=True,
            force_recompute=fake_data_r is not None,
            **bandpwr_opts
        )

        if likelihood:
            X.log("Computing multi-map likelihood...", "task")
            X.get_likelihood(qb, inv_fish, cls_shape=cls_shape, **like_opts)

    else:
        for map_tag, map_file in zip(X.map_tags, X.map_files):
            X.log("Processing map {}: {}".format(map_tag, map_file), "task")

            X.log("Computing bandpowers for map {}".format(map_tag), "part")
            qb, inv_fish = X.get_bandpowers(
                cls_shape,
                map_tag=map_tag,
                return_qb=True,
                force_recompute=fake_data_r is not None,
                **bandpwr_opts
            )

            if likelihood:
                X.log("Computing likelihoods for map {}".format(map_tag), "part")
                X.get_likelihood(
                    qb, inv_fish, map_tag=map_tag, cls_shape=cls_shape, **like_opts
                )

    cpu_elapsed = time.clock() - cpu_start
    time_elapsed = time.time() - time_start
    X.log("Wall time: {:.2f} s, CPU time: {:.2f} s".format(time_elapsed, cpu_elapsed))


xfaster_run.__doc__ = xfaster_run.__doc__.format(checkpoints=XFaster.checkpoints)


def xfaster_parse(args=None, test=False):
    """
    Return a parsed dictionary of arguments for the xfaster execution script.

    Arguments
    ---------
    args : list of strings, optional
        If not supplied, read from the command line (sys.argv) by argparse.
    test : bool, optional
        If True, raise a RuntimeError instead of exiting.  Useful for
        interactive testing.

    Returns
    -------
    args : dict
        Dictionary of parsed options
    """

    import argparse as ap
    import sys

    parser_opts = dict(
        description="Run the XFaster algorithm",
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )

    # initialize parser
    if test:

        class TestParser(ap.ArgumentParser):
            def __init__(self, *args, **kwargs):
                super(TestParser, self).__init__(*args, **kwargs)

            def error(self, msg):
                self.print_usage(sys.stderr)
                raise RuntimeError(msg)

            def exit(self, status=0, msg=None):
                msg = "exiting with status {}{}".format(
                    status, ": {}".format(msg) if msg else ""
                )
                raise RuntimeError(msg)

        P = TestParser(**parser_opts)
    else:
        P = ap.ArgumentParser(**parser_opts)

    # get default argument values from xfaster_run
    defaults = base.get_func_defaults(xfaster_run)
    defaults.pop("add_log", None)
    rem_args = list(defaults)

    # helper function for populating command line arguments
    def add_arg(P, name, argtype=None, default=None, short=None, help=None, **kwargs):

        name = name.replace("-", "_")
        argname = "--{}".format(name.replace("_", "-"))
        altname = kwargs.pop("altname", None)

        if default is None:
            default = defaults[name]
        if name in rem_args:
            rem_args.remove(name)

        if help is None:
            raise ValueError("Missing help text for argument {}".format(name))

        opts = dict(default=default, help=help, dest=name, action="store")
        opts.update(**kwargs)

        if default is True:
            argname = "--no-{}".format(name.replace("_", "-"))
            opts["action"] = "store_false"
        elif default is False:
            opts["action"] = "store_true"
        else:
            if argtype is None:
                if isinstance(default, (int, float)):
                    argtype = type(default)
            opts["type"] = argtype

        argnames = (argname,)
        if short is not None:
            if not short.startswith("-"):
                short = "-{}".format(short)
            argnames += (short,)
        if altname is not None:
            argnames += ("--{}".format(altname.replace("_", "-")),)

        P.add_argument(*argnames, **opts)

    # subparsers
    S = P.add_subparsers(
        dest="mode",
        metavar="MODE",
        title="subcommands",
        help="Function to perform. For more help, call: " "%(prog)s %(metavar)s -h",
    )
    parser_opts.pop("description")

    # populate subparsers
    for mode, helptext in [("run", "run xfaster"), ("submit", "submit xfaster job")]:

        PP = S.add_parser(mode, help=helptext, **parser_opts)

        # xfaster_run arguments
        G = PP.add_argument_group("run arguments")
        add_arg(G, "config", help="Config file")
        add_arg(G, "pol", help="Ignore polarization")
        add_arg(G, "pol_mask", help="Apply a separate mask for Q/U maps")
        add_arg(G, "bin_width", help="Bin width for each spectrum")
        add_arg(G, "lmin", help="Minimum ell for spectrum estimation")
        add_arg(G, "lmax", help="Maximum ell for spectrum estimation")
        add_arg(G, "multi_map", help="Run the analysis on single maps")
        add_arg(G, "likelihood", help="Compute likelihood")
        add_arg(G, "mcmc", argtype=bool, help="Sample likelihood with MCMC sampler")
        add_arg(G, "mcmc_walkers", help="Number of MCMC walkers to use")
        add_arg(G, "like_profiles", help="Compute bandpower profile likelihoods")
        add_arg(G, "r_prior", nargs="+", help="Prior on r [lower, upper] limits")
        add_arg(
            G, "alpha90_prior", nargs="+", help="Prior on alpha90 [lower, upper] limits"
        )
        add_arg(
            G,
            "alpha150_prior",
            nargs="+",
            help="Prior on alpha150 [lower, upper] limits",
        )
        add_arg(
            G, "res_prior", nargs="+", help="Prior on res qbs [lower, upper] limits"
        )
        add_arg(
            G,
            "beam_prior",
            nargs="+",
            help="Prior on beam (should be [0, 1] for "
            "[mean 0, width 1 sig] gaussian)",
        )
        add_arg(
            G,
            "betad_prior",
            nargs="+",
            help="Prior on dust index different from ref beta "
            "(should be [0, sig] for [mean 0, width sig] gaussian)",
        )
        add_arg(
            G,
            "dust_amp_prior",
            nargs="+",
            help="Prior on dust amplitude (rel to Planck 353 ref) "
            "[lower, upper] limits",
        )
        add_arg(
            G,
            "dust_ellind_prior",
            nargs="+",
            help="Prior on dust ell index different from ref "
            "(should be [0, sig] for [mean 0, width sig] gaussian)",
        )
        add_arg(
            G,
            "output_root",
            default=os.getcwd(),
            help="Working directory for storing output files",
        )
        add_arg(G, "output_tag", help="File tag for storing output files")
        add_arg(
            G,
            "data_root",
            required=True,
            help="Root directory containing the input data structure.",
        )
        add_arg(
            G,
            "data_subset",
            help="Glob_parseable map tag to include, "
            "include multiple tags as comma delimited sequence enclosed in "
            "double quotes",
        )
        add_arg(
            G,
            "signal_subset",
            help="Glob_parseable map tag to include " "for signal sims",
        )
        add_arg(
            G,
            "noise_subset",
            help="Glob_parseable map tag to include " "for noise sims",
        )
        add_arg(G, "clean_type", help="Variant of foreground-cleaned data maps to use")
        add_arg(G, "noise_type", help="Noise sim variant")
        add_arg(G, "noise_type_sim", help="Noise sim variant to use for sim_index")
        add_arg(G, "mask_type", help="Mask variant")
        add_arg(G, "signal_type", help="Signal sim variant")
        add_arg(G, "signal_type_sim", help="Signal sim variant to use for sim_index")
        add_arg(
            G, "signal_transfer_type", help="Signal sim variant for transfer functions"
        )
        add_arg(
            G,
            "signal_transfer_spec",
            help="Power spectrum used to create transfer signal simulations. "
            "Defaults to the spec_signal_{signal_transfer_type}.dat file found "
            "in the signal directory.",
        )
        E = G.add_mutually_exclusive_group()
        add_arg(
            E,
            "signal_spec",
            help="Power spectrum shape used for estimating bandpowers. "
            "Defaults to signal_spec if not supplied",
        )
        add_arg(
            E,
            "model_r",
            argtype=float,
            help="r value to use to compute a shape spectrum for "
            "estimating bandpowers",
        )
        add_arg(
            G,
            "data_root2",
            help="Root directory containing the input data structure " "for null tests",
        )
        add_arg(
            G,
            "data_subset2",
            help="Glob-parseable map tag for null test maps to include",
        )
        add_arg(
            G,
            "residual_fit",
            help="Fit residual shapes to observed power."
            " Residual shape can be split into a number of bins"
            "Use --residual-marg to marginalize over residual bins.",
        )
        add_arg(
            G,
            "foreground_fit",
            help="Include foreground residuals in the residual bins to be " "fit",
        )
        add_arg(
            G,
            "res_specs",
            nargs="+",
            help="Spectra to include in noise residual fitting.  Use 'EEBB' for "
            "fitting EE and BB residuals simultaneously.",
            choices=["TT", "EE", "BB", "TE", "EB", "TB", "EEBB"],
        )
        add_arg(
            G,
            "bin_width_res",
            help="Width of each bin to use for residual noise fitting",
            altname="noise_dl",
        )
        add_arg(
            G,
            "weighted_bins",
            help="Use lfac-weighted binning operator to construct Cbls",
        )
        add_arg(G, "tt_marg", help="Marginalize over TT spectra for any input maps")
        add_arg(G, "auto_marg_table", help="Table specifying what autos to marginalise")
        add_arg(
            G, "cross_marg_table", help="Table specifying what crosses to marginalise"
        )
        E = G.add_mutually_exclusive_group()
        add_arg(
            E,
            "ensemble_mean",
            help="Substitute S+N ensemble means for Cls to test biasing",
        )
        add_arg(
            E,
            "ensemble_median",
            help="Substitute S+N ensemble medians for Cls to test biasing",
        )
        add_arg(
            G,
            "sim_index",
            argtype=int,
            help="Substitute sim_index map S+N for observed Cls",
        )
        add_arg(
            G,
            "qb_file",
            argtype=str,
            help="File from which to get residual qbs to modify noise Cls",
        )
        add_arg(
            G,
            "sims_add_alms",
            argtype=bool,
            help="If True and sim_index is not None, add sim alms instead "
            "of sim Cls to include signal and noise correlations ",
        )
        add_arg(
            G,
            "converge_criteria",
            help="Criterion for convergence of the Fisher estimator",
        )
        add_arg(
            G,
            "iter_max",
            help="Maximum number of iterations to compute the Fisher " "matrix",
        )
        add_arg(G, "tbeb", help="Include TB/EB spectra in the estimator")
        add_arg(G, "fix_bb_xfer", help="Fix BB xfer to equal EE xfer")
        add_arg(G, "window_lmax", argtype=int, help="Kernel window size")
        add_arg(
            G,
            "like_lmin",
            argtype=int,
            help="Minimum ell to include in the likelihood calculation",
        )
        add_arg(
            G,
            "like_lmax",
            argtype=int,
            help="Maximum ell to include in the likelihood calculation",
        )
        add_arg(
            G,
            "like_profile_sigma",
            argtype=float,
            help="Range in units of 1sigma over which to compute profile likelihoods",
        )
        add_arg(
            G,
            "like_profile_points",
            argtype=int,
            help="Number of points to sample along each profile likelihood",
        )
        add_arg(G, "pixwin", help="Do not apply the pixel window function")
        add_arg(
            G,
            "save_iters",
            help="Save data for each fisher iteration.  " "Useful for debugging",
        )
        add_arg(
            G,
            "verbose",
            short="-v",
            choices=["user", "info", "task", "part", "detail", "all"],
            help="Verbosity level",
        )
        add_arg(G, "debug", help="Store extra data in output files for debugging")
        add_arg(
            G,
            "checkpoint",
            short="-c",
            choices=XFaster.checkpoints,
            help="Checkpoint for recomputing all following stages, "
            "rather than loading from disk.",
        )
        add_arg(
            G,
            "cond_noise",
            argtype=float,
            help="The level of regularizing noise to add.",
        )
        add_arg(
            G,
            "cond_criteria",
            argtype=float,
            help="Threshold on covariance condition number.",
        )
        add_arg(
            G,
            "null_first_cmb",
            argtype=bool,
            help="Keep first CMB bandpowers fixed to input shape (qb=1).",
        )
        add_arg(G, "bin_width_fg", help="Bin width for dust spectra")
        add_arg(G, "ref_freq", help="In GHz, reference frequency for dust model")
        add_arg(
            G,
            "beta_ref",
            argtype=float,
            help="The dust spectral index. The parameter fit for is an "
            "additive constant away from this value.",
        )
        add_arg(
            G,
            "delta_beta_prior",
            argtype=float,
            help="The width of the prior on the additive deviation from " "beta_ref",
        )
        add_arg(
            G, "foreground_type", help="Foreground sim variant to use for sim_index"
        )
        add_arg(
            G, "template_type", help="Template type to use for template subtraction"
        )
        add_arg(
            G,
            "template_alpha90",
            argtype=float,
            help="Scaling to use for 90 GHz template subtraction",
        )
        add_arg(
            G,
            "template_alpha150",
            argtype=float,
            help="Scaling to use for 150 GHz template subtraction",
        )
        add_arg(
            G,
            "sub_hm_noise",
            help="Subtract hm1xhm2 Planck noise sim average from template-cleaned"
            " spectra.",
        )
        add_arg(
            G,
            "like_converge_criteria",
            argtype=float,
            help="Convergence criteria for likelihood MCMC chains",
        )
        add_arg(G, "tophat_bins", help="Use uniform weights in each bin.")
        add_arg(G, "return_cls", help="Return C_ls rather than the default D_ls.")
        add_arg(
            G,
            "apply_gcorr",
            help="Apply an empirically-computed correction factor to the g matrix.",
        )
        add_arg(
            G,
            "reload_gcorr",
            help="Reload correction factor from file in masks directory.",
        )
        add_arg(G, "bp_tag", help="Append tag to bandpowers output file")
        add_arg(G, "like_tag", help="Append tag to likelihood output files")
        add_arg(
            G, "sub_planck", argtype=bool, help="Subtract Planck maps from data maps"
        )
        add_arg(
            G,
            "fake_data_r",
            argtype=float,
            help="Construct fake data with tensor map scaled by this r",
        )
        add_arg(
            G,
            "fake_data_template",
            help="Add halfmission-1 map from this directory scaled by alpha"
            " to fake data map",
        )
        add_arg(
            G, "do_fake_signal", argtype=bool, help="Vary sim index for fake signal"
        )
        add_arg(G, "do_fake_noise", argtype=bool, help="Vary sim index for fake noise")
        add_arg(
            G, "save_fake_data", argtype=bool, help="Save fake data data_xcorr file"
        )
        add_arg(
            G, "planck_reobs", argtype=bool, help="Input Planck maps are 150a reobs?"
        )

        # submit args
        if mode == "submit":
            G = PP.add_argument_group("submit arguments")
            G.add_argument(
                "--job-prefix",
                action="store",
                help="Name to prefix to all submitted jobs",
            )
            G.add_argument(
                "-q", "--queue", action="store", default=None, help="Queue to submit to"
            )
            G.add_argument(
                "--nodes",
                action="store",
                type=str,
                default="1",
                help="Number of nodes to use. Or node name",
            )
            G.add_argument(
                "--ppn",
                action="store",
                type=int,
                default=8,
                help="Number of processors per node",
            )
            G.add_argument(
                "--mem",
                action="store",
                type=float,
                default=5,
                help="Memory per process, in GB",
            )
            E = G.add_mutually_exclusive_group()
            E.add_argument(
                "--cput",
                action="store",
                default=None,
                type=float,
                help="cput per process in hours",
            )
            E.add_argument(
                "--wallt",
                action="store",
                default=None,
                type=float,
                help="walltime in hours",
            )
            G.add_argument(
                "--nice",
                action="store",
                type=int,
                default=0,
                help="Scheduling priority from -5000 (high) to 5000",
            )
            G.add_argument(
                "--omp-threads",
                action="store",
                type=int,
                default=None,
                help="Number of OMP threads to use",
            )
            G.add_argument(
                "--slurm",
                action="store_true",
                default=False,
                help="Submit a slurm script rather than PBS",
            )
            G.add_argument(
                "--env-script", help="Script to source in jobs to set up " "environment"
            )
            G.add_argument("--exclude", help="Nodes to exclude")
            G.add_argument(
                "--dep-afterok",
                action="store",
                nargs="+",
                default=None,
                help="List of job IDs to wait on completion of " "before running job",
            )

        # other arguments
        PP.add_argument(
            "--test",
            action="store_true",
            default=False,
            help="Print options for debugging",
        )

    # check that all xfaster_run arguments have been handled by the parser
    if len(rem_args):
        warnings.warn("Arguments {} is not handled by the parser".format(rem_args))
    # parse arguments
    args = P.parse_args(args=args)
    # fix arguments meant to be empty
    for k, v in vars(args).items():
        if v is not None and str(v).lower() in ["none", "['none']"]:
            setattr(args, k, None)

    # test mode
    if args.mode != "submit":
        if args.test:
            msg = ",\n".join(
                "{}={!r}".format(k, v) for k, v in sorted(vars(args).items())
            )
            P.exit(0, "{}\nargument test\n".format(msg))
        delattr(args, "test")

    # return a dictionary
    return vars(args)


class XFasterJobGroup(object):
    def __init__(self):
        """
        Class for parsing xfaster options into a job script, and optionally
        grouping the jobs together.
        """
        self.reset()

    def reset(self):
        """
        Initialize to a reset state.
        """
        self.script_path = None
        self.output = None
        self.job_list = []
        self.qsub_args = {}

    def add_job(self, **kwargs):
        """
        Add xfaster job to script.

        Keyword arguments
        -----------------
        Most should correspond to arguments accepted by `xfaster_run`.
        If job-related arguments are present, they will be passed to
        `set_job_options`.
        """

        # set job options
        job_opts = base.extract_func_kwargs(
            self.set_job_options, kwargs, pop=True, others_ok=True
        )

        output_root = kwargs.get("output_root")
        output_tag = kwargs.get("output_tag")
        if output_root is not None:
            if output_tag is not None:
                output_root = os.path.join(output_root, output_tag)
            job_opts["output"] = output_root

        if job_opts:
            job_prefix = job_opts.get("job_prefix")
            if job_prefix is None:
                job_prefix = "xfaster"
                if output_tag is not None:
                    job_prefix = "_".join([job_prefix, output_tag])
                job_opts["job_prefix"] = job_prefix
            self.set_job_options(**job_opts)

        # construct command
        cmd = "python {script} run".split()

        # handle deprecated arguments
        if "noise_dl" in kwargs:
            warnings.warn("Argument noise_dl is deprecated, use bin_width_res instead")
            kwargs["bin_width_res"] = kwargs.pop("noise_dl")
        if "model_spec" in kwargs:
            warnings.warn("Argument model_spec is deprecated, use signal_spec instead")
            kwargs["signal_transfer_spec"] = kwargs.pop("signal_spec")
            kwargs["signal_spec"] = kwargs.pop("model_spec")
        if "residual_marg" in kwargs:
            warnings.warn("Argument residual_marg is deprecated")
            kwargs.pop("residual_marg")

        # figure out variable types from default values
        defaults = base.get_func_defaults(xfaster_run)
        for a in defaults:
            if a not in kwargs:
                continue

            v = kwargs.pop(a)
            s = a.replace("_", "-")

            da = defaults[a]
            if da is True and v is False:
                cmd += ["--no-{}".format(s)]
            elif da is False and v is True:
                cmd += ["--{}".format(s)]
            elif v != da:
                if a in [
                    "res_specs",
                    "like_mask",
                    "r_prior",
                    "alpha90_prior",
                    "alpha150_prior",
                    "beam_prior",
                    "res_prior",
                    "betad_prior",
                    "dust_amp_prior",
                    "dust_ellind_prior",
                ]:
                    if np.isscalar(v):
                        v = [v]
                    if "prior" in a:
                        # special case to allow -inf to work
                        cmd += ["--{}".format(a.replace("_", "-"))]
                        if v is None:
                            cmd += ["None"]
                        else:
                            cmd += ["' {}'".format(s) for s in v]

                    else:
                        cmd += ["--{}".format(a.replace("_", "-"))]
                        cmd += [str(s) for s in v]

                elif a in [
                    "noise_subset",
                    "signal_subset",
                    "data_subset",
                    "data_subset2",
                ]:
                    # add quotes around glob parseable args to avoid weird
                    # behavior
                    cmd += ["--{}".format(s)]
                    cmd += ["'{}'".format(v)]
                else:
                    cmd += "--{} {}".format(s, v).split()

        if len(kwargs):
            # Args that are not being parsed-- raise an error
            raise KeyError("{} arguments not recognized.".format(kwargs))

        self.job_list.append(" ".join(cmd))

    def set_job_options(
        self,
        output=None,
        cput=None,
        wallt=None,
        ppn=8,
        nodes=1,
        mem=5,
        env_script=None,
        omp_threads=None,
        nice=0,
        queue=None,
        job_prefix=None,
        test=False,
        script_path=None,
        pbs=False,
        workdir=None,
        dep_afterok=None,
        exclude=None,
    ):
        """
        Parse options that control the job script, rather than xfaster.
        """

        # XFaster runs faster with OMP and does not use MPI
        mpi = False
        mpi_procs = None
        if omp_threads is None:
            omp_threads = ppn

        # default job prefix
        if job_prefix is None:
            job_prefix = "xfaster"

        # different default script path
        if script_path is None and self.script_path is None:
            script_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "scripts", "xfaster")
            )
            self.script_path = script_path

        # create output directories
        if output is None:
            output = self.output
        output = os.path.abspath(output)
        if workdir is None:
            workdir = os.path.join(output, "logs")

        if cput is not None:
            try:
                # split at ":" since extra resource modifiers can go after
                cput *= ppn * int(str(nodes).split(":")[0])
            except ValueError:
                # could not convert nodes to int, assume one node (name)
                cput *= ppn
        if mem is not None:
            mem *= ppn

        if not mpi:
            mpi_procs = None
        elif mpi_procs is None and ppn is not None:
            mpi_procs = ppn

        if pbs:
            scheduler = "pbs"
        else:
            scheduler = "slurm"

        self.batch_args = dict(
            workdir=workdir,
            mem=mem,
            nodes=nodes,
            ppn=ppn,
            cput=cput,
            wallt=wallt,
            queue=queue,
            env_script=env_script,
            omp_threads=omp_threads,
            mpi_procs=mpi_procs,
            nice=nice,
            delete=False,
            submit=not test,
            debug=test,
            scheduler=scheduler,
            name=job_prefix,
            dep_afterok=dep_afterok,
            exclude=exclude,
        )

    def submit(self, group_by=None, verbose=True, **kwargs):
        """
        Submit jobs that have been added.

        Arguments
        ---------
        group_by : int, optional
            Group xfaster calls into jobs with this many calls each.
        """
        if not self.job_list:
            raise RuntimeError("No unimap jobs have been added.")
        if self.script_path is None:
            self.script_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "scripts", "xfaster")
            )
        for idx in range(len(self.job_list)):
            self.job_list[idx] = self.job_list[idx].format(script=self.script_path)
        if group_by is None:
            group_by = len(self.job_list)
        if kwargs:
            self.set_job_options(**kwargs)
        if not self.batch_args:
            raise RuntimeError("No job options specified")
        job_ids = bt.batch_group(
            self.job_list,
            group_by=group_by,
            serial=True,
            verbose=verbose,
            **self.batch_args
        )
        self.reset()
        return job_ids


def xfaster_submit(script_path=None, **kwargs):
    """
    Submit a single xfaster job. The arguments here should agree exactly
    with the command line flags for submit mode, with kwargs passed to
    `xfaster_run`. Run `xfaster --help` for help.
    """
    xg = XFasterJobGroup()
    xg.add_job(script_path=script_path, **kwargs)
    return xg.submit(group_by=1, verbose=True)
