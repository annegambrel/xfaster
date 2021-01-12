from __future__ import absolute_import
import numpy as np
import glob
import copy
import os as os
from .base import XFasterConfig
from . import parse_tools as pt

__all__ = ["XFasterData"]


class XFasterData(object):
    """
    Class for interpreting the output of XFaster
    """

    def __init__(self, data_root=None):
        """
        Initialize data object

        Arguments
        ---------
        data_root : str
            Path to XFaster output
        """
        self.path = data_root
        self.config = {}
        self.bandpowers = {}
        self.transfer = {}
        self.beam = {}
        self.ensemble_cov = {}

    def __del__(self):
        """
        Clear cached data
        """
        self.config.clear()
        self.bandpowers.clear()
        self.transfer.clear()
        self.beam.clear()
        self.ensemble_cov.clear()

    def load_config(self, tag):
        """
        Load the run config file

        Arguments
        ---------
        tag : string
            XFaster output tag

        Returns
        -------
        cfg : XFasterConfig instance
            Run configuration for the given output tag
        """
        if tag not in self.config:
            file_path = os.path.join(self.path, tag)
            if not os.path.exists(file_path):
                raise IOError("{} does not exist".format(file_path))

            cfg = XFasterConfig()
            cfg.read(os.path.join(file_path, "config_{}.txt".format(tag)))

            self.config[tag] = cfg

        return self.config[tag]

    def get_spec_fields(self, tag, fields, prefix=None):
        """
        Get a list of spec fields.

        Arguments
        ---------
        tag : string
            XFaster output tag
        fields : int, str, list of ints or strings, or 'all'
            Spectra field index (order = [TT, EE, BB, TE, EB, TB]);
            can be an int, a string, a list, or 'all' to return all available
        prefix : string
            If supplied, the prefix is prepended to each spectrum field name

        Returns
        -------
        fields : list of strings
            List of spectrum fields to be used as keys into bin_def or other
            output data dictionaries.
        """
        cfg = self.load_config(tag)
        tbeb = cfg.getboolean("Spectrum Estimation Options", "tbeb")
        specs = ["tt", "ee", "bb", "te", "eb", "tb"][: 6 if tbeb else 4]

        if str(fields) == "all":
            fields = list(specs)
        elif not isinstance(fields, list):
            fields = [fields]

        out = []
        for field in fields:
            if isinstance(field, int):
                field = specs[field]
            field = str(field).lower()
            if field not in specs:
                raise ValueError("Unrecognized field name %s" % field)
            if prefix:
                field = "{}_{}".format(prefix, field)
            out.append(field)

        return out

    def load_bp_data(self, tag, iter=None, sim_index=None, mean=False):
        """
        Load a bandpower.npz file

        Arguments
        ---------
        tag : int or str
            XFaster output tag
        iter : int
            If not None, load this iteration of the bandpower (which
            was written to disk if save_iters=True)
        sim_index : int or str
            If not None, load this sim index of the bandpower (which
            was written to disk if sim_index was set). If it can't be
            cast to an int, assume it's a string to be used in the space
            where the sim_index tag goes (like, 'mean')
        mean : bool
            If True, return results from ensemble_mean run.

        Returns
        -------
        dat : OrderedDict
            The parsed bandpowers data dictionary
        """

        key = (tag, iter, sim_index, mean)

        if key not in self.bandpowers:
            file_path = os.path.join(self.path, tag)
            if not os.path.exists(file_path):
                raise IOError("{} does not exist".format(file_path))

            file_tag = tag
            if iter is not None:
                file_tag = "iter{:03}_{}".format(iter, file_tag)
            if sim_index is not None:
                if str(sim_index).isdigit():
                    sim_index = "sim{:04}".format(sim_index)
                file_tag = "{}_{}".format(sim_index, file_tag)
            if mean:
                file_tag = "mean_{}".format(file_tag)

            filename = os.path.join(file_path, "bandpowers_{}.npz".format(file_tag))
            bp = pt.load_and_parse(filename)

            self.bandpowers[key] = bp

        return self.bandpowers[key]

    def get_bp_data(
        self,
        tag,
        fields="all",
        err=False,
        sampvar=True,
        iter=None,
        sim_index=None,
        return_ell=False,
        return_cov=False,
        return_invfish=False,
        fg_bins=False,
        mean=False,
    ):
        """
        Parse a bandpower.npz file

        Arguments
        ---------
        tag : int or str
            XFaster output tag
        fields : int, list, or string
            Spectra field index (order = [TT, EE, BB, TE, EB, TB]);
            can be an int, a list, or 'all' to return all available
        err : bool
            If True, return error on spectrum instead of spectrum
        sampvar : bool
            If true and err is true, include sample variance in the error
        iter : int
            If not None, load this iteration of the bandpower (which
            was written to disk if save_iters=True)
        sim_index : int or str
            If not None, load this sim index of the bandpower (which
            was written to disk if sim_index was set). If it can't be
            cast to an int, assume it's a string to be used in the space
            where the sim_index tag goes (like, 'mean')
        return_ell : bool
            If True, return ell bin centers
        return_cov : bool
            If True, return covariance-- includes sample variance if
            sampvar=True
        return_invfish : bool
            If True, return inverse fisher matrix-- includes sample variance if
            sampvar=True
        fg_bins : bool
            If True, instead of returning results for CMB fits, return
            dust fit bins
        mean : bool
            If True, return results from ensemble_mean run.

        Returns
        -------
        ell : arr
            Ell bin centers, if return_ell is True
        dl : arr
            The binned power spectra (or error on the power spectra, if
            err is True) in units that include ell*(ell+1)/(2*pi)
        cov : arr
            Covariance matrix, if return_cov=True
        inv_fish : arr
            Inverse Fisher matrix, if return_invfish=True
        qb_beta : float
            Fit on beta, if fg_bins=True
        qb_beta_err : float
            Error on the beta qb value fit, if fg_bins=True
        """

        bp = self.load_bp_data(tag, iter=iter, sim_index=sim_index, mean=mean)

        fields = self.get_spec_fields(tag, fields, prefix="fg" if fg_bins else "cmb")

        bin_index = pt.dict_to_index(bp["bin_def"])

        out = None
        out_ellb = None
        out_cov = None
        out_invfish = None

        for field in fields:
            ell = bp["ellb"][field]

            sl = slice(*bin_index[field])
            sv = "" if sampvar else "_nosampvar"

            if not err:
                dat = bp["cb"][field]
            else:
                dat = bp["dcb{}".format(sv)][field]

            cov = None
            if return_cov:
                cov = bp["cov{}".format(sv)][sl, sl]

            inv_fish = None
            if return_invfish:
                inv_fish = bp["inv_fish{}".format(sv)][sl, sl]

            out = dat if out is None else np.vstack([out, dat])
            if return_ell:
                out_ellb = ell if out_ellb is None else np.vstack([out_ellb, ell])
            if return_cov:
                out_cov = cov if out_cov is None else np.dstack([out_cov, cov])
            if return_invfish:
                out_invfish = (
                    inv_fish
                    if out_invfish is None
                    else np.dstack([out_invfish, inv_fish])
                )

        if len(fields) > 1:
            if return_cov:
                out_cov = np.rollaxis(out_cov, 2)
            if return_invfish:
                out_invfish = np.rollaxis(out_invfish, 2)

        qb_beta = None
        qb_beta_err = None
        if fg_bins:
            qb_beta = bp["qb"]["delta_beta"][0]
            idx = slice(*bin_index["delta_beta"])
            qb_beta_err = np.sqrt(np.diag(bp["inv_fish"][idx, idx]))[0]

        return (
            (out_ellb,) * return_ell
            + (out,)
            + (out_cov,) * return_cov
            + (out_invfish,) * return_invfish
            + (qb_beta, qb_beta_err) * fg_bins
        )

    def load_tf_data(self, tag):
        """
        Load a transfer.npz file

        Arguments
        ---------
        tag : str
            XFaster output tag

        Returns
        -------
        dat : OrderedDict
            The transfer function data dictionary
        """
        if tag not in self.transfer:

            file_path = os.path.join(self.path, tag)
            if not os.path.exists(file_path):
                raise IOError("{} does not exist".format(file_path))

            filename = os.path.join(file_path, "transfer_all_{}.npz".format(tag))
            tf = pt.load_and_parse(filename)

            self.transfer[tag] = tf

        return self.transfer[tag]

    def get_tf_data(self, tag, fields="all", map=None, average=True):
        """
        Parse a transfer.npz file

        Arguments
        ---------
        tag : str
            XFaster output tag
        fields : int, list, or string
            Spectra field index (order = [TT, EE, BB, TE, EB, TB]);
            can be an int, a list, or 'all' to return all available
        map : int
            If not None, only return the transfer function for this map
            used in multi-map mode
        average : bool
            If True, return the average of the multi-map transfer functions.
            If False, return transfer functions for each map.
            Ignored if map is not None

        Returns
        -------
        F_ell : arr
            An array of filter transfer functions. If average is False in
            multi-map mode, array is of shape (n_maps, len(fields), n_bins).
            Else, if average or single map mode or map is not None, shape
            is (len(fields), n_bins)
        """

        tf = self.load_tf_data(tag)["qb_transfer"]

        fields = self.get_spec_fields(tag, fields, prefix="cmb")

        maps = list(tf[fields[0]])

        xfer_list = []
        for m in maps:
            data = np.vstack([tf[field][m] for field in fields])
            xfer_list.append(data)

        if map is not None:
            if map in maps:
                map = maps.index(map)
            out = xfer_list[map]
        elif average:
            out = np.mean(np.asarray(xfer_list), axis=0)
        else:
            out = np.asarray(xfer_list)

        return out

    def load_beam_data(self, tag, fwhm=None, beam_product=None):
        """
        Load a beam.npz file

        Arguments
        ---------
        tag : str
            XFaster output tag
        fwhm : float
            Beam width in arcmin
        beam_product : string
            Name of beam product to use.  Overrides fwhm.

        Returns
        -------
        dat : OrderedDict
            The beam window data dictionary.
        """
        key = (tag, fwhm, beam_product)

        if key not in self.beam:

            file_path = os.path.join(self.path, tag)
            if not os.path.exists(file_path):
                raise IOError("{} does not exist".format(file_path))

            if beam_product is not None:
                name = beam_product
            elif fwhm is not None:
                name = "{:.1f}".format(fwhm)
            else:
                name = "default_fwhm"

            filename = os.path.join(file_path, "beams_{}_{}.npz".format(name, tag))
            b = pt.load_and_parse(filename)

            self.beam[key] = b

        return self.beam[key]

    def get_beam_data(
        self, tag, field=0, map=None, average=True, fwhm=None, beam_product=None
    ):
        """
        Parse a beams.npz file

        Arguments
        ---------
        tag : str
            XFaster output tag
        field : int
            Field index for stokes parameter (order = [I, Q, U])
        map : int
            If not None, only return the beam window function for this map
            used in multi-map mode
        average : bool
            If True, return the average of the multi-map beam window functions.
            If False, return beam window functions for each map.
            Ignored if map is not None
        fwhm : float
            Beam width in arcmin
        beam_product : string
            Name of beam product to use.  Overrides fwhm.

        Returns
        -------
        B_ell : arr
            An array of beam window functions. If average is False in
            multi-map mode, array is of shape (n_maps, lmax+1).
            Else, if average or single map mode or map is not None, shape
            is (lmax+1)
        """
        b = self.load_beam_data(tag, fwhm, beam_product)["beam_windows"]

        if isinstance(field, int):
            field = {0: "tt", 1: "ee", 2: "te"}[field]

        maps = list(b[field])
        beams_list = []
        for m in maps:
            beams_list = [b[field][m] for m in maps]

        if map is not None:
            if map in maps:
                map = list(maps).index(map)
            out = beams_list[map]
        elif average:
            out = np.mean(np.asarray(beams_list), axis=0)
        else:
            out = np.asarray(beams_list)

        return out

    def get_Dl(
        self,
        tag,
        fields="all",
        iter=None,
        sim_index=None,
        return_ell=False,
        fg_bins=False,
        mean=False,
    ):
        """
        Get binned spectra (units of ell*(ell+1)/(2*pi))

        Arguments
        ---------
        tag : str
            XFaster output tag
        fields : int, list, or string
            Spectra field index (order = [TT, EE, BB, TE, EB, TB]);
            can be an int, a list, or 'all' to return all available
        iter : int
            If not None, load this iteration of the bandpower (which
            was written to disk if save_iters=True)
        sim_index : int
            If not None, load this sim index of the bandpower (which
            was written to disk if sim_index was set)
        return_ell : bool
            If True, return ell bin centers

        Returns
        -------
        ell : arr
            Ell bin centers, if return_ell is True
        dl : arr
            The binned power spectra in units that include ell*(ell+1)/(2*pi)
        """
        return self.get_bp_data(
            tag,
            fields=fields,
            iter=iter,
            sim_index=sim_index,
            return_ell=return_ell,
            fg_bins=fg_bins,
            mean=mean,
        )

    def get_Dl_err(
        self,
        tag,
        fields="all",
        sampvar=True,
        iter=None,
        sim_index=None,
        return_ell=False,
        fg_bins=False,
        mean=False,
        return_cov=False,
        return_invfish=False,
    ):
        """
        Get binned spectra error (units of ell*(ell+1)/(2*pi))

        Arguments
        ---------
        tag : str
            XFaster output tag
        fields : int, list, or string
            Spectra field index (order = [TT, EE, BB, TE, EB, TB]);
            can be an int, a list, or 'all' to return all available
        sampvar : bool
            If true and err is true, include sample variance in the error
        iter : int
            If not None, load this iteration of the bandpower (which
            was written to disk if save_iters=True)
        sim_index : int
            If not None, load this sim index of the bandpower (which
            was written to disk if sim_index was set)
        return_ell : bool
            If True, return ell bin centers
        return_cov : bool
            If True, return covariance-- includes sample variance if
            sampvar=True
        return_invfish : bool
            If True, return inverse fisher matrix-- includes sample variance if
            sampvar=True

        Returns
        -------
        ell : arr
            Ell bin centers, if return_ell is True
        dl_err : arr
            The binned power spectra error in units that include
            ell*(ell+1)/(2*pi)
        """
        ret = self.get_bp_data(
            tag,
            fields=fields,
            err=True,
            sampvar=sampvar,
            iter=iter,
            sim_index=sim_index,
            return_ell=return_ell,
            fg_bins=fg_bins,
            mean=mean,
            return_cov=return_cov,
            return_invfish=return_invfish,
        )
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def get_Fl(self, tag, fields="all", map=None, average=True):
        """
        Get binned filter transfer function

        Arguments
        ---------
        tag : str
            XFaster output tag
        fields : int, list, or string
            Spectra field index (order = [TT, EE, BB, TE, EB, TB]);
            can be an int, a list, or 'all' to return all available
        map : int
            If not None, only return the transfer function for this map
            used in multi-map mode
        average : bool
            If True, return the average of the multi-map transfer functions.
            If False, return transfer functions for each map.
            Ignored if map is not None

        Returns
        -------
        F_ell : arr
            An array of filter transfer functions. If average is False in
            multi-map mode, array is of shape (n_maps, len(fields), n_bins).
            Else, if average or single map mode or map is not None, shape
            is (len(fields), n_bins)
        """
        ret = self.get_tf_data(tag, fields=fields, map=map, average=average)
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def get_Bl(
        self, tag, field=0, map=None, average=True, fwhm=None, beam_product=None
    ):
        """
        Get unbinned beam window functions

        Arguments
        ---------
        tag : str
            XFaster output tag
        field : int
            Field index for stokes parameter (order = [I, Q, U])
        map : int
            If not None, only return the beam window function for this map
            used in multi-map mode
        average : bool
            If True, return the average of the multi-map beam window functions.
            If False, return beam window functions for each map.
            Ignored if map is not None
        fwhm : float
            Beam width in arcmin
        beam_product : string
            Name of beam product to use.  Overrides fwhm.

        Returns
        -------
        B_ell : arr
            An array of beam window functions. If average is False in
            multi-map mode, array is of shape (n_maps, lmax+1).
            Else, if average or single map mode or map is not None, shape
            is (lmax+1).
        """
        return self.get_beam_data(
            tag,
            field=field,
            map=map,
            average=average,
            fwhm=fwhm,
            beam_product=beam_product,
        )

    def get_ensemble_cov(self, tag, sim_indices):
        """
        Calculate the ensemble mean and covariance from the simulation bandpowers output.
        Save the results to a new file.
        """

        nsim = len(sim_indices)

        key = (tag, nsim)

        if key in self.ensemble_cov:
            return self.ensemble_cov[key]

        file_path = os.path.join(self.path, tag)
        if not os.path.exists(file_path):
            raise IOError("{} does not exist".format(file_path))

        filename = os.path.join(
            file_path, "bandpowers_ensemble_cov_nsim{}_{}.npz".format(nsim, tag)
        )
        if os.path.exists(filename):
            out = pt.load_and_parse(filename)
            self.ensemble_cov[key] = out
            return out

        out = None
        qb = []
        cb = []

        for sim_index in sim_indices:
            bp = self.load_bp_data(tag, sim_index=sim_index)
            if out is None:
                out = copy.deepcopy(bp)

            qb.append(pt.dict_to_arr(bp["qb"], flatten=True))
            cb.append(pt.dict_to_arr(bp["cb"], flatten=True))

        qb = np.asarray(qb)
        cb = np.asarray(cb)
        nbins = cb.shape[1]
        qbm = pt.arr_to_dict(np.mean(qb, axis=0), out["qb"])
        cbm = pt.arr_to_dict(np.mean(cb, axis=0), out["cb"])
        qb2cb = pt.dict_to_arr(out["qb2cb"], flatten=True)
        covm = np.cov(qb.T)[:nbins, :nbins]
        dcbm = qb2cb * np.sqrt(np.abs(np.diag(covm)))

        out["qb"] = qbm
        out["cb"] = cbm
        out["dcb"] = dcbm
        out["cov"] = covm
        out["inv_fish"] = covm

        np.savez_compressed(filename, **out)
        self.ensemble_cov[key] = out
        return out
