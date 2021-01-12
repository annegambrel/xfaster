import sys
import numpy as np
from warnings import warn
from collections import OrderedDict

__all__ = [
    "parse_data",
    "dict_to_arr",
    "arr_to_dict",
    "change_dict_keys",
    "unique_tags",
    "tag_pairs",
    "dict_to_index",
    "dict_to_dmat",
    "dict_to_dsdqb_mat",
    "load_and_parse",
    "corr_index",
    "num_maps",
    "num_corr",
]


def corr_index(idx, n):
    """
    This gets the index of the auto spectrum when getting all
    pairwise combinations of n maps, where idx is the index of the
    map in the list of maps being looped through.
    """
    return idx * n - idx * (idx - 1) // 2


def num_maps(n):
    """
    Returns how many maps there are if there are n total cross spectra.
    """
    return int(np.sqrt(8 * n + 1) - 1) // 2


def num_corr(n):
    """
    Returns how many cross spectra there are if there are n total maps.
    """
    return n * (n + 1) // 2


def unique_tags(tags):
    """
    If map tags are repeated (eg, two 150 maps in different chunk
    subdirectories), return a list modifying them with an index
    """
    if len(np.unique(tags)) == len(tags):
        return tags
    else:
        tags = np.asarray(tags)
        new_tags = []
        indices = {}
        for t in np.unique(tags):
            indices[t] = 0
        for i, m in enumerate(tags):
            if np.count_nonzero(tags == m) > 1:
                # append an index
                new_tags.append("{}_{}".format(m, indices[m]))
                indices[m] += 1
            else:
                new_tags.append(m)
        return new_tags


def tag_pairs(tags, index=False):
    """
    Return an OrderedDict whose keys are pairs of tags in the format "tag1:tag2"
    and whose values are a tuple of the two tags used to construct each key,
    or a tuple of the indices of the two tags in the original tag list, if
    `index` is True.

    Example
    -------
        >>> tags = ['a', 'b']
        >>> tag_pairs(tags)
        OrderedDict([('a:a', ('a', 'a')), ('a:b', ('a', 'b')), ('b:b', ('b', 'b'))])
        >>> tag_pairs(tags, index=True)
        OrderedDict([('a:a', (0, 0)), ('a:b', (0, 1)), ('b:b', (1, 1))])
    """
    pairs = OrderedDict()
    for it0, t0 in enumerate(tags):
        for it1, t1 in zip(range(it0, len(tags)), tags[it0:]):
            pairs["{}:{}".format(t0, t1)] = (it0, it1) if index else (t0, t1)
    return pairs


def dict_decode(d):
    if not isinstance(d, dict):
        if isinstance(d, bytes):
            return d.decode()
        if isinstance(d, np.ndarray) and d.dtype.char == "S":
            return d.astype(str)
        return d
    d2 = d.__class__()
    for k, v in d.items():
        if isinstance(k, bytes):
            k = k.decode()
        d2[k] = dict_decode(v)
    return d2


def load_compat(*args, **kwargs):
    if sys.version_info.major > 2:
        kwargs.setdefault("encoding", "latin1")
    if np.__version__ >= "1.16.0":
        kwargs.setdefault("allow_pickle", True)

    out = dict(np.load(*args, **kwargs))

    for k, v in out.items():
        # convert singletons to scalars
        if not v.shape:
            v = v.item()

        # handle unicode data
        if sys.version_info.major > 2:
            v = dict_decode(v)

        out[k] = v

    return out


def parse_data(data, field, indices=None):
    """
    Look for a field in some data, return as a dictionary with
    descriptive keys.

    Arguments
    ---------
    data : str or dict
        Either the path to an npz file on disk or a loaded npz dict
    field : str
        Which key in data to return as a dictionary.
        Options: bin_def, cls_residual, cbl, qb, fqb, qb_transfer,
                 cb, dcb, dcb_nosampvar, ellb, qb2cb, cls_obs,
                 cls_fg, cls_signal, cls_model, cls_noise, cls_data,
                 cls_shape, wls, w1, w2, w4, fsky, kern, pkern,
                 mkern, xkern, beam_windows, Dmat_obs, Dmat1,
                 dSdqb_mat1
    indices : str
        If given, get the coordinates of this subfield in the original
        bin_def matrix and return a boolean mask where that subfield
        is indexed. Used internally
    """
    if isinstance(data, str):
        data = load_compat(data)
    if "data_version" in data:
        version = data["data_version"]
    else:
        version = -1
    if version > 0:
        # data is already stored in proper format
        return data[field]
    # Parse bin_def first, if it's in the dictionary.
    all_specs = np.array(["tt", "ee", "bb", "te", "eb", "tb"])
    if "bin_def" in data:
        bin_def = data["bin_def"]
        specs = all_specs
        bd = OrderedDict()
        for i, spec in enumerate(specs):
            # populate cmb bins
            spec_bins = bin_def[bin_def[:, 0] == i][:, 1:]
            if len(spec_bins):
                bd["cmb_{}".format(spec)] = spec_bins
                if field == "bin_def" and indices == "cmb_{}".format(spec):
                    return bin_def[:, 0] == i
            else:
                # don't need to include specs that aren't solved for
                specs.remove(spec)
        if np.any(bin_def[:, 0] > 6):
            # populate foreground bins
            for i, spec in enumerate(specs):
                bd["fg_{}".format(spec)] = bin_def[bin_def[:, 0] == i + 10][:, 1:]
                if field == "bin_def" and indices == "fg_{}".format(spec):
                    return bin_def[:, 0] == i + 10
            do_fg = True
        else:
            nbinsf = 0
            do_fg = False
        # get cmb and fg bin parameters for future use
        lmin = np.min(bd["cmb_tt"][:, 0])
        lmax = np.max(bd["cmb_tt"][:, 1])
        dl_cmb = bd["cmb_tt"][0, 1] - bd["cmb_tt"][0, 0]
        nbinsc = len(np.arange(lmin, lmax, dl_cmb))
        if do_fg:
            dl_fg = bd["fg_tt"][0, 1] - bd["fg_tt"][0, 0]
            nbinsf = len(np.arange(lmin, lmax, dl_fg))

        if np.any(bin_def[:, 0] == 20):
            bd["delta_beta"] = bin_def[bin_def[:, 0] == 20][:, 1:]
            if indices == "delta_beta":
                return bin_def[:, 0] == 20

        if np.any(bin_def[:, 0] == 6):
            # populate residual bins
            if version < 1:
                warn("Noise residual bin definitions may be incorrect!")
            nmaps = len(data["map_tags"])
            if "nbins_res" in data:
                nbins_res = data["nbins_res"]
            else:
                nbins_res = (bin_def[:, 0] == 6).sum()
            # need to figure out which spectra are being fit per map--
            # TT, EE, and BB, EE and BB, or EE and BB fixed together.
            # Assume which one based on number of residual bins.
            resid_bins = bin_def[bin_def[:, 0] == 6]
            start_resid = np.where(bin_def[:, 0] == 6)[0][0]
            dl_resid = resid_bins[0, 2] - resid_bins[0, 1]
            nbinsr = len(np.arange(lmin, lmax, dl_resid))
            nspec_resid = nbins_res / nmaps / nbinsr
            resid_specs = {1: ["eebb"], 2: ["ee", "bb"], 3: ["tt", "ee", "bb"]}
            for i, m in enumerate(unique_tags(data["map_tags"])):
                for j, s in enumerate(resid_specs[nspec_resid]):
                    start = nspec_resid * i * nbinsr + j
                    stop = start + nbinsr * nspec_resid
                    rbins = resid_bins[start:stop:nspec_resid][:, 1:]
                    bd["res_{}_{}".format(s, m)] = rbins
                    if field == "bin_def" and indices == "res_{}_{}".format(s, m):
                        all_bins = np.zeros(len(bin_def), dtype=bool)
                        start += start_resid
                        stop += start_resid
                        all_bins[start:stop:nspec_resid] = True
                        return all_bins
        else:
            nbinsr = 0
    else:
        bd = None
        specs = None

    if field == "bin_def":
        return bd

    elif field == "cls_residual":
        # this array is [N maps x N noise bins, xspecs (only non-zero for auto),
        #                6 (TT-TB specs), lmax]
        cls_res = data[field]
        res_dict = OrderedDict()
        for s, spec in enumerate(specs):
            for im, m0 in enumerate(unique_tags(data["map_tags"])):
                diag = corr_index(im, len(data["map_tags"]))
                res = 0
                for ib in range(nbinsr):
                    d = cls_res[im * nbinsr + ib, diag, s]
                    if ib == 0:
                        res = d
                    else:
                        res = np.vstack([res, d])
                if np.any(res != 0):
                    res_dict.setdefault(spec, OrderedDict())
                    res_dict[spec]["{0}:{0}".format(m0)] = res
        return res_dict

    elif field == "cbl":
        # this array is [xspecs, CMB+FG bins, 2 (EE/BB mixing), lmax]
        # if this is a cbl for a transfer function, first dimension is 1--
        # just one transfer per map, not per xspec
        cbl = data[field]
        cbl_dict = OrderedDict()
        if cbl.shape[0] == 1 and len(data["map_tags"]) > 1:
            # this is a transfer function dict. only use that map.
            # not enough info to know which map, so just call
            # it transfer and depend on user to know.
            transfer = True
            maps = ["transfer"]
            map_pairs = {"transfer": "transfer"}
        else:
            transfer = False
            maps = unique_tags(data["map_tags"])
            map_pairs = tag_pairs(maps)
        comps = ["cmb"]
        if do_fg and not transfer:
            comps += ["fg"]
        if nbinsr > 0 and "cls_residual" in data and not transfer:
            comps += ["res"]
            cls_residual = parse_data(data, "cls_residual")
        for comp in comps:
            for s, spec in enumerate(specs):
                sn = "{}_{}".format(comp, spec)
                snmix = sn + "_mix"
                for xind, xname in enumerate(map_pairs):
                    if comp == "cmb":
                        i0 = nbinsc * s
                        di = nbinsc
                    elif comp == "fg":
                        # fg bins start after all the cmb bins
                        i0 = len(specs) * nbinsc + nbinsf * s
                        di = nbinsf
                    elif comp == "res":
                        if spec not in cls_residual:
                            continue
                        if xname not in cls_residual[spec]:
                            continue
                        # store res terms directly from cls_residual
                        cbl_dict.setdefault(sn, OrderedDict())
                        cbl_dict[sn][xname] = cls_residual[spec][xname]
                        continue
                    else:
                        continue

                    cblx = cbl[xind]
                    cbl_dict.setdefault(sn, OrderedDict())
                    cbl_dict[sn][xname] = cblx[i0 : i0 + di, 0]
                    if spec == "ee":
                        cbl_dict.setdefault(snmix, OrderedDict())
                        cbl_dict[snmix][xname] = cblx[i0 + di : i0 + 2 * di, 1]
                    elif spec == "bb":
                        cbl_dict.setdefault(snmix, OrderedDict())
                        cbl_dict[snmix][xname] = cblx[i0 - di : i0, 1]

        return cbl_dict

    elif field in ["qb", "fqb"]:
        # This is a vector of length bin_def
        qb = data[field]
        qb_dict = bd.copy()
        for k in qb_dict:
            qb_dict[k] = qb[parse_data(data, "bin_def", indices=k)]
        return qb_dict

    elif field == "qb_transfer":
        # This is an array of shape [N maps, CMB bins].
        # Complicated because fields in transfer_mapN_* do not match
        # transfer_all_*, and transfer_all* doesn't necessarily contain
        # the list of map tags, so use map indices instead in that case.
        qbt = data[field]
        qbt_dict = OrderedDict()
        # figure out which specs:
        nspecs = np.min([6, data["cls_shape"].shape[0]])
        spec_inds = np.arange(nspecs)
        specs = all_specs[spec_inds]
        if "map_tags" in data:
            maps = unique_tags(data["map_tags"])
        else:
            maps = np.arange(qbt.shape[0])

        nbinsc = qbt.shape[-1] / len(maps)
        for s, spec in enumerate(specs):
            stag = "cmb_{}".format(spec)
            qbt_dict[stag] = OrderedDict()
            for im0, m0 in enumerate(maps):
                s0 = s * nbinsc
                qbt_dict[stag][m0] = qbt[im0, s0 : s0 + nbinsc]
        return qbt_dict

    elif field in ["cb", "dcb", "dcb_nosampvar", "ellb", "qb2cb"]:
        # These are vectors of length cmb bins + fg bins
        dat = data[field]
        dat_dict = OrderedDict()
        comps = ["cmb"]
        if do_fg:
            comps += ["fg"]
        for comp in comps:
            for s, spec in enumerate(specs):
                fname = "{}_{}".format(comp, spec)
                if comp == "cmb":
                    s0 = s * nbinsc
                    dat_dict[fname] = dat[s0 : s0 + nbinsc]
                elif comp == "fg":
                    s0 = len(specs) * nbinsc + s * nbinsf
                    dat_dict[fname] = dat[s0 : s0 + nbinsf]
        return dat_dict

    elif field in [
        "cls_obs",
        "cls_fg",
        "cls_signal",
        "cls_model",
        "cls_noise",
        "cls_data",
    ]:
        # These are arrays of shape [xspecs, cl specs, lmax]
        dat = data[field]
        dat_dict = OrderedDict()
        # If this is a transfer function dict, some of these fields will exist
        # but be empty
        try:
            len(dat)
        except TypeError:
            warn("Field {} is empty".format(field))
            return None
        if dat.shape[0] == 1 and len(data["map_tags"]) > 1:
            # this is a transfer function dict. only use that map.
            # not enough info to know which map, so just call
            # it transfer and depend on user to know.
            maps = ["transfer"]
            # some fields are also not computed for all six spectra
            if len(specs) > dat.shape[1]:
                specs = specs[: dat.shape[1]]
            map_pairs = {"transfer": "transfer"}
        else:
            if "map_tags" in data:
                maps = unique_tags(data["map_tags"])
            else:
                # in sims_*.npz file, no map_tags field, so name by index
                maps = np.arange(num_maps(dat.shape[0]))
            map_pairs = tag_pairs(maps)

        if specs is None:
            # for *_xcorr_* files, there is no bin def field so specs is not set
            specs = all_specs[np.arange(dat.shape[1])]

        for s, spec in enumerate(specs):
            if field == "cls_model":
                stag = "total_{}".format(spec)
            elif field == "cls_fg":
                stag = "fg_{}".format(spec)
            else:
                stag = spec
            dat_dict[stag] = OrderedDict()
            for xspec_ind, xname in enumerate(map_pairs):
                dat_dict[stag][xname] = dat[xspec_ind, s]

        if field == "cls_model" and "cls_fg" in data:
            if data["cls_fg"] is not None:
                dat_dict_fg = parse_data(data, "cls_fg")
                for k, v in dat_dict_fg.items():
                    dat_dict[k] = v

        return dat_dict

    elif field == "cls_shape":
        # This is an array of shape [CMB specs + FG spec, 2*lmax+1]
        cls_shape = data[field]
        shape_dict = OrderedDict()
        for s, spec in enumerate(specs):
            shape_dict["cmb_{}".format(spec)] = cls_shape[s]
        if do_fg:
            shape_dict["fg"] = cls_shape[-1]
        return shape_dict

    elif field == "wls":
        # This is an array of shape [xspecs, I/Q/U, lmax]
        wls = data[field]
        wls_dict = OrderedDict()
        # masks_xcorr doesn't have names of maps
        maps = np.arange(num_maps(wls.shape[0]))
        map_pairs = tag_pairs(maps)
        for xspec_ind, xname in enumerate(map_pairs):
            wls_dict[xname] = wls[xspec_ind]
        return wls_dict

    elif field in ["w1", "w2", "w4", "fsky"]:
        # This is an array of shape [xspecs, I/Q/U] that is only non-zero
        # for autos
        w = data[field]
        w_dict = OrderedDict()
        # masks_xcorr doesn't have names of maps
        maps = np.arange(num_maps(w.shape[0]))
        for im0, m0 in enumerate(maps):
            diag = corr_index(im0, len(maps))
            w_dict["{}".format(m0)] = w[diag]
        return w_dict

    elif field in ["kern", "pkern", "mkern", "xkern"]:
        # This is an array of shape [xspecs, lmax+1, 2*lmax+1]
        kern = data[field]
        kern_dict = OrderedDict()
        xspec_ind = 0
        # kernels doesn't have names of maps
        maps = np.arange(num_maps(kern.shape[0]))
        map_pairs = tag_pairs(maps)
        for xspec_ind, xname in enumerate(map_pairs):
            kern_dict[xname] = kern[xspec_ind]
        return kern_dict

    elif field == "beam_windows":
        # This is an array of shape [nmaps, TT/EE/TE, 2*lmax+1]
        b = data[field]
        b_dict = OrderedDict()
        # beams.npz doesn't have names of maps
        maps = np.arange(b.shape[0])
        if b.shape[1] == 1:
            specs = OrderedDict([("tt", 0)])
        else:
            specs = OrderedDict(
                [("tt", 0), ("ee", 1), ("bb", 1), ("te", 2), ("eb", 1), ("tb", 2)]
            )
        for spec, s in specs.items():
            b_dict[spec] = OrderedDict()
            for im0 in maps:
                b_dict[spec]["{}".format(im0)] = b[im0][s]
        return b_dict

    elif field in ["Dmat1", "Dmat_obs"]:
        # This is an array of shape (Nmaps * 3, Nmaps * 3, lmax + 1)
        d = data[field]
        d_dict = OrderedDict()
        if "map_tags" in data:
            if data["cbl"].shape[0] == 1:
                # this is a transfer function file-- all map tags given but
                # only one computed
                maps = [0]
            else:
                maps = unique_tags(data["map_tags"])
        else:
            maps = np.arange(qbt.shape[0])

        inds = {
            "tt": [0, 0],
            "ee": [1, 1],
            "bb": [2, 2],
            "te": [0, 1],
            "eb": [1, 2],
            "tb": [0, 2],
        }

        if len(specs) > 1:
            pol_dim = 3
        else:
            pol_dim = 1

        map_pairs = tag_pairs(maps, index=True)
        for xname, (im0, im1) in map_pairs.items():
            d_dict[xname] = OrderedDict()
            for spec in specs:
                ind = inds[spec]
                d_dict[xname][spec] = d[im0 * pol_dim + ind[0], im1 * pol_dim + ind[1]]
        return d_dict

    elif field in ["dSdqb_mat1", "dSdqb_mat1_freq"]:
        # This is an array of shape
        # (Nmaps * 3, Nmaps * 3, cmb+fg+res+delta_beta bins, lmax + 1)
        d = data[field]
        d_dict = OrderedDict()
        if "map_tags" in data:
            if data["cbl"].shape[0] == 1:
                # this is a transfer function file-- all map tags given but
                # only one computed
                maps = [0]
            else:
                maps = unique_tags(data["map_tags"])
        else:
            maps = np.arange(qbt.shape[0])

        inds = {
            "tt": [0, 0],
            "ee": [1, 1],
            "bb": [2, 2],
            "te": [0, 1],
            "eb": [1, 2],
            "tb": [0, 2],
        }

        if len(specs) > 1:
            pol_dim = 3
        else:
            pol_dim = 1

        comps = ["cmb"]
        if nbinsf > 0:
            comps += ["fg", "delta_beta"]
        if nbinsr > 0:
            comps += ["res"]

        map_pairs = tag_pairs(maps, index=True)
        for comp in comps:
            for xname, (im0, im1) in map_pairs.items():
                do_specs = specs
                if comp == "res":
                    if im0 != im1:
                        continue
                    if nspec_resid == 3:
                        do_specs = ["tt", "ee", "bb"]
                    else:
                        # eebb or ee + bb
                        do_specs = ["ee", "bb"]
                for s, spec in enumerate(do_specs):
                    ind = inds[spec]
                    ind0 = im0 * pol_dim + ind[0]
                    ind1 = im1 * pol_dim + ind[1]
                    if comp == "cmb":
                        s0 = s * nbinsc
                        ds = nbinsc
                    elif comp == "fg":
                        s0 = nbinsc * len(specs) + s * nbinsf
                        ds = nbinsf
                    elif comp == "delta_beta":
                        # add a single entry for delta_beta
                        s0 = (nbinsc + nbinsf) * len(specs)
                        ds = 1
                    elif comp == "res":
                        s0 = (nbinsc + nbinsf) * len(specs)
                        if nbinsf > 0:
                            s0 += 1
                        if nspec_resid > 1:
                            # EE and BB spectra use the same bins if nspec_resid = 1
                            # otherwise, separate bins for each spectrum
                            s0 += s * nbinsr
                        s0 += im0 * nbinsr
                        ds = nbinsr
                    dd = d[:, :, s0 : s0 + ds]
                    d_dict.setdefault(comp, OrderedDict()).setdefault(
                        xname, OrderedDict()
                    ).setdefault(spec, OrderedDict())
                    d_dict[comp][xname][spec][spec] = dd[ind0, ind1]
                    if comp in ["cmb", "fg"]:
                        if spec == "ee":
                            d_dict[comp][xname]["ee"]["bb"] = dd[ind0 + 1, ind1 + 1]
                        elif spec == "bb":
                            d_dict[comp][xname]["bb"]["ee"] = dd[ind0 - 1, ind1 - 1]

        return d_dict

    else:
        return data[field]


def dict_to_arr(d, out=None, flatten=False):
    """
    Transform ordered dict into an array, if all items are same shape

    If not all items are the same shape, eg, for qb, or if flatten=True,
    flatten everything into a vector
    """
    if not isinstance(d, dict):
        return d
    for key, val in d.items():
        if isinstance(val, dict):
            out = dict_to_arr(val, out=out, flatten=flatten)
        else:
            val = np.atleast_1d(val)
            if out is None:
                out = val
            else:
                if val.shape[-1] == out.shape[-1] and not flatten:
                    out = np.vstack([out, val])
                else:
                    out = np.append(out.flatten(), val.flatten())
    return out


def arr_to_dict(arr, ref_dict):
    """
    Transform an array of data into a dictionary keyed by the same keys in
    ref_dict, with data divided into chunks of the same length as in ref_dict.
    Requires that the length of the array is the sum of the lengths of the
    arrays in each entry of ref_dict.  The other dimensions of the input
    array and reference dict can differ.
    """
    out = OrderedDict()
    idx = 0
    assert len(arr) == sum([len(v) for v in ref_dict.values()])
    for k, bd in ref_dict.items():
        out[k] = arr[idx : idx + len(bd)]
        idx += len(bd)
    return out


def dict_to_index(d):
    """
    Construct a dictionary of (start, stop) indices that correspond to the
    location of each sub-array when the dict is converted to a single array
    using `dict_to_arr`.

    For example, use this function to index into a (nbins, nbins) array:

        bin_index = dict_to_index(bin_def)

        # extract TT bins from fisher matrix
        sl_tt = slice(*bin_index['cmb_tt'])
        fisher_tt = fisher[sl_tt, sl_tt]

        # extract all CMB bins from fisher matrix
        sl_cmb = slice(bin_index['cmb_tt'][0], bin_index['cmb_tb'][1])
        fisher_cmb = fisher[sl_cmb, sl_cmb]
    """
    index = OrderedDict()
    idx = 0
    for k, v in d.items():
        index[k] = (idx, idx + len(v))
        idx += len(v)
    return index


def change_dict_keys(dat, keys):
    """
    For each key in dat that is actually an index, replace with
    key that corresponds to that index-- needed for dicts that have to
    be constructed without knowing what maps are called.
    """
    ndat = OrderedDict()
    for k, v in dat.items():
        # key can be index:index or just index
        if ":" in k:
            ind0 = int(k.split(":")[0])
            ind1 = int(k.split(":")[1])
            k_new = "{}:{}".format(keys[ind0], keys[ind1])
            ndat[k_new] = v
        else:
            k_new = keys[int(k)]
            ndat[k_new] = v
    return ndat


def dict_to_dmat(dmat_dict):
    """
    Take a dmat dictionary and return the right shaped Dmat matrix:
    (Nmaps * 3, Nmaps * 3, lmax + 1) if pol else
    (Nmaps, Nmaps, lmax + 1)
    """
    nmaps = num_maps(len(dmat_dict))

    # get the unique map tags in order from the keys map1:map2
    mtags = [x.split(":")[0] for x in dmat_dict]
    _, uind = np.unique(mtags, return_index=True)
    map_tags = np.asarray(mtags)[sorted(uind)]
    map_pairs = tag_pairs(map_tags, index=True)

    nmaps = len(map_tags)
    pol_dim = 0

    Dmat = None
    inds = {
        "tt": [0, 0],
        "ee": [1, 1],
        "bb": [2, 2],
        "te": [0, 1],
        "eb": [1, 2],
        "tb": [0, 2],
    }

    for xname, (im0, im1) in map_pairs.items():
        pol_dim = 3 if "ee" in dmat_dict[xname] else 1
        for spec, val in dmat_dict[xname].items():
            if Dmat is None:
                shape = (pol_dim * nmaps, pol_dim * nmaps)
                if not np.isscalar(val):
                    shape += (len(val),)
                Dmat = np.zeros(shape)
            sind = inds[spec]
            xind = im0 * pol_dim + sind[0]
            yind = im1 * pol_dim + sind[1]
            Dmat[xind, yind] = Dmat[yind, xind] = val
            xind = im1 * pol_dim + sind[0]
            yind = im0 * pol_dim + sind[1]
            Dmat[xind, yind] = Dmat[yind, xind] = val

    return Dmat


def dict_to_dsdqb_mat(dsdqb_dict, bin_def):
    """
    Take a dSdqb dictionary and return the right shaped dSdqb matrix:
    (Nmaps * 3, Nmaps * 3, nbins_cmb+nbins_fg+nbins_res, lmax + 1) if pol
    else first two dimensions are Nmaps.

    If gmat is given, the terms in the resulting matrix are multiplied by the
    appriopriate mode density term.
    """
    # get the unique map tags in order from the keys map1:map2
    mtags = [x.split(":")[0] for x in dsdqb_dict["cmb"]]
    _, uind = np.unique(mtags, return_index=True)
    map_tags = np.asarray(mtags)[sorted(uind)]
    map_pairs = tag_pairs(map_tags, index=True)

    nmaps = len(map_tags)
    pol_dim = 3 if "cmb_ee" in bin_def else 1

    inds = {
        "tt": [0, 0],
        "ee": [1, 1],
        "bb": [2, 2],
        "te": [0, 1],
        "eb": [1, 2],
        "tb": [0, 2],
    }

    bin_index = dict_to_index(bin_def)
    nbins = bin_index[list(bin_index)[-1]][-1]

    dsdqb_mat = None
    seen_keys = []

    for key, (start, stop) in bin_index.items():
        bins = slice(start, stop)

        if key == "delta_beta":
            comp = "delta_beta"
            specs = ["tt", "ee", "bb", "te", "eb", "tb"]
            pairs = map_pairs
        else:
            comp, rem = key.split("_", 1)
            if "_" in rem:
                specs, tag = rem.split("_", 1)
                xname = "{0}:{0}".format(tag)
                pairs = {xname: map_pairs[xname]}
                if specs == "eebb":
                    specs = ["ee", "bb"]
                else:
                    specs = [specs]
            else:
                specs = [rem]
                pairs = map_pairs

        if comp not in dsdqb_dict:
            continue

        for xname, (im0, im1) in pairs.items():
            if xname not in dsdqb_dict[comp]:
                continue
            for spec in specs:
                if spec not in dsdqb_dict[comp][xname]:
                    continue
                for spec2, d2 in dsdqb_dict[comp][xname][spec].items():
                    if dsdqb_mat is None:
                        sz = d2.shape[-1]
                        dsdqb_mat = np.zeros(
                            (nmaps * pol_dim, nmaps * pol_dim, nbins, sz)
                        )
                    sind = inds[spec2]
                    ind0 = im0 * pol_dim + sind[0]
                    ind1 = im1 * pol_dim + sind[1]
                    dsdqb_mat[ind0, ind1, bins] = dsdqb_mat[ind1, ind0, bins] = d2
                    ind0 = im1 * pol_dim + sind[0]
                    ind1 = im0 * pol_dim + sind[1]
                    dsdqb_mat[ind0, ind1, bins] = dsdqb_mat[ind1, ind0, bins] = d2
                if key not in seen_keys:
                    seen_keys.append(key)

    # transfer function runs do not include tbeb in the dsdqb matrix
    nbins_seen = max([bin_index[k][-1] for k in seen_keys])
    if nbins_seen != nbins:
        dsdqb_mat = dsdqb_mat[:, :, :nbins_seen, :]

    return dsdqb_mat


def load_and_parse(filename):
    """
    Load a .npz data file from disk and parse all the fields it contains.

    Returns a dictionary of parsed fields.
    """
    data = load_compat(filename)
    ret = dict()
    for k in data:
        ret[k] = parse_data(data, k)
    return ret
