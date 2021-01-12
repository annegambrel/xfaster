from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import camb
from collections import OrderedDict

__all__ = [
    "ThreeJC_2",
    "get_camb_cl",
    "expand_qb",
    "bin_spec",
    "scale_dust",
]


def BBody(nu, ref_freq=353.0):
    k = 1.38064852e-23  # Boltzmann constant
    h = 6.626070040e-34  # Planck constant
    T = 19.6
    nu_ref = ref_freq * 1.0e9
    # T = 2.725 #Cmb BB temp in K
    nu *= 1.0e9  # Ghz
    x = h * nu / k / T
    x_ref = h * nu_ref / k / T
    return x ** 3 / x_ref ** 3 * (np.exp(x_ref) - 1) / (np.exp(x) - 1)


def RJ2CMB(nu_in, ccorr=True):
    """
    planck_cc_gnu = {101: 1.30575, 141: 1.6835, 220: 3.2257, 359: 14.1835}
    if ccorr:
        if np.isscalar(nu_in):
            for f, g in planck_cc_gnu.items():
                if int(nu_in) == int(f):
                    return g
    """
    k = 1.38064852e-23  # Boltzmann constant
    h = 6.626070040e-34  # Planck constant
    T = 2.72548  # Cmb BB temp in K
    nu = nu_in * 1.0e9  # Ghz
    x = h * nu / k / T
    return (np.exp(x) - 1.0) ** 2 / (x ** 2 * np.exp(x))


def scale_dust(freq0, freq1, ref_freq, beta, delta_beta=None, deriv=False):
    """
    Get the factor by which you must dividide the cross spectrum from maps of
    frequencies freq0 and freq1 to match the dust power at ref_freq given
    spectra index beta.

    If deriv is True, return the frequency scaling at the reference beta,
    and the first derivative w.r.t. beta.

    Otherwise if delta_beta is given, return the scale factor adjusted
    for a linearized offset delta_beta from the reference beta.
    """
    freq_scale = (
        RJ2CMB(freq0)
        * RJ2CMB(freq1)
        / RJ2CMB(ref_freq) ** 2.0
        * BBody(freq0, ref_freq=ref_freq)
        * BBody(freq1, ref_freq=ref_freq)
        * (freq0 * freq1 / ref_freq ** 2) ** (beta - 2.0)
    )

    if deriv or delta_beta is not None:
        delta = np.log(freq0 * freq1 / ref_freq ** 2)
        if deriv:
            return (freq_scale, freq_scale * delta)
        return freq_scale * (1 + delta * delta_beta)

    return freq_scale


def ThreeJC_2(l2i, m2i, l3i, m3i):
    """
    Wigner 3j symbols
    """
    try:
        from camb.mathutils import threej
    except ImportError:
        from camb.bispectrum import threej
    arr = threej(l2i, l3i, m2i, m3i)

    lmin = np.max([np.abs(l2i - l3i), np.abs(m2i + m3i)])
    lmax = l2i + l3i
    fj = np.zeros(lmax + 2, dtype=arr.dtype)
    fj[lmin : lmax + 1] = arr
    return fj, lmin, lmax


# base_r_plikHM_TT_lowTEB_lensing.minimum
#    1  0.2226767E-01   omegabh2              \Omega_b h^2
#    2  0.1184800E+00   omegach2              \Omega_c h^2
#    3  0.1041032E+01   theta                 100\theta_{MC}
#    4  0.6715808E-01   tau                   \tau
#   17  0.3064628E+01   logA                  {\rm{ln}}(10^{10} A_s)
#   18  0.9686477E+00   ns                    n_s
#   21  0.3067417E-01   r                     r
#    5  0.0000000E+00   omegak                \Omega_K
#    6  0.6000000E-01   mnu                   \Sigma m_\nu
#   51  0.6784119E+02   H0                    H_0


def get_camb_cl(r, lmax, nt=None, spec="total"):
    """
    Compute camb spectrum with tensors and lensing.

    Arguments
    ---------
    r : float
        Tensor-to-scalar ratio
    lmax : int
        Maximum ell for which to compute spectra
    nt : scalar, optional
        Tensor spectral index.  If not supplied, assumes
        slow-roll consistency relation.
    spec : string, optional
        Spectrum component to return.  Can be 'total', 'unlensed_total',
        'unlensed_scalar', 'lensed_scalar', 'tensor', 'lens_potential'.

    Returns
    -------
    cls : array_like
        Array of spectra of shape (lmax + 1, nspec), including the
        ell*(ell+1)/2/pi scaling. Diagonal ordering (TT, EE, BB, TE).
    """
    # Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()

    # This function sets up CosmoMC-like settings, with one massive neutrino and
    # helium set using BBN consistency
    pars.set_cosmology(
        H0=67.84119, ombh2=0.02226767, omch2=0.1184800, mnu=0.06, omk=0, tau=0.06715808
    )
    pars.InitPower.set_params(As=2.142649e-9, ns=0.9686477, r=r, nt=nt)
    if lmax < 2500:
        # This results in unacceptable bias. Use higher lmax, then cut it down
        lmax0 = 2500
    else:
        lmax0 = lmax
    pars.set_for_lmax(lmax0, lens_potential_accuracy=2)
    pars.WantTensors = True
    pars.do_lensing = True

    # calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")

    totCL = powers[spec][: lmax + 1, :4].T
    return totCL


def expand_qb(qb, bin_def, lmax=None):
    """
    Expand a qb-type array to an ell-by-ell spectrum using bin_def.

    Arguments
    ---------
    qb : array_like, (nbins,)
        Array of bandpower deviations
    bin_def : array_like, (nbins, 2)
        Array of bin edges for each bin
    lmax : int, optional
        If supplied, limit the output spectrum to this value.
        Otherwise the output spectrum extends to include the last bin.

    Returns
    -------
    cl : array_like, (lmax + 1,)
        Array of expanded bandpowers
    """
    lmax = lmax if lmax is not None else bin_def.max() - 1

    cl = np.zeros(lmax + 1)

    for idx, (left, right) in enumerate(bin_def):
        cl[left:right] = qb[idx]

    return cl


def bin_spec(qb, cls_shape, bin_def, inv_fish=None, tophat=False, lfac=True):
    """
    Compute binned output spectra and covariances by averaging the shape
    spectrum over each bin, and applying the appropriate `qb` bandpower
    amplitude.

    Arguments
    ---------
    qb : dict
        Bandpower amplitudes for each spectrum bin.
    cls_shape : dict
        Shape spectrum
    bin_def : dict
        Bin definition dictionary
    inv_fish : array_like, (nbins, nbins)
        Inverse fisher matrix for computing the bin errors and covariance.  If
        not supplied, these are not computed.
    tophat : bool
        If True, compute binned bandpowers using a tophat weight.  Otherwise a
        logarithmic ell-space weighting is applied.
    lfac : bool
        If False, return binned C_l spectrum rather than the default D_l

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
        The conversion factor from `qb` to `cb`, computed by averaging over the
        input shape spectrum.
    """

    from . import parse_tools as pt

    lmax = pt.dict_to_arr(bin_def).max()

    qb2cb = OrderedDict()
    ellb = OrderedDict()
    cb = OrderedDict()

    ell = np.arange(lmax + 1)
    fac1 = (2 * ell + 1) / 4.0 / np.pi
    fac2 = ell * (ell + 1) / 2.0 / np.pi

    if tophat:
        fac = fac2 if lfac else 1
    else:
        fac3 = fac1.copy()
        fac3[ell > 0] /= fac2[ell > 0]
        fac = fac1 if lfac else fac3

    ecls_shape = {k: fac * v[: lmax + 1] for k, v in cls_shape.items()}

    bin_index = pt.dict_to_index(bin_def)
    nbins = 0

    for stag, qb1 in qb.items():
        comp, spec = stag.split("_", 1)

        if comp not in ["cmb", "fg"]:
            continue

        shape = ecls_shape["fg" if comp == "fg" else stag]
        ellb[stag] = np.zeros_like(qb1)
        qb2cb[stag] = np.zeros_like(qb1)

        nbins = max([nbins, bin_index[stag][1]])

        for idx, (left, right) in enumerate(bin_def[stag]):
            il = slice(left, right)
            if tophat:
                qb2cb[stag][idx] = np.mean(shape[il])
                ellb[stag][idx] = np.mean(ell[il])
            else:
                v = np.sum(shape[il])
                qb2cb[stag][idx] = v / np.sum(fac3[il])
                av = np.abs(shape[il])
                ellb[stag][idx] = np.sum(av * ell[il]) / np.sum(av)

        cb[stag] = qb1 * qb2cb[stag]

    if inv_fish is not None:
        inv_fish = inv_fish[:nbins, :nbins]
        qb2cb_arr = pt.dict_to_arr(qb2cb, flatten=True)
        dcb_arr = np.sqrt(qb2cb_arr * np.abs(np.diag(inv_fish)) * qb2cb_arr)
        dcb = pt.arr_to_dict(dcb_arr, qb2cb)
        cov = np.outer(qb2cb_arr, qb2cb_arr) * inv_fish
    else:
        dcb = None
        cov = None

    return cb, dcb, ellb, cov, qb2cb
