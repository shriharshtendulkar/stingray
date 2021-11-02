import copy
import warnings
from collections.abc import Iterable, Iterator

import numpy as np
import scipy
import scipy.optimize
import scipy.stats

from stingray.exceptions import StingrayError
from stingray.gti import bin_intervals_from_gtis, check_gtis, cross_two_gtis
from stingray.largememory import createChunkedSpectra, saveData
from stingray.utils import genDataPath, rebin_data, rebin_data_log, simon

from .events import EventList
from .lightcurve import Lightcurve
from .utils import show_progress

# location of factorial moved between scipy versions
try:
    from scipy.misc import factorial
except ImportError:
    from scipy.special import factorial

from .fourier import fftfreq, fft

__all__ = [
    "Crossspectrum", "AveragedCrossspectrum", "coherence", "time_lag",
    "cospectra_pvalue", "normalize_crossspectrum"
]


def normalize_crossspectrum(unnorm_power, tseg, nbins, nphots1, nphots2, norm="none", power_type="real"):
    """
    Normalize the real part of the cross spectrum to Leahy, absolute rms^2,
    fractional rms^2 normalization, or not at all.

    Parameters
    ----------
    unnorm_power: numpy.ndarray
        The unnormalized cross spectrum.

    tseg: int
        The length of the Fourier segment, in seconds.

    nbins : int
        Number of bins in the light curve

    nphots1 : int
        Number of photons in the light curve no. 1

    nphots2 : int
        Number of photons in the light curve no. 2

    Other parameters
    ----------------
    norm : str
        One of `'leahy'` (Leahy+83), `'frac'` (fractional rms), `'abs'`
        (absolute rms)

    power_type : str
        One of `'real'` (real part), `'all'` (all complex powers), `'abs'`
        (absolute value)

    Returns
    -------
    power: numpy.nd.array
        The normalized co-spectrum (real part of the cross spectrum). For
        'none' normalization, imaginary part is returned as well.
    """

    # The "effective" counts/bin is the geometrical mean of the counts/bin
    # of the two light curves. Same goes for counts/second in meanrate.

    log_nphots1 = np.log(nphots1)
    log_nphots2 = np.log(nphots2)

    actual_nphots = np.float64(np.sqrt(np.exp(log_nphots1 + log_nphots2)))

    if power_type == "all":
        c_num = unnorm_power
    elif power_type == "real":
        c_num = unnorm_power.real
    elif power_type == "absolute":
        c_num = np.absolute(unnorm_power)
    else:
        raise ValueError("`power_type` not recognized!")

    if norm.lower() == 'leahy':
        power = c_num * 2. / actual_nphots

    elif norm.lower() == 'frac':
        meancounts1 = nphots1 / nbins
        meancounts2 = nphots2 / nbins

        actual_mean = np.sqrt(meancounts1 * meancounts2)

        assert actual_mean > 0.0, \
            "Mean count rate is <= 0. Something went wrong."

        c = c_num / float(nbins ** 2.)
        power = c * 2. * tseg / (actual_mean ** 2.0)

    elif norm.lower() == 'abs':
        meanrate = np.sqrt(nphots1 * nphots2) / tseg

        power = c_num * 2. * meanrate / actual_nphots

    elif norm.lower() == 'none':
        power = unnorm_power

    else:
        raise ValueError("Value for `norm` not recognized.")

    return power


def normalize_crossspectrum_gauss(
        unnorm_power, mean_flux, var, dt, N, norm="none", power_type="real"):
    """
    Normalize the real part of the cross spectrum to Leahy, absolute rms^2,
    fractional rms^2 normalization, or not at all.

    Parameters
    ----------
    unnorm_power: numpy.ndarray
        The unnormalized cross spectrum.

    mean_flux: float
        The mean flux of the light curve (if a cross spectrum, the geometrical
        mean of the flux in the two channels)

    var: float
        The variance of the light curve (if a cross spectrum, the geometrical
        mean of the variance in the two channels)

    dt: float
        The sampling time of the light curve

    N: int
        The number of bins in the light curve

    Other parameters
    ----------------
    norm : str
        One of `'leahy'` (Leahy+83), `'frac'` (fractional rms), `'abs'`
        (absolute rms)

    power_type : str
        One of `'real'` (real part), `'all'` (all complex powers), `'abs'`
        (absolute value)

    Returns
    -------
    power: numpy.nd.array
        The normalized co-spectrum (real part of the cross spectrum). For
        'none' normalization, imaginary part is returned as well.

    Examples
    --------
    >>> lc_c = np.random.poisson(10000, 10000)
    >>> lc_c_var = 10000
    >>> lc = lc_c / 17.3453
    >>> lc_var = (100 / 17.3453)**2
    >>> pds_c = np.absolute(np.fft.fft(lc_c))**2
    >>> pds = np.absolute(np.fft.fft(lc))**2
    >>> norm_c = normalize_crossspectrum_gauss(pds_c, np.mean(lc_c), lc_c_var, 0.1, len(lc_c), norm='leahy')
    >>> norm = normalize_crossspectrum_gauss(pds, np.mean(lc), lc_var, 0.1, len(lc), norm='leahy')
    >>> np.allclose(norm, norm_c)
    True
    >>> np.isclose(np.mean(norm[1:]), 2, atol=0.1)
    True
    >>> norm_c = normalize_crossspectrum_gauss(pds_c, np.mean(lc_c), np.mean(lc_c), 0.1, len(lc_c), norm='frac')
    >>> norm = normalize_crossspectrum_gauss(pds, np.mean(lc), lc_var, 0.1, len(lc), norm='frac')
    >>> np.allclose(norm, norm_c)
    True
    >>> norm_c = normalize_crossspectrum_gauss(pds_c, np.mean(lc_c), np.mean(lc_c), 0.1, len(lc_c), norm='abs')
    >>> norm = normalize_crossspectrum_gauss(pds, np.mean(lc), lc_var, 0.1, len(lc), norm='abs')
    >>> np.allclose(norm / np.mean(lc)**2, norm_c / np.mean(lc_c)**2)
    True
    >>> np.isclose(np.mean(norm_c[2:]), 2 * np.mean(lc_c * 0.1), rtol=0.1)
    True
    """

    # The "effective" counts/bin is the geometrical mean of the counts/bin
    # of the two light curves. Same goes for counts/second in meanrate.
    if power_type == "all":
        c_num = unnorm_power
    elif power_type == "real":
        c_num = unnorm_power.real
    elif power_type == "absolute":
        c_num = np.absolute(unnorm_power)
    else:
        raise ValueError("`power_type` not recognized!")

    common_factor = 2 * dt / N
    rate_mean = mean_flux * dt
    if norm.lower() == 'leahy':
        norm = 2 / var / N

    elif norm.lower() == 'frac':
        norm = common_factor / rate_mean**2

    elif norm.lower() == 'abs':
        norm = common_factor

    elif norm.lower() == 'none':
        norm = 1

    else:
        raise ValueError("Value for `norm` not recognized.")

    return norm * c_num


def _averaged_cospectra_cdf(xcoord, n):
    """
    Function calculating the cumulative distribution function for
    averaged cospectra, Equation 19 of Huppenkothen & Bachetti (2018).

    Parameters
    ----------
    xcoord : float or iterable
        The cospectral power for which to calculate the CDF.

    n : int
        The number of averaged cospectra

    Returns
    -------
    cdf : float
        The value of the CDF at `xcoord` for `n` averaged cospectra
    """
    if np.size(xcoord) == 1:
        xcoord = [xcoord]

    cdf = np.zeros_like(xcoord)

    for i, x in enumerate(xcoord):
        prefac_bottom1 = factorial(n - 1)
        for j in range(n):
            prefac_top = factorial(n - 1 + j)
            prefac_bottom2 = factorial(
                n - 1 - j) * factorial(j)
            prefac_bottom3 = 2.0 ** (n + j)

            prefac = prefac_top / (prefac_bottom1 * prefac_bottom2 *
                                   prefac_bottom3)

            gf = -j + n

            first_fac = scipy.special.gamma(gf)
            if x >= 0:
                second_fac = scipy.special.gammaincc(gf, n * x) * first_fac
                fac = 2.0 * first_fac - second_fac
            else:
                fac = scipy.special.gammaincc(gf, -n * x) * first_fac

            cdf[i] += (prefac * fac)
        if np.size(xcoord) == 1:
            return cdf[i]
        else:
            continue
    return cdf


def cospectra_pvalue(power, nspec):
    """
    This function computes the single-trial p-value that the power was
    observed under the null hypothesis that there is no signal in
    the data.

    Important: the underlying assumption that make this calculation valid
    is that the powers in the power spectrum follow a Laplace distribution,
    and this requires that:

    1. the co-spectrum is normalized according to [Leahy 1983]_
    2. there is only white noise in the light curve. That is, there is no
       aperiodic variability that would change the overall shape of the power
       spectrum.

    Also note that the p-value is for a *single trial*, i.e. the power
    currently being tested. If more than one power or more than one power
    spectrum are being tested, the resulting p-value must be corrected for the
    number of trials (Bonferroni correction).

    Mathematical formulation in [Huppenkothen 2017]_.

    Parameters
    ----------
    power :  float
        The squared Fourier amplitude of a spectrum to be evaluated

    nspec : int
        The number of spectra or frequency bins averaged in ``power``.
        This matters because averaging spectra or frequency bins increases
        the signal-to-noise ratio, i.e. makes the statistical distributions
        of the noise narrower, such that a smaller power might be very
        significant in averaged spectra even though it would not be in a single
        power spectrum.

    Returns
    -------
    pval : float
        The classical p-value of the observed power being consistent with
        the null hypothesis of white noise

    References
    ----------

    * .. [Leahy 1983] https://ui.adsabs.harvard.edu/#abs/1983ApJ...266..160L/abstract
    * .. [Huppenkothen 2017] http://adsabs.harvard.edu/abs/2018ApJS..236...13H

    """
    if not np.all(np.isfinite(power)):
        raise ValueError("power must be a finite floating point number!")

    # if power < 0:
    #    raise ValueError("power must be a positive real number!")

    if not np.isfinite(nspec):
        raise ValueError("nspec must be a finite integer number")

    if not np.isclose(nspec % 1, 0):
        raise ValueError("nspec must be an integer number!")

    if nspec < 1:
        raise ValueError("nspec must be larger or equal to 1")

    elif nspec == 1:
        lapl = scipy.stats.laplace(0, 1)
        pval = lapl.sf(power)

    elif nspec > 50:
        exp_sigma = np.sqrt(2) / np.sqrt(nspec)
        gauss = scipy.stats.norm(0, exp_sigma)
        pval = gauss.sf(power)

    else:
        pval = 1. - _averaged_cospectra_cdf(power, nspec)

    return pval


def coherence(lc1, lc2):
    """
    Estimate coherence function of two light curves.
    For details on the definition of the coherence, see Vaughan and Nowak,
    1996 [#]_.

    Parameters
    ----------
    lc1: :class:`stingray.Lightcurve` object
        The first light curve data for the channel of interest.

    lc2: :class:`stingray.Lightcurve` object
        The light curve data for reference band

    Returns
    -------
    coh : ``np.ndarray``
        The array of coherence versus frequency

    References
    ----------
    .. [#] http://iopscience.iop.org/article/10.1086/310430/pdf
    """

    if not isinstance(lc1, Lightcurve):
        raise TypeError("lc1 must be a lightcurve.Lightcurve object")

    if not isinstance(lc2, Lightcurve):
        raise TypeError("lc2 must be a lightcurve.Lightcurve object")

    cs = Crossspectrum(lc1, lc2, norm='none')

    return cs.coherence()


def time_lag(lc1, lc2):
    """
    Estimate the time lag of two light curves.
    Calculate time lag and uncertainty.

        Equation from Bendat & Piersol, 2011 [bendat-2011]_.

        Returns
        -------
        lag : np.ndarray
            The time lag

        lag_err : np.ndarray
            The uncertainty in the time lag

        References
        ----------

        .. [bendat-2011] https://www.wiley.com/en-us/Random+Data%3A+Analysis+and+Measurement+Procedures%2C+4th+Edition-p-9780470248775

    """

    if not isinstance(lc1, Lightcurve):
        raise TypeError("lc1 must be a lightcurve.Lightcurve object")

    if not isinstance(lc2, Lightcurve):
        raise TypeError("lc2 must be a lightcurve.Lightcurve object")

    cs = Crossspectrum(lc1, lc2, norm='none')
    lag = cs.time_lag()

    return lag


class Crossspectrum(object):
    """
    Make a cross spectrum from a (binned) light curve.
    You can also make an empty :class:`Crossspectrum` object to populate with your
    own Fourier-transformed data (this can sometimes be useful when making
    binned power spectra). Stingray uses the scipy.fft standards for the sign
    of the Nyquist frequency.

    Parameters
    ----------
    data1: :class:`stingray.Lightcurve` or :class:`stingray.events.EventList`, optional, default ``None``
        The first light curve data for the channel/band of interest.

    data2: :class:`stingray.Lightcurve` or :class:`stingray.events.EventList`, optional, default ``None``
        The light curve data for the reference band.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, default ``none``
        The normalization of the (real part of the) cross spectrum.

    power_type: string, optional, default ``real``
        Parameter to choose among complete, real part and magnitude of the cross spectrum.

    fullspec: boolean, optional, default ``False``
        If False, keep only the positive frequencies, or if True, keep all of them .

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    lc1: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects
        For backwards compatibility only. Like ``data1``, but no
        :class:`stingray.events.EventList` objects allowed

    lc2: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects
        For backwards compatibility only. Like ``data2``, but no
        :class:`stingray.events.EventList` objects allowed

    dt: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where ``data1``, ``data2`` are
        :class:`EventList` objects


    Attributes
    ----------
    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of cross spectra (complex numbers)

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging more than one spectra). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged cross-spectra amplitudes in each bin.

    n: int
        The number of data points/time bins in one segment of the light
        curves.

    nphots1: float
        The total number of photons in light curve 1

    nphots2: float
        The total number of photons in light curve 2
    """

    def __init__(self, data1=None, data2=None, norm='none', gti=None,
                 lc1=None, lc2=None, power_type="real", dt=None, fullspec=False):

        if isinstance(norm, str) is False:
            raise TypeError("norm must be a string")

        if norm.lower() not in ["frac", "abs", "leahy", "none"]:
            raise ValueError("norm must be 'frac', 'abs', 'leahy', or 'none'!")

        self.norm = norm.lower()

        # check if input data is a Lightcurve object, if not make one or
        # make an empty Crossspectrum object if lc1 == ``None`` or lc2 == ``None``

        if lc1 is not None or lc2 is not None:
            warnings.warn("The lcN keywords are now deprecated. Use dataN "
                          "instead", DeprecationWarning)
        # for backwards compatibility
        if data1 is None:
            data1 = lc1
        if data2 is None:
            data2 = lc2

        if data1 is None or data2 is None:
            if data1 is not None or data2 is not None:
                raise TypeError("You can't do a cross spectrum with just one "
                                "light curve!")
            else:
                self.freq = None
                self.power = None
                self.power_err = None
                self.df = None
                self.nphots1 = None
                self.nphots2 = None
                self.m = 1
                self.n = None
                return

        if (isinstance(data1, EventList) or isinstance(data2, EventList)) and \
                dt is None:
            raise ValueError("If using event lists, please specify the bin "
                             "time to generate lightcurves.")

        if not isinstance(data1, EventList):
            lc1 = data1
        else:
            lc1 = data1.to_lc(dt)

        if not isinstance(data2, EventList):
            lc2 = data2
        elif isinstance(data2, EventList) and data2 is not data1:
            lc2 = data2.to_lc(dt)
        elif data2 is data1:
            lc2 = lc1

        self.gti = gti
        self.lc1 = lc1
        self.lc2 = lc2
        self.power_type = power_type
        self.fullspec = fullspec

        self._make_crossspectrum(lc1, lc2, fullspec)

        # These are needed to calculate coherence
        self._make_auxil_pds(lc1, lc2)

    def _make_auxil_pds(self, lc1, lc2):
        """
        Helper method to create the power spectrum of both light curves
        independently.

        Parameters
        ----------
        lc1, lc2 : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.
        """
        if lc1 is not lc2 and isinstance(lc1, Lightcurve):
            self.pds1 = Crossspectrum(lc1, lc1, norm='none')
            self.pds2 = Crossspectrum(lc2, lc2, norm='none')

    def _make_crossspectrum(self, lc1, lc2, fullspec=False):
        """
        Auxiliary method computing the normalized cross spectrum from two
        light curves. This includes checking for the presence of and
        applying Good Time Intervals, computing the unnormalized Fourier
        cross-amplitude, and then renormalizing using the required
        normalization. Also computes an uncertainty estimate on the cross
        spectral powers.

        Parameters
        ----------
        lc1, lc2 : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.

        fullspec: boolean, default ``False``
            Return full frequency array (True) or just positive frequencies (False)

        """

        # make sure the inputs work!
        if not isinstance(lc1, Lightcurve):
            raise TypeError("lc1 must be a lightcurve.Lightcurve object")

        if not isinstance(lc2, Lightcurve):
            raise TypeError("lc2 must be a lightcurve.Lightcurve object")

        if self.lc2.mjdref != self.lc1.mjdref:
            raise ValueError("MJDref is different in the two light curves")

        # Then check that GTIs make sense
        if self.gti is None:
            self.gti = cross_two_gtis(lc1.gti, lc2.gti)

        check_gtis(self.gti)

        if self.gti.shape[0] != 1:
            raise TypeError("Non-averaged Cross Spectra need "
                            "a single Good Time Interval")

        lc1 = lc1.split_by_gti()[0]
        lc2 = lc2.split_by_gti()[0]

        # total number of photons is the sum of the
        # counts in the light curve
        self.meancounts1 = lc1.meancounts
        self.meancounts2 = lc2.meancounts
        self.nphots1 = np.float64(np.sum(lc1.counts))
        self.nphots2 = np.float64(np.sum(lc2.counts))

        self.err_dist = 'poisson'
        if lc1.err_dist == 'poisson':
            self.var1 = lc1.meancounts
        else:
            self.var1 = np.mean(lc1.counts_err) ** 2
            self.err_dist = 'gauss'

        if lc2.err_dist == 'poisson':
            self.var2 = lc2.meancounts
        else:
            self.var2 = np.mean(lc2.counts_err) ** 2
            self.err_dist = 'gauss'

        if lc1.n != lc2.n:
            raise StingrayError("Light curves do not have same number "
                                "of time bins per segment.")

        # If dt differs slightly, its propagated error must not be more than
        # 1/100th of the bin
        if not np.isclose(lc1.dt, lc2.dt, rtol=0.01 * lc1.dt / lc1.tseg):
            raise StingrayError("Light curves do not have same time binning "
                                "dt.")

        # In case a small difference exists, ignore it
        lc1.dt = lc2.dt

        self.dt = lc1.dt
        self.n = lc1.n

        # the frequency resolution
        self.df = 1.0 / lc1.tseg

        # the number of averaged periodograms in the final output
        # This should *always* be 1 here
        self.m = 1

        # make the actual Fourier transform and compute cross spectrum
        self.freq, self.unnorm_power = self._fourier_cross(lc1, lc2, fullspec)

        # If co-spectrum is desired, normalize here. Otherwise, get raw back
        # with the imaginary part still intact.
        self.power = self._normalize_crossspectrum(self.unnorm_power, lc1.tseg)

        if lc1.err_dist.lower() != lc2.err_dist.lower():
            simon("Your lightcurves have different statistics."
                  "The errors in the Crossspectrum will be incorrect.")
        elif lc1.err_dist.lower() != "poisson":
            simon("Looks like your lightcurve statistic is not poisson."
                  "The errors in the Powerspectrum will be incorrect.")

        if self.__class__.__name__ in ['Powerspectrum',
                                       'AveragedPowerspectrum']:
            self.power_err = self.power / np.sqrt(self.m)
        elif self.__class__.__name__ in ['Crossspectrum',
                                         'AveragedCrossspectrum']:
            # This is clearly a wild approximation.
            simon("Errorbars on cross spectra are not thoroughly tested. "
                  "Please report any inconsistencies.")
            unnorm_power_err = np.sqrt(2) / np.sqrt(self.m)  # Leahy-like
            unnorm_power_err /= (2 / np.sqrt(self.nphots1 * self.nphots2))
            unnorm_power_err += np.zeros_like(self.power)

            self.power_err = \
                self._normalize_crossspectrum(unnorm_power_err, lc1.tseg)
        else:
            self.power_err = np.zeros(len(self.power))

    def _fourier_cross(self, lc1, lc2, fullspec=False):
        """
        Fourier transform the two light curves, then compute the cross spectrum.
        Computed as CS = lc1 x lc2* (where lc2 is the one that gets
        complex-conjugated). The user has the option to either get just the
        positive frequencies or the full spectrum.

        Parameters
        ----------
        lc1: :class:`stingray.Lightcurve` object
            One light curve to be Fourier transformed. Ths is the band of
            interest or channel of interest.

        lc2: :class:`stingray.Lightcurve` object
            Another light curve to be Fourier transformed.
            This is the reference band.

        fullspec: boolean. Default is False.
            If True, return the whole array of frequencies, or only positive frequencies (False).

        Returns
        -------
        fr: numpy.ndarray
            The squared absolute value of the Fourier amplitudes

        """
        fourier_1 = fft(lc1.counts)  # do Fourier transform 1
        fourier_2 = fft(lc2.counts)  # do Fourier transform 2

        freqs = scipy.fft.fftfreq(lc1.n, lc1.dt)
        cross = np.multiply(fourier_1, np.conj(fourier_2))

        if fullspec is  True:
            return freqs, cross
        else:
            return freqs[freqs > 0], cross[freqs > 0]

    def rebin(self, df=None, f=None, method="mean"):
        """
        Rebin the cross spectrum to a new frequency resolution ``df``.

        Parameters
        ----------
        df: float
            The new frequency resolution

        Other Parameters
        ----------------
        f: float
            the rebin factor. If specified, it substitutes df with ``f*self.df``

        Returns
        -------
        bin_cs = :class:`Crossspectrum` (or one of its subclasses) object
            The newly binned cross spectrum or power spectrum.
            Note: this object will be of the same type as the object
            that called this method. For example, if this method is called
            from :class:`AveragedPowerspectrum`, it will return an object of class
            :class:`AveragedPowerspectrum`, too.
        """

        if f is None and df is None:
            raise ValueError('You need to specify at least one between f and '
                             'df')
        elif f is not None:
            df = f * self.df

        # rebin cross spectrum to new resolution
        binfreq, bincs, binerr, step_size = \
            rebin_data(self.freq, self.power, df, self.power_err,
                       method=method, dx=self.df)
        # make an empty cross spectrum object
        # note: syntax deliberate to work with subclass Powerspectrum
        bin_cs = copy.copy(self)

        # store the binned periodogram in the new object
        bin_cs.freq = binfreq
        bin_cs.power = bincs
        bin_cs.df = df
        bin_cs.n = self.n
        bin_cs.norm = self.norm
        bin_cs.nphots1 = self.nphots1
        bin_cs.power_err = binerr

        if hasattr(self, 'unnorm_power'):
            _, binpower_unnorm, _, _ = \
                rebin_data(self.freq, self.unnorm_power, df,
                           method=method, dx=self.df)

            bin_cs.unnorm_power = binpower_unnorm

        if hasattr(self, 'cs_all'):
            cs_all = []
            for c in self.cs_all:
                cs_all.append(c.rebin(df=df, f=f, method=method))
            bin_cs.cs_all = cs_all
        if hasattr(self, 'pds1'):
            bin_cs.pds1 = self.pds1.rebin(df=df, f=f, method=method)
        if hasattr(self, 'pds2'):
            bin_cs.pds2 = self.pds2.rebin(df=df, f=f, method=method)

        try:
            bin_cs.nphots2 = self.nphots2
        except AttributeError:
            if self.type == 'powerspectrum':
                pass
            else:
                raise AttributeError(
                    'Spectrum has no attribute named nphots2.')

        bin_cs.m = np.rint(step_size * self.m)

        return bin_cs

    def _normalize_crossspectrum(self, unnorm_power, tseg):
        """
        Normalize the real part of the cross spectrum to Leahy, absolute rms^2,
        fractional rms^2 normalization, or not at all.

        Parameters
        ----------
        unnorm_power: numpy.ndarray
            The unnormalized cross spectrum.

        tseg: int
            The length of the Fourier segment, in seconds.

        Returns
        -------
        power: numpy.nd.array
            The normalized co-spectrum (real part of the cross spectrum). For
            'none' normalization, imaginary part is returned as well.
        """

        if self.err_dist == 'poisson':
            return normalize_crossspectrum(
                unnorm_power, tseg, self.n, self.nphots1, self.nphots2, self.norm,
                self.power_type)

        return normalize_crossspectrum_gauss(
            unnorm_power, np.sqrt(self.meancounts1 * self.meancounts2),
            np.sqrt(self.var1 * self.var2),
            dt=self.dt,
            N=self.n,
            norm=self.norm,
            power_type=self.power_type)

    def rebin_log(self, f=0.01):
        """
        Logarithmic rebin of the periodogram.
        The new frequency depends on the previous frequency
        modified by a factor f:

        .. math::

            d\\nu_j = d\\nu_{j-1} (1+f)

        Parameters
        ----------
        f: float, optional, default ``0.01``
            parameter that steers the frequency resolution


        Returns
        -------
        new_spec : :class:`Crossspectrum` (or one of its subclasses) object
            The newly binned cross spectrum or power spectrum.
            Note: this object will be of the same type as the object
            that called this method. For example, if this method is called
            from :class:`AveragedPowerspectrum`, it will return an object of class
        """

        binfreq, binpower, binpower_err, nsamples = \
            rebin_data_log(self.freq, self.power, f,
                           y_err=self.power_err, dx=self.df)

        # the frequency resolution
        df = np.diff(binfreq)

        # shift the lower bin edges to the middle of the bin and drop the
        # last right bin edge
        binfreq = binfreq[:-1] + df / 2

        new_spec = copy.copy(self)
        new_spec.freq = binfreq
        new_spec.power = binpower
        new_spec.power_err = binpower_err
        new_spec.m = nsamples * self.m

        if hasattr(self, 'unnorm_power'):
            _, binpower_unnorm, _, _ = \
                rebin_data_log(self.freq, self.unnorm_power, f, dx=self.df)

            new_spec.unnorm_power = binpower_unnorm

        if hasattr(self, 'pds1'):
            new_spec.pds1 = self.pds1.rebin_log(f)
        if hasattr(self, 'pds2'):
            new_spec.pds2 = self.pds2.rebin_log(f)

        if hasattr(self, 'cs_all'):
            cs_all = []
            for c in self.cs_all:
                cs_all.append(c.rebin_log(f))
            new_spec.cs_all = cs_all

        return new_spec

    def coherence(self):
        """ Compute Coherence function of the cross spectrum.

        Coherence is defined in Vaughan and Nowak, 1996 [#]_.
        It is a Fourier frequency dependent measure of the linear correlation
        between time series measured simultaneously in two energy channels.

        Returns
        -------
        coh : numpy.ndarray
            Coherence function

        References
        ----------
        .. [#] http://iopscience.iop.org/article/10.1086/310430/pdf
        """
        # this computes the averaged power spectrum, but using the
        # cross spectrum code to avoid circular imports

        return self.unnorm_power.real / (self.pds1.power.real *
                                         self.pds2.power.real)

    def _phase_lag(self):
        """Return the fourier phase lag of the cross spectrum."""
        return np.angle(self.unnorm_power)

    def time_lag(self):
        """
        Calculate the fourier time lag of the cross spectrum. The time lag is
        calculate using the center of the frequency bins.
        """
        if self.__class__ in [Crossspectrum, AveragedCrossspectrum]:
            ph_lag = self._phase_lag()

            return ph_lag / (2 * np.pi * self.freq)
        else:
            raise AttributeError("Object has no attribute named 'time_lag' !")

    def plot(self, labels=None, axis=None, title=None, marker='-', save=False,
             filename=None):
        """
        Plot the amplitude of the cross spectrum vs. the frequency using ``matplotlib``.

        Parameters
        ----------
        labels : iterable, default ``None``
            A list of tuple with ``xlabel`` and ``ylabel`` as strings.

        axis : list, tuple, string, default ``None``
            Parameter to set axis properties of the ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for the``matplotlib.pyplot.axis()`` method.

        title : str, default ``None``
            The title of the plot.

        marker : str, default '-'
            Line style and color of the plot. Line styles and colors are
            combined in a single format string, as in ``'bo'`` for blue
            circles. See ``matplotlib.pyplot.plot`` for more options.

        save : boolean, optional, default ``False``
            If ``True``, save the figure with specified filename.

        filename : str
            File name of the image to save. Depends on the boolean ``save``.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for plot()")

        plt.figure('crossspectrum')
        plt.plot(self.freq,
                 np.abs(self.power),
                 marker,
                 color='b',
                 label='Amplitude')
        plt.plot(self.freq,
                 np.abs(self.power.real),
                 marker,
                 color='r',
                 alpha=0.5,
                 label='Real Part')
        plt.plot(self.freq,
                 np.abs(self.power.imag),
                 marker,
                 color='g',
                 alpha=0.5,
                 label='Imaginary Part')

        if labels is not None:
            try:
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
            except TypeError:
                simon("``labels`` must be either a list or tuple with "
                      "x and y labels.")
                raise
            except IndexError:
                simon("``labels`` must have two labels for x and y "
                      "axes.")
                # Not raising here because in case of len(labels)==1, only
                # x-axis will be labelled.
        plt.legend(loc='best')
        if axis is not None:
            plt.axis(axis)

        if title is not None:
            plt.title(title)

        if save:
            if filename is None:
                plt.savefig('spec.png')
            else:
                plt.savefig(filename)
        else:
            plt.show(block=False)

    def classical_significances(self, threshold=1, trial_correction=False):
        """
        Compute the classical significances for the powers in the power
        spectrum, assuming an underlying noise distribution that follows a
        chi-square distributions with 2M degrees of freedom, where M is the
        number of powers averaged in each bin.

        Note that this function will *only* produce correct results when the
        following underlying assumptions are fulfilled:

        1. The power spectrum is Leahy-normalized
        2. There is no source of variability in the data other than the
           periodic signal to be determined with this method. This is important!
           If there are other sources of (aperiodic) variability in the data, this
           method will *not* produce correct results, but instead produce a large
           number of spurious false positive detections!
        3. There are no significant instrumental effects changing the
           statistical distribution of the powers (e.g. pile-up or dead time)

        By default, the method produces ``(index,p-values)`` for all powers in
        the power spectrum, where index is the numerical index of the power in
        question. If a ``threshold`` is set, then only powers with p-values
        *below* that threshold with their respective indices. If
        ``trial_correction`` is set to ``True``, then the threshold will be corrected
        for the number of trials (frequencies) in the power spectrum before
        being used.

        Parameters
        ----------
        threshold : float, optional, default ``1``
            The threshold to be used when reporting p-values of potentially
            significant powers. Must be between 0 and 1.
            Default is ``1`` (all p-values will be reported).

        trial_correction : bool, optional, default ``False``
            A Boolean flag that sets whether the ``threshold`` will be corrected
            by the number of frequencies before being applied. This decreases
            the ``threshold`` (p-values need to be lower to count as significant).
            Default is ``False`` (report all powers) though for any application
            where `threshold`` is set to something meaningful, this should also
            be applied!

        Returns
        -------
        pvals : iterable
            A list of ``(index, p-value)`` tuples for all powers that have p-values
            lower than the threshold specified in ``threshold``.

        """
        if not self.norm == "leahy":
            raise ValueError("This method only works on "
                             "Leahy-normalized power spectra!")

        if np.size(self.m) == 1:
            # calculate p-values for all powers
            # leave out zeroth power since it just encodes the number of photons!
            pv = np.array([cospectra_pvalue(power, self.m)
                           for power in self.power])
        else:
            pv = np.array([cospectra_pvalue(power, m)
                           for power, m in zip(self.power, self.m)])

        # if trial correction is used, then correct the threshold for
        # the number of powers in the power spectrum
        if trial_correction:
            threshold /= self.power.shape[0]

        # need to add 1 to the indices to make up for the fact that
        # we left out the first power above!
        indices = np.where(pv < threshold)[0]

        pvals = np.vstack([pv[indices], indices])

        return pvals


class AveragedCrossspectrum(Crossspectrum):
    """
    Make an averaged cross spectrum from a light curve by segmenting two
    light curves, Fourier-transforming each segment and then averaging the
    resulting cross spectra.

    Parameters
    ----------
    data1: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects OR :class:`stingray.EventList` object
        A light curve from which to compute the cross spectrum. In some cases, this would
        be the light curve of the wavelength/energy/frequency band of interest.

    data2: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects OR :class:`stingray.EventList` object
        A second light curve to use in the cross spectrum. In some cases, this would be
        the wavelength/energy/frequency reference band to compare the band of interest with.

    segment_size: float
        The size of each segment to average. Note that if the total
        duration of each :class:`Lightcurve` object in ``lc1`` or ``lc2`` is not an
        integer multiple of the ``segment_size``, then any fraction left-over
        at the end of the time series will be lost. Otherwise you introduce
        artifacts.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, default ``none``
        The normalization of the (real part of the) cross spectrum.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    dt : float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where data1 or data2 are of :class:EventList

    power_type: string, optional, default ``real``
         Parameter to choose among complete, real part and magnitude of
         the cross spectrum.

    silent : bool, default False
         Do not show a progress bar when generating an averaged cross spectrum.
         Useful for the batch execution of many spectra

    lc1: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects
        For backwards compatibility only. Like ``data1``, but no
        :class:`stingray.events.EventList` objects allowed

    lc2: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects
        For backwards compatibility only. Like ``data2``, but no
        :class:`stingray.events.EventList` objects allowed

    fullspec: boolean, optional, default ``False``
        If True, return the full array of frequencies, otherwise return just the
        positive frequencies.

    large_data : bool, default False
        Use only for data larger than 10**7 data points!! Uses zarr and dask for computation.

    save_all : bool, default False
        Save all intermediate PDSs used for the final average. Use with care.
        This is likely to fill up your RAM on medium-sized datasets, and to
        slow down the computation when rebinning.

    Attributes
    ----------
    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of cross spectra

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging powerspectrum). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged cross spectra

    n: int
        The number of time bins per segment of light curve

    nphots1: float
        The total number of photons in the first (interest) light curve

    nphots2: float
        The total number of photons in the second (reference) light curve

    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        They are calculated by taking the common GTI between the
        two light curves
    """

    def __init__(self, data1=None, data2=None, segment_size=None, norm='none',
                 gti=None, power_type="real", silent=False, lc1=None, lc2=None,
                 dt=None, fullspec=False, large_data=False, save_all=False):


        if lc1 is not None or lc2 is not None:
            warnings.warn("The lcN keywords are now deprecated. Use dataN "
                          "instead", DeprecationWarning)
        # for backwards compatibility
        if data1 is None:
            data1 = lc1
        if data2 is None:
            data2 = lc2

        if segment_size is None and data1 is not None:
            raise ValueError("segment_size must be specified")
        if segment_size is not None and not np.isfinite(segment_size):
            raise ValueError("segment_size must be finite!")

        if large_data and data1 is not None and data2 is not None:
            if isinstance(data1, EventList):
                input_data = 'EventList'
            elif isinstance(data1, Lightcurve):
                input_data = 'Lightcurve'
                chunks = int(np.rint(segment_size // data1.dt))
                segment_size = chunks * data1.dt
            else:
                raise ValueError(
                    f'Invalid input data type: {type(data1).__name__}')

            dir_path1 = saveData(data1, persist=False, chunks=chunks)
            dir_path2 = saveData(data2, persist=False, chunks=chunks)

            data_path1 = genDataPath(dir_path1)
            data_path2 = genDataPath(dir_path2)

            spec = createChunkedSpectra(input_data,
                                        'AveragedCrossspectrum',
                                        data_path=list(data_path1 +
                                                       data_path2),
                                        segment_size=segment_size,
                                        norm=norm,
                                        gti=gti,
                                        power_type=power_type,
                                        silent=silent,
                                        dt=dt)

            for key, val in spec.__dict__.items():
                setattr(self, key, val)

            return

        self.type = "crossspectrum"


        self.segment_size = segment_size
        self.power_type = power_type
        self.fullspec = fullspec

        self.show_progress = not silent
        self.dt = dt
        self.save_all = save_all

        if isinstance(data1, EventList):
            lengths = data1.gti[:, 1] - data1.gti[:, 0]
            good = lengths >= segment_size
            data1.gti = data1.gti[good]
            data1 = list(data1.to_lc_list(dt))

        if isinstance(data2, EventList):
            lengths = data2.gti[:, 1] - data2.gti[:, 0]
            good = lengths >= segment_size
            data2.gti = data2.gti[good]
            data2 = list(data2.to_lc_list(dt))

        Crossspectrum.__init__(self, data1, data2, norm, gti=gti,
                               power_type=power_type, dt=dt, fullspec=fullspec)

        return

    def _make_auxil_pds(self, lc1, lc2):
        """
        Helper method to create the power spectrum of both light curves
        independently.

        Parameters
        ----------
        lc1, lc2 : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.
        """
        is_event = isinstance(lc1, EventList)
        is_lc = isinstance(lc1, Lightcurve)
        is_lc_iter = isinstance(lc1, Iterator)
        is_lc_list = isinstance(lc1, Iterable) and not is_lc_iter
        # A way to say that this is actually not a power spectrum
        if self.type != "powerspectrum" and \
                (lc1 is not lc2) and (is_event or is_lc or is_lc_list):
            self.pds1 = AveragedCrossspectrum(lc1, lc1,
                                              segment_size=self.segment_size,
                                              norm='none', gti=self.gti,
                                              power_type=self.power_type,
                                              dt=self.dt, fullspec=self.fullspec,
                                              save_all=self.save_all,
                                              silent=not self.show_progress)

            self.pds2 = AveragedCrossspectrum(lc2, lc2,
                                              segment_size=self.segment_size,
                                              norm='none', gti=self.gti,
                                              power_type=self.power_type,
                                              dt=self.dt, fullspec=self.fullspec,
                                              save_all=self.save_all,
                                              silent=not self.show_progress)

    def _make_segment_spectrum(self, lc1, lc2, segment_size, silent=False):
        """
        Split the light curves into segments of size ``segment_size``, and calculate a cross spectrum for
        each.

        Parameters
        ----------
        lc1, lc2 : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.

        segment_size : ``numpy.float``
            Size of each light curve segment to use for averaging.

        Other parameters
        ----------------
        silent : bool, default False
            Suppress progress bars

        Returns
        -------
        cs_all : list of :class:`Crossspectrum`` objects
            A list of cross spectra calculated independently from each light curve segment

        nphots1_all, nphots2_all : ``numpy.ndarray` for each of ``lc1`` and ``lc2``
            Two lists containing the number of photons for all segments calculated from ``lc1`` and ``lc2``.

        """

        assert isinstance(lc1, Lightcurve)
        assert isinstance(lc2, Lightcurve)

        if lc1.tseg != lc2.tseg:
            simon("Lightcurves do not have same tseg. This means that the data"
                  "from the two channels are not completely in sync. This "
                  "might or might not be an issue. Keep an eye on it.")

        # If dt differs slightly, its propagated error must not be more than
        # 1/100th of the bin
        if not np.isclose(lc1.dt, lc2.dt, rtol=0.01 * lc1.dt / lc1.tseg):
            raise ValueError("Light curves do not have same time binning dt.")

        # In case a small difference exists, ignore it
        lc1.dt = lc2.dt

        current_gtis = cross_two_gtis(lc1.gti, lc2.gti)
        lc1.gti = lc2.gti = current_gtis
        lc1.apply_gtis()
        lc2.apply_gtis()

        if self.gti is None:
            self.gti = current_gtis
        else:
            if not np.allclose(self.gti, current_gtis):
                self.gti = np.vstack([self.gti, current_gtis])

        check_gtis(current_gtis)

        cs_all = []
        nphots1_all = []
        nphots2_all = []

        start_inds, end_inds = \
            bin_intervals_from_gtis(current_gtis, segment_size, lc1.time,
                                    dt=lc1.dt)
        simon("Errorbars on cross spectra are not thoroughly tested. "
              "Please report any inconsistencies.")

        local_show_progress = show_progress
        if not self.show_progress or silent:
            local_show_progress = lambda a: a

        for start_ind, end_ind in \
                local_show_progress(zip(start_inds, end_inds)):
            time_1 = copy.deepcopy(lc1.time[start_ind:end_ind])
            counts_1 = copy.deepcopy(lc1.counts[start_ind:end_ind])
            counts_1_err = copy.deepcopy(lc1.counts_err[start_ind:end_ind])
            time_2 = copy.deepcopy(lc2.time[start_ind:end_ind])
            counts_2 = copy.deepcopy(lc2.counts[start_ind:end_ind])
            counts_2_err = copy.deepcopy(lc2.counts_err[start_ind:end_ind])
            if np.sum(counts_1) == 0 or np.sum(counts_2) == 0:
                warnings.warn(
                    "No counts in interval {}--{}s".format(time_1[0],
                                                           time_1[-1]))
                continue

            gti1 = np.array([[time_1[0] - lc1.dt / 2,
                              time_1[-1] + lc1.dt / 2]])
            gti2 = np.array([[time_2[0] - lc2.dt / 2,
                              time_2[-1] + lc2.dt / 2]])
            lc1_seg = Lightcurve(time_1, counts_1, err=counts_1_err,
                                 err_dist=lc1.err_dist,
                                 gti=gti1,
                                 dt=lc1.dt, skip_checks=True)
            lc2_seg = Lightcurve(time_2, counts_2, err=counts_2_err,
                                 err_dist=lc2.err_dist,
                                 gti=gti2,
                                 dt=lc2.dt, skip_checks=True)
            with warnings.catch_warnings(record=True) as w:
                cs_seg = Crossspectrum(lc1_seg, lc2_seg, norm=self.norm,
                                   power_type=self.power_type, fullspec=self.fullspec)

            cs_all.append(cs_seg)
            nphots1_all.append(np.sum(lc1_seg.counts))
            nphots2_all.append(np.sum(lc2_seg.counts))

        return cs_all, nphots1_all, nphots2_all

    def _make_crossspectrum(self, lc1, lc2, fullspec=False):
        """
        Auxiliary method computing the normalized cross spectrum from two light curves.
        This includes checking for the presence of and applying Good Time Intervals, computing the
        unnormalized Fourier cross-amplitude, and then renormalizing using the required normalization.
        Also computes an uncertainty estimate on the cross spectral powers. Stingray uses the
        scipy.fft standards for the sign of the Nyquist frequency.

        Parameters
        ----------
        lc1, lc2 : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.

        fullspec: boolean, default ``False``,
            If True, return all frequencies otherwise return only positive frequencies
        """
        local_show_progress = show_progress
        if not self.show_progress:
            local_show_progress = lambda a: a

        # chop light curves into segments
        if isinstance(lc1, Lightcurve) and \
                isinstance(lc2, Lightcurve):

            if self.type == "crossspectrum":
                cs_all, nphots1_all, nphots2_all = \
                    self._make_segment_spectrum(lc1, lc2, self.segment_size)

            elif self.type == "powerspectrum":
                cs_all, nphots1_all = \
                    self._make_segment_spectrum(lc1, self.segment_size)

            else:
                raise ValueError("Type of spectrum not recognized!")

        else:
            cs_all, nphots1_all, nphots2_all = [], [], []

            for lc1_seg, lc2_seg in local_show_progress(zip(lc1, lc2)):
                if self.type == "crossspectrum":
                    cs_sep, nphots1_sep, nphots2_sep = \
                        self._make_segment_spectrum(lc1_seg, lc2_seg,
                                                    self.segment_size,
                                                    silent=True)
                    nphots2_all.append(nphots2_sep)
                elif self.type == "powerspectrum":
                    cs_sep, nphots1_sep = \
                        self._make_segment_spectrum(lc1_seg, self.segment_size,
                            silent=True)

                else:
                    raise ValueError("Type of spectrum not recognized!")
                cs_all.append(cs_sep)
                nphots1_all.append(nphots1_sep)

            cs_all = np.hstack(cs_all)
            nphots1_all = np.hstack(nphots1_all)

            if self.type == "crossspectrum":
                nphots2_all = np.hstack(nphots2_all)

        m = len(cs_all)
        nphots1 = np.mean(nphots1_all)

        power_avg = np.zeros_like(cs_all[0].power)
        power_err_avg = np.zeros_like(cs_all[0].power_err)
        unnorm_power_avg = np.zeros_like(cs_all[0].unnorm_power)
        for cs in cs_all:
            power_avg += cs.power
            unnorm_power_avg += cs.unnorm_power
            power_err_avg += (cs.power_err) ** 2

        power_avg /= float(m)
        power_err_avg = np.sqrt(power_err_avg) / m
        unnorm_power_avg /= float(m)

        self.freq = cs_all[0].freq
        self.power = power_avg
        self.unnorm_power = unnorm_power_avg
        self.m = m
        self.power_err = power_err_avg
        self.df = cs_all[0].df
        self.n = cs_all[0].n
        self.nphots1 = nphots1
        if self.save_all:
            self.cs_all = cs_all

        if self.type == "crossspectrum":
            self.nphots1 = nphots1
            nphots2 = np.mean(nphots2_all)

            self.nphots2 = nphots2

    def coherence(self):
        """Averaged Coherence function.


        Coherence is defined in Vaughan and Nowak, 1996 [#]_.
        It is a Fourier frequency dependent measure of the linear correlation
        between time series measured simultaneously in two energy channels.

        Compute an averaged Coherence function of cross spectrum by computing
        coherence function of each segment and averaging them. The return type
        is a tuple with first element as the coherence function and the second
        element as the corresponding uncertainty associated with it.

        Note : The uncertainty in coherence function is strictly valid for Gaussian \
               statistics only.

        Returns
        -------
        (coh, uncertainty) : tuple of np.ndarray
            Tuple comprising the coherence function and uncertainty.

        References
        ----------
        .. [#] http://iopscience.iop.org/article/10.1086/310430/pdf
        """
        if np.any(self.m < 50):
            simon("Number of segments used in averaging is "
                  "significantly low. The result might not follow the "
                  "expected statistical distributions.")

        # Calculate average coherence
        unnorm_power_avg = self.unnorm_power

        num = np.absolute(unnorm_power_avg) ** 2

        # The normalization was 'none'!
        unnorm_powers_avg_1 = self.pds1.power.real
        unnorm_powers_avg_2 = self.pds2.power.real

        coh = num / (unnorm_powers_avg_1 * unnorm_powers_avg_2)
        coh[~np.isfinite(coh)] = 0.0

        # Calculate uncertainty
        uncertainty = \
            (2 ** 0.5 * coh * (1 - coh)) / (np.abs(coh) * self.m ** 0.5)

        uncertainty[coh == 0] = 0.0

        return (coh, uncertainty)

    def time_lag(self):
        """Calculate time lag and uncertainty.

        Equation from Bendat & Piersol, 2011 [bendat-2011]__.

        Returns
        -------
        lag : np.ndarray
            The time lag

        lag_err : np.ndarray
            The uncertainty in the time lag
        """
        lag = super(AveragedCrossspectrum, self).time_lag()
        coh, uncert = self.coherence()

        dum = (1. - coh) / (2. * coh)

        dum[coh == 0] = 0.0

        lag_err = np.sqrt(dum / self.m) / (2 * np.pi * self.freq)

        return lag, lag_err
