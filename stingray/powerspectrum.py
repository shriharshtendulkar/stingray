import copy
import warnings

import numpy as np
import scipy
import scipy.optimize
import scipy.stats

import stingray.utils as utils
from stingray.crossspectrum import AveragedCrossspectrum, Crossspectrum
from stingray.gti import bin_intervals_from_gtis, check_gtis
from stingray.largememory import createChunkedSpectra, saveData
from stingray.stats import pds_probability, amplitude_upper_limit
from stingray.utils import genDataPath

from .events import EventList
from .gti import cross_two_gtis
from .lightcurve import Lightcurve

try:
    from tqdm import tqdm as show_progress
except ImportError:
    def show_progress(a, **kwargs):
        return a

__all__ = ["Powerspectrum", "AveragedPowerspectrum", "DynamicalPowerspectrum"]


class Powerspectrum(Crossspectrum):
    """
    Make a :class:`Powerspectrum` (also called periodogram) from a (binned) light curve.
    Periodograms can be normalized by either Leahy normalization, fractional rms
    normalizaation, absolute rms normalization, or not at all.

    You can also make an empty :class:`Powerspectrum` object to populate with your
    own fourier-transformed data (this can sometimes be useful when making
    binned power spectra).

    Parameters
    ----------
    data: :class:`stingray.Lightcurve` object, optional, default ``None``
        The light curve data to be Fourier-transformed.

    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }, optional, default ``frac``
        The normaliation of the power spectrum to be used. Options are
        ``leahy``, ``frac``, ``abs`` and ``none``, default is ``frac``.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    Attributes
    ----------
    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }
        the normalization of the power spectrun

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of normalized squared absolute values of Fourier
        amplitudes

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging power spectrum). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged powers in each bin

    n: int
        The number of data points in the light curve

    nphots: float
        The total number of photons in the light curve

    """
    def __init__(self, data=None, norm="frac", gti=None,
                 dt=None, lc=None):
        if lc is not None:
            warnings.warn("The lc keyword is now deprecated. Use data "
                          "instead", DeprecationWarning)
        if data is None:
            data = lc

        Crossspectrum.__init__(self, data1=data, data2=data, norm=norm, gti=gti,
                               dt=dt)
        self.nphots = self.nphots1
        self.dt = dt

    def rebin(self, df=None, f=None, method="mean"):
        """
        Rebin the power spectrum.

        Parameters
        ----------
        df: float
            The new frequency resolution

        Other Parameters
        ----------------
        f: float
            the rebin factor. If specified, it substitutes ``df`` with ``f*self.df``

        Returns
        -------
        bin_cs = :class:`Powerspectrum` object
            The newly binned power spectrum.
        """
        bin_ps = Crossspectrum.rebin(self, df=df, f=f, method=method)
        bin_ps.nphots = bin_ps.nphots1

        return bin_ps

    def compute_rms(self, min_freq, max_freq, white_noise_offset=0.):
        """
        Compute the fractional rms amplitude in the power spectrum
        between two frequencies.

        Parameters
        ----------
        min_freq: float
            The lower frequency bound for the calculation

        max_freq: float
            The upper frequency bound for the calculation

        Other parameters
        ----------------
        white_noise_offset : float, default 0
            This is the white noise level, in Leahy normalization. In the ideal
            case, this is 2. Dead time and other instrumental effects can alter
            it. The user can fit the white noise level outside this function
            and it will get subtracted from powers here.

        Returns
        -------
        rms: float
            The fractional rms amplitude contained between ``min_freq`` and
            ``max_freq``

        rms_err: float
            The error on the fractional rms amplitude

        """
        minind = self.freq.searchsorted(min_freq)
        maxind = self.freq.searchsorted(max_freq)
        powers = self.power[minind:maxind]
        nphots = self.nphots

        if self.norm.lower() == 'leahy':
            powers_leahy = powers.copy()
        elif self.norm.lower() == "frac":
            powers_leahy = \
                self.unnorm_power[minind:maxind].real * 2 / nphots
        else:
            raise TypeError("Normalization not recognized!")

        rms = np.sqrt(np.sum(powers_leahy - white_noise_offset) / nphots)
        rms_err = self._rms_error(powers_leahy)

        return rms, rms_err

    def _rms_error(self, powers):
        """
        Compute the error on the fractional rms amplitude using error
        propagation.
        Note: this uses the actual measured powers, which is not
        strictly correct. We should be using the underlying power spectrum,
        but in the absence of an estimate of that, this will have to do.

        .. math::

           r = \sqrt{P}

        .. math::

           \delta r = \\frac{1}{2 * \sqrt{P}} \delta P

        Parameters
        ----------
        powers: iterable
            The list of powers used to compute the fractional rms amplitude.

        Returns
        -------
        delta_rms: float
            The error on the fractional rms amplitude
        """
        nphots = self.nphots
        p_err = scipy.stats.chi2(2.0 * self.m).var() * powers / self.m / nphots

        rms = np.sum(powers) / nphots
        pow = np.sqrt(rms)

        drms_dp = 1 / (2 * pow)

        sq_sum_err = np.sqrt(np.sum(p_err**2))
        delta_rms = sq_sum_err * drms_dp
        return delta_rms

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

        if trial_correction:
            ntrial = self.power.shape[0]
        else:
            ntrial = 1

        if np.size(self.m) == 1:
            # calculate p-values for all powers
            # leave out zeroth power since it just encodes the number of photons!
            pv = pds_probability(self.power, n_summed_spectra=self.m,
                                 ntrial=ntrial)
        else:
            pv = np.array([pds_probability(power, n_summed_spectra=m,
                                           ntrial=ntrial)
                           for power, m in zip(self.power, self.m)])

        # need to add 1 to the indices to make up for the fact that
        # we left out the first power above!
        indices = np.where(pv < threshold)[0]

        pvals = np.vstack([pv[indices], indices])

        return pvals

    def modulation_upper_limit(self, fmin=None, fmax=None, c=0.95):
        """Upper limit on a sinusoidal modulation.

        To understand the meaning of this amplitude: if the modulation is described by:

        ..math:: p = \overline{p} (1 + a * \sin(x))

        this function returns a.

        If it is a sum of sinusoidal harmonics instead
        ..math:: p = \overline{p} (1 + \sum_l a_l * \sin(lx))
        a is equivalent to :math:`\sqrt(\sum_l a_l^2)`.

        See `stingray.stats.power_upper_limit`, `stingray.stats.amplitude_upper_limit`
        for more information.

        Parameters
        ----------
        fmin: float
            The minimum frequency to search (defaults to the first nonzero bin)

        fmax: float
            The maximum frequency to search (defaults to the Nyquist frequency)

        Other Parameters
        ----------------
        c: float
            The confidence value for the upper limit (e.g. 0.95 = 95%)

        Results
        -------
        a: float
            The modulation amplitude that could produce P>pmeas with 1 - c probability

        Examples
        --------
        >>> pds = Powerspectrum()
        >>> pds.norm = "leahy"
        >>> pds.freq = np.arange(0., 5.)
        >>> # Note: this pds has 40 as maximum value between 2 and 5 Hz
        >>> pds.power = np.array([100000, 1, 1, 40, 1])
        >>> pds.m = 1
        >>> pds.nphots = 30000
        >>> pds.modulation_upper_limit(fmin=2, fmax=5, c=0.99)
        0.1016...
        >>> pds.norm = "frac"
        >>> pds.modulation_upper_limit(2, 5, 0.99)
        Traceback (most recent call last):
         ...
        ValueError: Modulation upper limit is only available in the Leahy...
        """

        # TODO: add case with norm different from Leahy
        if self.norm != 'leahy':
            raise ValueError("Modulation upper limit is only available in the Leahy normalization")

        freq = self.freq
        fnyq = np.max(freq)
        power = self.power
        freq_mask = freq > 0
        if fmin is not None or fmax is not None:
            if fmin is not None:
                freq_mask[freq < fmin] = 0
            if fmax is not None:
                freq_mask[freq > fmax] = 0
        freq = freq[freq_mask]
        power = power[freq_mask]

        maximum_val = np.argmax(power)
        nyq_ratio = freq[maximum_val] / fnyq

        # I multiply by M because the formulas from Vaughan+94 treat summed powers, while here
        # we have averaged powers.
        return amplitude_upper_limit(
            power[maximum_val] * self.m, self.nphots, n=self.m, c=c, nyq_ratio=nyq_ratio, fft_corr=True)


class AveragedPowerspectrum(AveragedCrossspectrum, Powerspectrum):
    """
    Make an averaged periodogram from a light curve by segmenting the light
    curve, Fourier-transforming each segment and then averaging the
    resulting periodograms.

    Parameters
    ----------
    data: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects OR :class:`stingray.EventList` object
        The light curve data to be Fourier-transformed.

    segment_size: float
        The size of each segment to average. Note that if the total
        duration of each :class:`Lightcurve` object in lc is not an integer multiple
        of the ``segment_size``, then any fraction left-over at the end of the
        time series will be lost.

    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }, optional, default ``frac``
        The normaliation of the periodogram to be used.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    silent : bool, default False
         Do not show a progress bar when generating an averaged cross spectrum.
         Useful for the batch execution of many spectra

    dt: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where data is of :class:EventList

    large_data : bool, default False
        Use only for data larger than 10**7 data points!! Uses zarr and dask for computation.

    save_all : bool, default False
        Save all intermediate PDSs used for the final average. Use with care.
        This is likely to fill up your RAM on medium-sized datasets, and to
        slow down the computation when rebinning.

    Attributes
    ----------
    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }
        the normalization of the periodogram

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of normalized squared absolute values of Fourier
        amplitudes

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging powerspectrum). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged periodograms

    n: int
        The number of data points in the light curve

    nphots: float
        The total number of photons in the light curve

    """
    def __init__(self, data=None, segment_size=None, norm="frac", gti=None,
                 silent=False, dt=None, lc=None, large_data=False,
                 save_all=False):

        if lc is not None:
            warnings.warn("The lc keyword is now deprecated. Use data "
                          "instead", DeprecationWarning)
        # Backwards compatibility: user might have supplied lc instead
        if data is None:
            data = lc

        if segment_size is None and data is not None:
            raise ValueError("segment_size must be specified")
        if segment_size is not None and not np.isfinite(segment_size):
            raise ValueError("segment_size must be finite!")

        if large_data and data is not None:
            chunks = None

            if isinstance(data, EventList):
                input_data = 'EventList'
            elif isinstance(data, Lightcurve):
                input_data = 'Lightcurve'
                chunks = int(np.rint(segment_size // data.dt))
                segment_size = chunks * data.dt
            else:
                raise ValueError(
                    f'Invalid input data type: {type(data).__name__}')

            dir_path = saveData(data, persist=False, chunks=chunks)

            data_path = genDataPath(dir_path)
            spec = createChunkedSpectra(input_data,
                                        'AveragedPowerspectrum',
                                        data_path=data_path,
                                        segment_size=segment_size,
                                        norm=norm,
                                        gti=gti,
                                        power_type=None,
                                        silent=silent,
                                        dt=dt)
            for key, val in spec.__dict__.items():
                setattr(self, key, val)

            return

        self.type = "powerspectrum"
        self.dt = dt
        self.save_all = save_all

        if isinstance(data, EventList):
            lengths = data.gti[:, 1] - data.gti[:, 0]
            good = lengths >= segment_size
            data.gti = data.gti[good]

        self.segment_size = segment_size
        self.show_progress = not silent
        Powerspectrum.__init__(self, data, norm, gti=gti, dt=dt)

        return

    @staticmethod
    def from_time_array(times, dt, segment_size, gti, norm='none',
                        power_type="real", silent=False,
                        fullspec=False, use_common_mean=True):
        """Calculate AveragedPowerspectrum from an array of event times.

        Parameters
        ----------
        times : `np.array`
            Event arrival times
        dt : float
            The time resolution of the intermediate light curves
            (sets the Nyquist frequency)
        segment_size : float
            The length, in seconds, of the light curve segments that will be averaged
        gti : [[gti0, gti1], ...]
            Good Time intervals

        Other parameters
        ----------------
        norm : str, default "abs"
            The normalization of the periodogram. "abs" is absolute rms, "frac" is
            fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
            unnormalized periodogram
        use_common_mean : bool, default True
            The mean of the light curve can be estimated in each interval, or on
            the full light curve. This gives different results (Alston+2013).
            Here we assume the mean is calculated on the full light curve, but
            the user can set ``use_common_mean`` to False to calculate it on a
            per-segment basis.
        fullspec : bool, default False
            Return the full periodogram, including negative frequencies
        silent : bool, default False
            Silence the progress bars
        power_type : str, default 'all'
            If 'all', give complex powers. If 'abs', the absolute value; if 'real',
            the real part
        """
        from .fourier import avg_pds_from_events

        freq, power, N, M, mean = avg_pds_from_events(
            times, gti, segment_size, dt,
            norm=norm, use_common_mean=use_common_mean,
            silent=silent)

        cs = AveragedPowerspectrum()
        cs.freq = freq
        cs.power = power
        cs.m = M
        cs.n = N
        cs.df = 1 / segment_size
        cs.nphots = mean * N
        cs.fullspec = fullspec
        cs.segment_size = segment_size

        return cs

    @staticmethod
    def from_events(events, dt, segment_size, norm='none',
                    power_type="real", silent=False,
                    fullspec=False, use_common_mean=True):
        """Calculate AveragedPowerspectrum from an event list

        Parameters
        ----------
        events : `stingray.EventList`
            Event list to be analyzed
        dt : float
            The time resolution of the intermediate light curves
            (sets the Nyquist frequency)
        segment_size : float
            The length, in seconds, of the light curve segments that will be averaged

        Other parameters
        ----------------
        norm : str, default "abs"
            The normalization of the periodogram. "abs" is absolute rms, "frac" is
            fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
            unnormalized periodogram
        use_common_mean : bool, default True
            The mean of the light curve can be estimated in each interval, or on
            the full light curve. This gives different results (Alston+2013).
            Here we assume the mean is calculated on the full light curve, but
            the user can set ``use_common_mean`` to False to calculate it on a
            per-segment basis.
        fullspec : bool, default False
            Return the full periodogram, including negative frequencies
        silent : bool, default False
            Silence the progress bars
        power_type : str, default 'all'
            If 'all', give complex powers. If 'abs', the absolute value; if 'real',
            the real part
        """

        return AveragedPowerspectrum.from_time_array(
            events.time, dt, segment_size, events.gti, norm=norm,
            power_type=power_type, silent=silent,
            fullspec=fullspec, use_common_mean=use_common_mean
        )

    @staticmethod
    def from_lightcurve(lc, segment_size, norm='none',
                        power_type="real", silent=False,
                        fullspec=False, use_common_mean=True):
        """Calculate AveragedPowerspectrum from a light curve

        Parameters
        ----------
        events : `stingray.Lightcurve`
            Light curve to be analyzed
        dt : float
            The time resolution of the intermediate light curves
            (sets the Nyquist frequency)
        segment_size : float
            The length, in seconds, of the light curve segments that will be averaged

        Other parameters
        ----------------
        norm : str, default "abs"
            The normalization of the periodogram. "abs" is absolute rms, "frac" is
            fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
            unnormalized periodogram
        use_common_mean : bool, default True
            The mean of the light curve can be estimated in each interval, or on
            the full light curve. This gives different results (Alston+2013).
            Here we assume the mean is calculated on the full light curve, but
            the user can set ``use_common_mean`` to False to calculate it on a
            per-segment basis.
        fullspec : bool, default False
            Return the full periodogram, including negative frequencies
        silent : bool, default False
            Silence the progress bars
        power_type : str, default 'all'
            If 'all', give complex powers. If 'abs', the absolute value; if 'real',
            the real part
        """
        from .fourier import avg_pds_from_events

        freq, power, N, M, mean = avg_pds_from_events(
            lc.time, lc.gti, segment_size, lc.dt,
            norm=norm, use_common_mean=use_common_mean,
            silent=silent,
            counts=lc.counts)

        cs = AveragedPowerspectrum()
        cs.freq = freq
        cs.power = power
        cs.m = M
        cs.n = N
        cs.df = 1 / segment_size
        cs.nphots = mean * N
        cs.fullspec = fullspec
        cs.segment_size = segment_size
        return cs

    def _make_segment_spectrum(self, lc, segment_size, silent=False):
        """
        Split the light curves into segments of size ``segment_size``, and
        calculate a power spectrum for each.

        Parameters
        ----------
        lc  : :class:`stingray.Lightcurve` objects\
            The input light curve

        segment_size : ``numpy.float``
            Size of each light curve segment to use for averaging.

        Other parameters
        ----------------
        silent : bool, default False
            Suppress progress bars

        Returns
        -------
        power_all : list of :class:`Powerspectrum` objects
            A list of power spectra calculated independently from each light curve segment

        nphots_all : ``numpy.ndarray``
            List containing the number of photons for all segments calculated from ``lc``
        """
        if not isinstance(lc, Lightcurve):
            raise TypeError("lc must be a Lightcurve object")

        current_gtis = lc.gti

        if self.gti is None:
            self.gti = lc.gti
        else:
            if not np.allclose(lc.gti, self.gti):
                self.gti = np.vstack([self.gti, lc.gti])

        check_gtis(self.gti)

        start_inds, end_inds = \
            bin_intervals_from_gtis(current_gtis, segment_size, lc.time, dt=lc.dt)

        power_all = []
        nphots_all = []

        local_show_progress = show_progress
        if not self.show_progress or silent:
            local_show_progress = lambda a: a

        for start_ind, end_ind in \
                local_show_progress(zip(start_inds, end_inds)):
            time = lc.time[start_ind:end_ind]
            counts = lc.counts[start_ind:end_ind]
            counts_err = lc.counts_err[start_ind: end_ind]

            if np.sum(counts) == 0:
                warnings.warn(
                    "No counts in interval {}--{}s".format(time[0], time[-1]))
                continue

            lc_seg = Lightcurve(time, counts, err=counts_err,
                                err_dist=lc.err_dist.lower(),
                                skip_checks=True, dt=lc.dt)

            power_seg = Powerspectrum(lc_seg, norm=self.norm)
            power_all.append(power_seg)
            nphots_all.append(np.sum(lc_seg.counts))

        return power_all, nphots_all


class DynamicalPowerspectrum(AveragedPowerspectrum):
    """
    Create a dynamical power spectrum, also often called a *spectrogram*.

    This class will divide a :class:`Lightcurve` object into segments of
    length ``segment_size``, create a power spectrum for each segment and store
    all powers in a matrix as a function of both time (using the mid-point of each
    segment) and frequency.

    This is often used to trace changes in period of a (quasi-)periodic signal over
    time.

    Parameters
    ----------
    lc : :class:`stingray.Lightcurve` object
        The time series of which the Dynamical powerspectrum is
        to be calculated.

    segment_size : float, default 1
         Length of the segment of light curve, default value is 1 (in whatever units
         the ``time`` array in the :class:`Lightcurve`` object uses).

    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }, optional, default ``frac``
        The normaliation of the periodogram to be used.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    Attributes
    ----------
    segment_size: float
        The size of each segment to average. Note that if the total
        duration of each Lightcurve object in lc is not an integer multiple
        of the ``segment_size``, then any fraction left-over at the end of the
        time series will be lost.

    dyn_ps : np.ndarray
        The matrix of normalized squared absolute values of Fourier
        amplitudes. The axis are given by the ``freq``
        and ``time`` attributes

    norm: {``leahy`` | ``frac`` | ``abs`` | ``none``}
        the normalization of the periodogram

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    df: float
        The frequency resolution

    dt: float
        The time resolution
    """

    def __init__(self, lc, segment_size, norm="frac", gti=None, dt=None):
        if segment_size < 2 * lc.dt:
            raise ValueError("Length of the segment is too short to form a "
                             "light curve!")
        elif segment_size > lc.tseg:
            raise ValueError("Length of the segment is too long to create "
                             "any segments of the light curve!")
        AveragedPowerspectrum.__init__(self, data=lc,
                                       segment_size=segment_size, norm=norm,
                                       gti=gti, dt=dt)
        self._make_matrix(lc)

    def _make_matrix(self, lc):
        """
        Create a matrix of powers for each time step and each frequency step.

        Time increases with row index, frequency with column index.

        Parameters
        ----------
        lc : :class:`Lightcurve` object
            The :class:`Lightcurve` object from which to generate the dynamical
            power spectrum
        """
        ps_all, _ = AveragedPowerspectrum._make_segment_spectrum(
            self, lc, self.segment_size)
        self.dyn_ps = np.array([ps.power for ps in ps_all]).T

        self.freq = ps_all[0].freq
        current_gti = lc.gti
        if self.gti is not None:
            current_gti = cross_two_gtis(self.gti, current_gti)

        start_inds, end_inds = \
            bin_intervals_from_gtis(current_gti, self.segment_size, lc.time,
                                    dt=lc.dt)

        tstart = lc.time[start_inds]
        tend = lc.time[end_inds]

        self.time = tstart + 0.5 * (tend - tstart)

        # Assign length of lightcurve as time resolution if only one value
        if len(self.time) > 1:
            self.dt = self.time[1] - self.time[0]
        else:
            self.dt = lc.n

        # Assign biggest freq. resolution if only one value
        if len(self.freq) > 1:
            self.df = self.freq[1] - self.freq[0]
        else:
            self.df = 1 / lc.n

    def rebin_frequency(self, df_new, method="sum"):
        """
        Rebin the Dynamic Power Spectrum to a new frequency resolution. Rebinning is
        an in-place operation, i.e. will replace the existing ``dyn_ps`` attribute.

        While the new resolution need not be an integer multiple of the
        previous frequency resolution, be aware that if it is not, the last
        bin will be cut off by the fraction left over by the integer division.

        Parameters
        ----------
        df_new: float
            The new frequency resolution of the Dynamical Power Spectrum.
            Must be larger than the frequency resolution of the old Dynamical
            Power Spectrum!

        method: {``sum`` | ``mean`` | ``average``}, optional, default ``sum``
            This keyword argument sets whether the counts in the new bins
            should be summed or averaged.
        """
        new_dynspec_object = copy.deepcopy(self)
        dynspec_new = []
        for data in self.dyn_ps.T:
            freq_new, bin_counts, bin_err, _ = \
                utils.rebin_data(self.freq, data, dx_new=df_new,
                                 method=method)
            dynspec_new.append(bin_counts)

        new_dynspec_object.freq = freq_new
        new_dynspec_object.dyn_ps = np.array(dynspec_new).T
        new_dynspec_object.df = df_new
        return new_dynspec_object

    def trace_maximum(self, min_freq=None, max_freq=None):
        """
        Return the indices of the maximum powers in each segment :class:`Powerspectrum`
        between specified frequencies.

        Parameters
        ----------
        min_freq: float, default ``None``
            The lower frequency bound.

        max_freq: float, default ``None``
            The upper frequency bound.

        Returns
        -------
        max_positions : np.array
            The array of indices of the maximum power in each segment having
            frequency between ``min_freq`` and ``max_freq``.
        """
        if min_freq is None:
            min_freq = np.min(self.freq)
        if max_freq is None:
            max_freq = np.max(self.freq)

        max_positions = []
        for ps in self.dyn_ps.T:
            indices = np.logical_and(self.freq <= max_freq,
                                     min_freq <= self.freq)
            max_power = np.max(ps[indices])
            max_positions.append(np.where(ps == max_power)[0][0])

        return np.array(max_positions)

    def rebin_time(self, dt_new, method='sum'):

        """
        Rebin the Dynamic Power Spectrum to a new time resolution.
        While the new resolution need not be an integer multiple of the
        previous time resolution, be aware that if it is not, the last bin
        will be cut off by the fraction left over by the integer division.

        Parameters
        ----------
        dt_new: float
            The new time resolution of the Dynamical Power Spectrum.
            Must be larger than the time resolution of the old Dynamical Power
            Spectrum!

        method: {"sum" | "mean" | "average"}, optional, default "sum"
            This keyword argument sets whether the counts in the new bins
            should be summed or averaged.


        Returns
        -------
        time_new: numpy.ndarray
            Time axis with new rebinned time resolution.

        dynspec_new: numpy.ndarray
            New rebinned Dynamical Power Spectrum.
        """
        if dt_new < self.dt:
            raise ValueError("New time resolution must be larger than "
                             "old time resolution!")

        new_dynspec_object = copy.deepcopy(self)

        dynspec_new = []
        for data in self.dyn_ps:
            time_new, bin_counts, bin_err, _ = \
                utils.rebin_data(self.time, data, dt_new,
                                 method=method)
            dynspec_new.append(bin_counts)

        new_dynspec_object.time = time_new
        new_dynspec_object.dyn_ps = np.array(dynspec_new)
        new_dynspec_object.dt = dt_new
        return new_dynspec_object
