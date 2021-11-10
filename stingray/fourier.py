import warnings
import glob
from collections.abc import Iterable
import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np
from .gti import create_gti_from_condition, gti_border_bins
from .gti import get_segment_events_idx, time_intervals_from_gtis, cross_two_gtis

from .utils import histogram, show_progress

try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft, fftfreq
    pyfftw.interfaces.cache.enable()
except ImportError:
    warnings.warn("pyfftw not installed. Using standard scipy fft")
    from scipy.fft import fft, fftfreq



def poisson_level(meanrate=0, norm="abs"):
    """Poisson (white)-noise level in a periodogram of pure counting noise.

    Other Parameters
    ----------
    meanrate : float, default 0
        Mean count rate in counts/s
    norm : str, default "abs"
        Normalization of the periodogram. One of ["abs", "frac", "leahy"]

    Examples
    --------
    >>> poisson_level(norm="leahy")
    2.0
    >>> poisson_level(meanrate=10., norm="abs")
    20.0
    >>> poisson_level(meanrate=10., norm="frac")
    0.2
    >>> poisson_level(meanrate=10., norm="asdfwrqfasdh3r")
    Traceback (most recent call last):
    ...
    ValueError: Unknown value for norm: asdfwrqfasdh3r...
    """
    if norm == "abs":
        return 2 * meanrate
    if norm == "frac":
        return 2 / meanrate
    if norm == "leahy":
        return 2.0
    raise ValueError(f"Unknown value for norm: {norm}")


def normalize_frac(power, dt, N, mean):
    """Fractional rms normalization, from the variance of the lc.

    Examples
    --------
    >>> mean = var = 1000000
    >>> N = 1000000
    >>> dt = 0.2
    >>> meanrate = mean / dt
    >>> lc = np.random.poisson(mean, N)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_frac(pds, dt, lc.size, mean)
    >>> np.isclose(pdsnorm[1:N//2].mean(), poisson_level(meanrate=meanrate,norm="frac"), rtol=0.01)
    True
    """
#     (mean * N) / (mean /dt) = N * dt
#     It's Leahy / meanrate;
#     Nph = mean * N
#     meanrate = mean / dt
#     norm = 2 / (Nph * meanrate) = 2 * dt / (mean**2 * N)

    return power * 2 * dt / (mean**2 * N)


def normalize_abs(power, dt, N):
    """Absolute rms normalization, from the variance of the lc.

    Examples
    --------
    >>> mean = var = 100000
    >>> N = 1000000
    >>> dt = 0.2
    >>> meanrate = mean / dt
    >>> lc = np.random.poisson(mean, N)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_abs(pds, dt, lc.size)
    >>> np.isclose(pdsnorm[1:N//2].mean(), poisson_level(meanrate=meanrate, norm="abs"), rtol=0.01)
    True
    """
#     It's frac * meanrate**2; Leahy / meanrate * meanrate**2
#     Nph = mean * N
#     meanrate = mean / dt
#     norm = 2 / (Nph * meanrate) * meanrate**2 = 2 * dt / (mean**2 * N) * mean**2 / dt**2

    return power * 2 / N / dt


def normalize_leahy_from_variance(power, variance, N):
    """Leahy+83 normalization, from the variance of the lc.

    Examples
    --------
    >>> mean = var = 100000
    >>> N = 1000000
    >>> lc = np.random.poisson(mean, N)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_leahy_from_variance(pds, var, lc.size)
    >>> np.isclose(pdsnorm[0], 2 * np.sum(lc), rtol=0.01)
    True
    >>> np.isclose(pdsnorm[1:N//2].mean(), poisson_level(norm="leahy"), rtol=0.01)
    True
    """
    return power * 2 / (variance * N)


def normalize_leahy_poisson(power, Nph):
    """Leahy+83 normalization, from the variance of the lc.

    Examples
    --------
    >>> mean = var = 100000
    >>> N = 1000000
    >>> lc = np.random.poisson(mean, N)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_leahy_poisson(pds, np.sum(lc))
    >>> np.isclose(pdsnorm[0], 2 * np.sum(lc), rtol=0.01)
    True
    >>> np.isclose(pdsnorm[1:N//2].mean(), poisson_level(norm="leahy"), rtol=0.01)
    True
    """
    return power * 2 / Nph


def normalize_crossspectrum(unnorm_power, dt, N, mean, variance=None,
                            norm="abs", power_type="all"):
    """Wrapper around all the normalize_NORM methods."""
    if norm == "leahy" and variance is not None:
        pds = normalize_leahy_from_variance(unnorm_power, variance, N)
    elif norm == "leahy":
        pds = normalize_leahy_poisson(unnorm_power, N * mean)
    elif norm == "frac":
        pds = normalize_frac(unnorm_power, dt, N, mean)
    elif norm == "abs":
        pds = normalize_abs(unnorm_power, dt, N)
    elif norm == "none":
        pds = unnorm_power
    else:
        raise ValueError("Unknown value for the norm")

    if power_type == "real":
        pds = pds.real
    elif power_type == "abs":
        pds = np.abs(pds)

    return pds


def bias_term(C, P1, P2, P1noise, P2noise, N, intrinsic_coherence=1.):
    """Bias term from Ingram 2019.

    Parameters
    ----------
    C : complex `np.array`
        cross spectrum
    P1 : float `np.array`
        sub-band periodogram
    P2 : float `np.array`
        reference-band periodogram
    P1noise : float
        Poisson noise level of the sub-band periodogram
    P2noise : float
        Poisson noise level of the reference-band periodogram
    N : int
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    intrinsic_coherence : float, default 1
        If known, the intrinsic coherence.
    """
    bsq = P1 * P2 - intrinsic_coherence * (P1 - P1noise) * (P2 - P2noise)
    return bsq / N


def raw_coherence(C, P1, P2, P1noise, P2noise, N, intrinsic_coherence=1):
    """Raw coherence from Ingram 2019.

    Parameters
    ----------
    C : complex `np.array`
        cross spectrum
    P1 : float `np.array`
        sub-band periodogram
    P2 : float `np.array`
        reference-band periodogram
    P1noise : float
        Poisson noise level of the sub-band periodogram
    P2noise : float
        Poisson noise level of the reference-band periodogram
    N : int
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    intrinsic_coherence : float, default 1
        If known, the intrinsic coherence.
    """
    bsq = bias_term(C, P1, P2, P1noise, P2noise, N, intrinsic_coherence=intrinsic_coherence)
    num = (C * C.conj()).real - bsq
    if isinstance(num, Iterable):
        num[num < 0] = (C * C.conj()).real[num < 0]
    elif num < 0:
        num = (C * C.conj()).real
    den = P1 * P2
    return num / den


def estimate_intrinsic_coherence(C, P1, P2, P1noise, P2noise, N):
    """Estimate intrinsic coherence

    Use the iterative procedure from sec. 5 of Ingram 2019

    Parameters
    ----------
    C : complex `np.array`
        cross spectrum
    P1 : float `np.array`
        sub-band periodogram
    P2 : float `np.array`
        reference-band periodogram
    P1noise : float
        Poisson noise level of the sub-band periodogram
    P2noise : float
        Poisson noise level of the reference-band periodogram
    N : int
        number of intervals that have been averaged to obtain the input spectra

    """
    new_coherence = np.ones_like(P1)
    old_coherence = np.zeros_like(P1)
    count = 0
    while not np.allclose(new_coherence, old_coherence, atol=0.01) and count < 40:
        # TODO: make it only iterate over the places at low coherence
        old_coherence = new_coherence
        bsq = bias_term(C, P1, P2, P1noise, P2noise, N, intrinsic_coherence=new_coherence)
#         old_coherence = new_coherence
        den = (P1 - P1noise) * (P2 - P2noise)
        num = (C * C.conj()).real - bsq
        num[num < 0] = (C * C.conj()).real[num < 0]
        new_coherence = num / den
        count += 1

    return new_coherence


def error_on_cross_spectrum(C, Ps, Pr, N, Psnoise, Prnoise, common_ref="False"):
    """Error on cross spectral quantities, From Ingram 2019.

    Parameters
    ----------
    C : complex `np.array`
        cross spectrum
    Ps : float `np.array`
        sub-band periodogram
    Pr : float `np.array`
        reference-band periodogram
    Psnoise : float
        Poisson noise level of the sub-band periodogram
    Prnoise : float
        Poisson noise level of the reference-band periodogram
    N : int
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    common_ref : bool, default False
        Are data in the sub-band also included in the reference band?

    Returns
    -------
    dRe : float `np.array`
        Error on the real part of the cross spectrum
    dIm : float `np.array`
        Error on the imaginary part of the cross spectrum
    dphi : float `np.array`
        Error on the angle (or phase lag)
    dG : float `np.array`
        Error on the modulus of the cross spectrum

    """
    twoN = 2 * N
    if common_ref:
        Gsq = (C * C.conj()).real
        bsq = bias_term(C, Ps, Pr, Psnoise, Prnoise, N)
        frac = (Gsq - bsq) / (Pr - Prnoise)
        PoN = Pr / twoN

        # Eq. 18
        dRe = dIm = dG = np.sqrt(PoN * (Ps - frac))
        # Eq. 19
        dphi = np.sqrt(PoN * (Ps / (Gsq - bsq) - 1 / (Pr - Prnoise)))
    else:
        PrPs = Pr * Ps
        dRe = np.sqrt((PrPs + C.real**2 - C.imag**2) / twoN)
        dIm = np.sqrt((PrPs - C.real**2 + C.imag**2) / twoN)
        gsq = raw_coherence(C, Ps, Pr, Psnoise, Prnoise, N)
        dphi = np.sqrt((1 - gsq) / (2 * gsq**2 * N))
        dG = np.sqrt(PrPs / N)

    return dRe, dIm, dphi, dG


def cross_to_covariance(C, Pr, Prnoise, delta_nu):
    """Convert a cross spectrum into a covariance.
     Parameters
    ----------
    C : complex `np.array`
        cross spectrum
    Pr : float `np.array`
        reference-band periodogram
    Prnoise : float
        Poisson noise level of the reference-band periodogram
    delta_nu : float or `np.array`
        spectral resolution. Can be a float, or an array if the spectral
        resolution is not constant throughout the periodograms

    """
    return C * np.sqrt(delta_nu / (Pr - Prnoise))


def get_total_ctrate(times, gti, segment_size):
    """Calculate the average count rate during the observation.

    Parameters
    ----------
    times : float `np.array`
        Array of times
    gti : [[gti00, gti01], [gti10, gti11], ...]
        good time intervals
    segment_size : float
        length of segments

    """
    Nph = 0
    Nintvs = 0
    for s, e, idx0, idx1 in get_segment_events_idx(times, gti, segment_size):
        Nph += idx1 - idx0
        Nintvs += 1

    return Nph / (Nintvs * segment_size)


def get_fts_from_event_segments(times, gti, segment_size, N):
    """Get Fourier transforms from different segments of the observation.

    Parameters
    ----------
    times : float `np.array`
        Array of times
    gti : [[gti00, gti01], [gti10, gti11], ...]
        good time intervals
    segment_size : float
        length of segments
    N : int
        Number of bins to divide the ``segment_size`` in

    Returns
    -------
    ft : complex `np.array`
        the Fourier transform
    N : int
        the number of photons in the segment.
    """
    for s, e, idx0, idx1 in get_segment_events_idx(times, gti, segment_size):
        if idx1 - idx0 < 2:
            yield None, None
            continue
        event_times = times[idx0:idx1]

        # counts, _ = np.histogram(event_times - s, bins=bins)
        counts = histogram((event_times - s).astype(float), bins=N,
                           range=[0, segment_size])
        ft = fft(counts)
        yield ft, idx1 - idx0


def avg_pds_from_events(times, gti, segment_size, dt,
                        norm="abs", use_common_mean=True,
                        fullspec=False, silent=False, power_type="all"):
    """Calculate the average periodogram from a list of event times.

    Parameters
    ----------
    times : float `np.array`
        Array of times
    gti : [[gti00, gti01], [gti10, gti11], ...]
        good time intervals
    segment_size : float
        length of segments
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
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

    Returns
    -------
    freq : `np.array`
        The periodogram frequencies
    pds : `np.array`
        The normalized periodogram powers
    N : int
        the number of bins in the light curves used in each segment
    M : int
        the number of averaged periodograms
    mean : float
        the mean counts per bin
    """
    N = np.rint(segment_size / dt).astype(int)
    # adjust dt
    dt = segment_size / N

    freq = fftfreq(N, dt)
    fgt0 = freq > 0

    cross = None
    M = 0
    local_show_progress = show_progress
    if silent:
        local_show_progress = lambda a: a

    if use_common_mean:
        ctrate = get_total_ctrate(times, gti, segment_size)
        mean = ctrate * dt

    for ft, nph in local_show_progress(get_fts_from_event_segments(times, gti, segment_size, N)):

        if ft is None:
            continue

        unnorm_power = (ft * ft.conj()).real

        if not fullspec:
            unnorm_power = unnorm_power[fgt0]

        if not use_common_mean:
            mean = nph / N

        cs_seg = normalize_crossspectrum(unnorm_power, dt, N, mean, norm=norm,
                                         power_type=power_type)

        if cross is None:
            cross = cs_seg
        else:
            cross += cs_seg
        M += 1

    if cross is None:
        return None, None, None, None, None
    cross /= M
    if not fullspec:
        freq = freq[fgt0]
    return freq, cross, N, M, mean


def avg_cs_from_events(times1, times2, gti,
                       segment_size, dt, norm="abs",
                       use_common_mean=False, fullspec=False, common_ref=False,
                       silent=False, power_type="all"):
    """Calculate the average cross spectrum from a list of event times.

    Parameters
    ----------
    times1 : float `np.array`
        Array of times in the sub-band
    times2 : float `np.array`
        Array of times in the reference band
    gti : [[gti00, gti01], [gti10, gti11], ...]
        common good time intervals
    segment_size : float
        length of segments
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
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

    Returns
    -------
    freq : `np.array`
        The periodogram frequencies
    pds : `np.array`
        The normalized periodogram powers
    N : int
        the number of bins in the light curves used in each segment
    M : int
        the number of averaged periodograms
    """
    N = np.rint(segment_size / dt).astype(int)
    # adjust dt
    dt = segment_size / N

    freq = fftfreq(N, dt)
    fgt0 = freq > 0
    # gti = cross_two_gtis(events1.gti, events2.gti)
    # events1.gti = events2.gti = gti

    cross = None
    M = 0
    local_show_progress = show_progress
    if silent:
        local_show_progress = lambda a: a

    if use_common_mean:
        ctrate1 = get_total_ctrate(times1, gti, segment_size)
        ctrate2 = get_total_ctrate(times2, gti, segment_size)
        ctrate = np.sqrt(ctrate1 * ctrate2)
        mean = ctrate * dt

    for (ft1, nph1), (ft2, nph2) in local_show_progress(zip(
        get_fts_from_event_segments(times1, gti, segment_size, N),
        get_fts_from_event_segments(times2, gti, segment_size, N)
    )):

        if ft1 is None or ft2 is None:
            continue

        unnorm_power = ft1 * ft2.conj()

        if not fullspec:
            unnorm_power = unnorm_power[fgt0]

        if not use_common_mean:
            nph = np.sqrt(nph1 * nph2)
            mean = nph / N

            cs_seg = normalize_crossspectrum(unnorm_power, dt, N, mean, norm=norm,
                                             power_type=power_type)
        else:
            cs_seg = unnorm_power

        if cross is None:
            cross = cs_seg
        else:
            cross += cs_seg
        M += 1
    if cross is None:
        return None, None, None, None, None
    cross /= M
    if use_common_mean:
        cross = normalize_crossspectrum(cross, dt, N, mean, norm=norm,
                                        power_type=power_type)
    if not fullspec:
        freq = freq[fgt0]
    return freq, cross, N, M, mean
