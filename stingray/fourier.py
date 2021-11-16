import warnings
from collections.abc import Iterable
import numpy as np
from .gti import get_segment_events_idx, get_segment_binned_array_idx
from .utils import histogram, show_progress

try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft, fftfreq

    pyfftw.interfaces.cache.enable()
except ImportError:
    warnings.warn("pyfftw not installed. Using standard scipy fft")
    from scipy.fft import fft, fftfreq


def positive_fft_bins(N, include_zero=False):
    """See https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details

    Examples
    --------
    >>> freq = np.fft.fftfreq(10)
    >>> good = freq > 0
    >>> goodbins = positive_fft_bins(10)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    >>> freq = np.fft.fftfreq(11)
    >>> good = freq > 0
    >>> goodbins = positive_fft_bins(11)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    >>> freq = np.fft.fftfreq(10)
    >>> good = freq >= 0
    >>> goodbins = positive_fft_bins(10, include_zero=True)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    >>> freq = np.fft.fftfreq(11)
    >>> good = freq >= 0
    >>> goodbins = positive_fft_bins(11, include_zero=True)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    """
    minbin = 1
    if include_zero:
        minbin = 0

    if N % 2 == 0:
        return slice(minbin, N // 2)
    else:
        return slice(minbin, (N + 1) // 2)


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

    return power * 2 * dt / (mean ** 2 * N)


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


def normalize_crossspectrum(unnorm_power, dt, N, mean, variance=None, norm="abs", power_type="all"):
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


def bias_term(C, P1, P2, P1noise, P2noise, N, intrinsic_coherence=1.0):
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


def error_on_averaged_cross_spectrum(C, Ps, Pr, N, Psnoise, Prnoise, common_ref="False"):
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
        dRe = np.sqrt((PrPs + C.real ** 2 - C.imag ** 2) / twoN)
        dIm = np.sqrt((PrPs - C.real ** 2 + C.imag ** 2) / twoN)
        gsq = raw_coherence(C, Ps, Pr, Psnoise, Prnoise, N)
        dphi = np.sqrt((1 - gsq) / (2 * gsq ** 2 * N))
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


def _which_segment_idx_fun(counts=None):
    # Make function interface equal (counts gets ignored)
    if counts is None:
        return get_segment_events_idx
    return get_segment_binned_array_idx


def get_total_ctrate(times, gti, segment_size, counts=None):
    """Calculate the average count rate during the observation.

    This function finds the same segments that the PDS will use and
    returns the mean count rate.
    If ``counts`` is ``None``, the input times are interpreted as events.
    Otherwise, the number of events is taken from ``counts``

    Parameters
    ----------
    times : float `np.array`
        Array of times
    gti : [[gti00, gti01], [gti10, gti11], ...]
        good time intervals
    segment_size : float
        length of segments

    Other parameters
    ----------------
    counts : float `np.array`, default None
        Array of counts per bin

    Examples
    --------
    >>> times = np.sort(np.random.uniform(0, 1000, 1000))
    >>> gti = np.asarray([[0, 1000]])
    >>> counts, _ = np.histogram(times, bins=np.linspace(0, 1000, 11))
    >>> bin_times = np.arange(50, 1000, 100)
    >>> get_total_ctrate(bin_times, gti, 1000, counts=counts)
    1.0
    >>> get_total_ctrate(times, gti, 1000)
    1.0
    """
    Nph = 0
    Nintvs = 0
    func = _which_segment_idx_fun(counts)

    for _, _, idx0, idx1 in func(times, gti, segment_size):
        if counts is None:
            Nph += idx1 - idx0
        else:
            Nph += np.sum(counts[idx0:idx1])
        Nintvs += 1

    return Nph / (Nintvs * segment_size)


def get_flux_iterable_from_segments(times, gti, segment_size, N=None, counts=None, errors=None):
    """Get fluxes from different segments of the observation.

    If ``counts`` is ``None``, the input times are interpreted as events.
    At least one between ``N`` and ``counts`` needs to be specified.
    Otherwise, they are assumed uniformly binned inside each GTI, and the number
    of events per bin is taken from ``counts``

    Parameters
    ----------
    times : float `np.array`
        Array of times
    gti : [[gti00, gti01], [gti10, gti11], ...]
        good time intervals
    segment_size : float
        length of segments

    Other parameters
    ----------------
    N : int, default None
        Number of bins to divide the ``segment_size`` in
    counts : float `np.array`, default None
        Array of counts per bin

    Returns
    -------
    cts : `np.array`
        Array of counts
    err : `np.array`
        (optional) if ``errors`` is None, an array of errors in the segment

    """
    if counts is None and N is None:
        raise ValueError(
            "At least one between counts (if light curve) and N (if events) has to be set"
        )

    fun = _which_segment_idx_fun(counts)

    for s, e, idx0, idx1 in fun(times, gti, segment_size):
        if idx1 - idx0 < 2:
            yield None
            continue

        if counts is None:
            event_times = times[idx0:idx1]
            # counts, _ = np.histogram(event_times - s, bins=bins)
            cts = histogram((event_times - s).astype(float), bins=N, range=[0, segment_size])
        else:
            cts = counts[idx0:idx1]
            if errors is not None:
                cts = cts, errors[idx0:idx1]
        yield cts


def avg_pds_from_iterable(flux_iterable, dt, norm="abs", use_common_mean=True, silent=False):
    """Calculate the average periodogram from an iterable of light curves

    Parameters
    ----------
    flux_iterable : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal segments of light curve, or of light curve
        and errors. They must all be of the same length.
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
    silent : bool, default False
        Silence the progress bars

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
    local_show_progress = show_progress
    if silent:

        def local_show_progress(a):
            return a

    cross = None
    M = 0

    common_mean = 0
    common_variance = None
    for flux in local_show_progress(flux_iterable):
        if flux is None:
            continue

        variance = None
        if flux is tuple:
            flux, err = flux
            variance = np.mean(err) ** 2

        N = flux.size
        ft = fft(flux)

        nph = flux.sum()
        unnorm_power = (ft * ft.conj()).real
        common_mean += nph

        if variance is not None:
            common_variance += variance

        if cross is None:
            fgt0 = positive_fft_bins(N)
            freq = fftfreq(N, dt)[fgt0]

        unnorm_power = unnorm_power[fgt0]

        if use_common_mean:
            cs_seg = unnorm_power
        else:
            mean = nph / N

            cs_seg = normalize_crossspectrum(
                unnorm_power, dt, N, mean, norm=norm, variance=variance,
            )

        if cross is None:
            cross = cs_seg
        else:
            cross += cs_seg
        M += 1

    if cross is None:
        return None, None, None, None, None
    common_mean /= M * N
    if common_variance is not None:
        # Note: the variances we summed were means, not sums. Hence M, not M*N
        common_variance /= M
    cross /= M * N
    if use_common_mean:
        cross = normalize_crossspectrum(
            unnorm_power, dt, N, common_mean, norm=norm, variance=common_variance
        )

    return freq, cross, N, M, common_mean


def avg_cs_from_iterables(
    flux_iterable1,
    flux_iterable2,
    dt,
    norm="abs",
    use_common_mean=True,
    silent=False,
    fullspec=False,
    power_type="all",
):
    """Calculate the average cross spectrum from an iterable of light curves

    Parameters
    ----------
    flux_iterable1 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal segments of light curve, or of light curve
        and errors. They must all be of the same length.
    flux_iterable2 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Same as ``flux_iterable1``, for the reference channel
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

    local_show_progress = show_progress
    if silent:

        def local_show_progress(a):
            return a

    cross = None
    M = 0

    common_mean = 0
    common_variance1 = common_variance2 = common_variance = None

    for flux1, flux2 in local_show_progress(zip(flux_iterable1, flux_iterable2)):
        if flux1 is None or flux2 is None:
            continue

        variance1 = variance2 = None

        if flux1 is tuple:
            flux1, err1 = flux1
            variance1 = np.mean(err1) ** 2
        if flux2 is tuple:
            flux2, err2 = flux2
            variance2 = np.mean(err2) ** 2

        if variance1 is None or variance2 is None:
            variance1 = variance2 = None
        else:
            if common_variance1 is None:
                common_variance1 = variance1
                common_variance2 = variance2
            else:
                common_variance1 += variance1
                common_variance2 += variance2

        N = flux1.size
        ft1 = fft(flux1)
        ft2 = fft(flux2)

        if cross is None:
            freq = fftfreq(N, dt)
            fgt0 = positive_fft_bins(N)

        nph = np.sqrt(flux1.sum() * flux2.sum())
        unnorm_power = ft1 * ft2.conj()

        common_mean += nph

        unnorm_power = ft1 * ft2.conj()

        if not fullspec:
            unnorm_power = unnorm_power[fgt0]

        if use_common_mean:
            cs_seg = unnorm_power
        else:
            mean = nph / N

            cs_seg = normalize_crossspectrum(
                unnorm_power, dt, N, mean, norm=norm, power_type=power_type
            )

        if cross is None:
            cross = cs_seg
        else:
            cross += cs_seg
        M += 1

    if cross is None:
        return None, None, None, None, None
    common_mean /= M * N
    if common_variance1 is not None:
        # Note: the variances we summed were means, not sums. Hence M, not M*N
        common_variance1 /= M
        common_variance2 /= M
        common_variance = np.sqrt(common_variance1 * common_variance2)

    cross /= M * N
    if use_common_mean:
        cross = normalize_crossspectrum(
            unnorm_power,
            dt,
            N,
            common_mean,
            norm=norm,
            variance=common_variance,
            power_type=power_type,
        )

    if not fullspec:
        freq = freq[fgt0]
    return freq, cross, N, M, common_mean


def avg_pds_from_events(
    times,
    gti,
    segment_size,
    dt,
    norm="abs",
    use_common_mean=True,
    silent=False,
    counts=None,
    errors=None,
):
    """Calculate the average periodogram from a list of event times or a light curve.

    If the input is a light curve, the time array needs to be uniformly sampled
    inside GTIs (it can have gaps outside), and the counts need to be passed
    through the ``counts`` array.
    Otherwise, times are interpeted as photon arrival times.

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
    silent : bool, default False
        Silence the progress bars
    counts : float `np.array`, default None
        Array of counts per bin or fluxes per bin
    errors : float `np.array`, default None
        Array of errors on the counts above

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
    dt = segment_size / N

    flux_iterable = get_flux_iterable_from_segments(
        times, gti, segment_size, N, counts=counts, errors=errors
    )
    return avg_pds_from_iterable(
        flux_iterable, dt, norm=norm, use_common_mean=use_common_mean, silent=silent
    )


def avg_cs_from_events(
    times1,
    times2,
    gti,
    segment_size,
    dt,
    norm="abs",
    use_common_mean=True,
    fullspec=False,
    silent=False,
    power_type="all",
    counts1=None,
    counts2=None,
    errors1=None,
    errors2=None,
):
    """Calculate the average cross spectrum from a list of event times or a light curve.

    If the input is a light curve, the time arrays need to be uniformly sampled
    inside GTIs (they can have gaps outside), and the counts need to be passed
    through the ``counts1`` and ``counts2`` arrays.
    Otherwise, times are interpeted as photon arrival times

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
    counts1 : float `np.array`, default None
        Array of counts per bin for channel 1
    counts2 : float `np.array`, default None
        Array of counts per bin for channel 2
    errors1 : float `np.array`, default None
        Array of errors on the counts on channel 1
    errors2 : float `np.array`, default None
        Array of errors on the counts on channel 2

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

    flux_iterable1 = get_flux_iterable_from_segments(
        times1, gti, segment_size, N, counts=counts1, errors=errors1
    )
    flux_iterable2 = get_flux_iterable_from_segments(
        times2, gti, segment_size, N, counts=counts2, errors=errors2
    )
    return avg_cs_from_iterables(
        flux_iterable1,
        flux_iterable2,
        dt,
        norm=norm,
        use_common_mean=use_common_mean,
        silent=silent,
        fullspec=fullspec,
        power_type=power_type,
    )
