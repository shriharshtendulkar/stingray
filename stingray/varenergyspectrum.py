
import numpy as np
import warnings
from stingray.gti import check_separate, cross_two_gtis, create_gti_mask, check_gtis, bin_intervals_from_gtis, get_total_gti_length
from stingray.lightcurve import Lightcurve
from stingray.utils import assign_value_if_none, simon, excess_variance, show_progress,equal_count_energy_ranges
from stingray.crossspectrum import AveragedCrossspectrum
from stingray.fourier import avg_cs_from_events, avg_pds_from_events, fftfreq
from stingray.fourier import poisson_level, error_on_cross_spectrum, cross_to_covariance
from abc import ABCMeta, abstractmethod
import six


__all__ = ["VarEnergySpectrum", "RmsEnergySpectrum", "RmsSpectrum", "LagEnergySpectrum", "LagSpectrum", "ExcessVarianceSpectrum", "CovarianceSpectrum", "ComplexCovarianceSpectrum", "CountSpectrum"]


def _decode_energy_specification(energy_spec):
    """Decode the energy specification tuple.

    Parameters
    ----------
    energy_spec : iterable
        list containing the energy specification
        Must have the following structure:
            * energy_spec[0]: lower edge of (log) energy space
            * energy_spec[1]: upper edge of (log) energy space
            * energy_spec[2] +1 : energy bin edges (hence the +1)
            * {`lin` | `log`} flat deciding whether the energy space is linear
              or logarithmic

    Returns
    -------
    energies : numpy.ndarray
        An array of lower/upper bin edges for the energy array

    Examples
    --------
    >>> _decode_energy_specification([0, 2, 2, 'lin'])
    Traceback (most recent call last):
     ...
    ValueError: Energy specification must be a tuple
    >>> a = _decode_energy_specification((0, 2, 2, 'lin'))
    >>> np.allclose(a, [0, 1, 2])
    True
    >>> a = _decode_energy_specification((1, 4, 2, 'log'))
    >>> np.allclose(a, [1, 2, 4])
    True
    """
    if not isinstance(energy_spec, tuple):
        raise ValueError("Energy specification must be a tuple")

    if energy_spec[-1].lower() not in ["lin", "log"]:
        raise ValueError("Incorrect energy specification")

    log_distr = True if energy_spec[-1].lower() == "log" else False

    if log_distr:
        energies = np.logspace(np.log10(energy_spec[0]),
                               np.log10(energy_spec[1]),
                               energy_spec[2] + 1)
    else:
        energies = np.linspace(energy_spec[0], energy_spec[1],
                               energy_spec[2] + 1)

    return energies


@six.add_metaclass(ABCMeta)
class VarEnergySpectrum(object):
    """
    Base class for variability-energy spectrum.

    This class is only a base for the various variability spectra, and it's
    not to be instantiated by itself.

    Parameters
    ----------
    events : :class:`stingray.events.EventList` object
        event list

    freq_interval : ``[f0, f1]``, floats
        the frequency range over which calculating the variability quantity

    energy_spec : list or tuple ``(emin, emax, N, type)``
        if a ``list`` is specified, this is interpreted as a list of bin edges;
        if a ``tuple`` is provided, this will encode the minimum and maximum
        energies, the number of intervals, and ``lin`` or ``log``.

    Other Parameters
    ----------------
    ref_band : ``[emin, emax``], floats; default ``None``
        minimum and maximum energy of the reference band. If ``None``, the
        full band is used.

    use_pi : bool, default ``False``
        Use channel instead of energy

    events2 : :class:`stingray.events.EventList` object
        event list for the second channel, if not the same. Useful if the
        reference band has to be taken from another detector.

    Attributes
    ----------
    events1 : array-like
        list of events used to produce the spectrum

    events2 : array-like
        if the spectrum requires it, second list of events

    freq_interval : array-like
        interval of frequencies used to calculate the spectrum

    energy_intervals : ``[[e00, e01], [e10, e11], ...]``
        energy intervals used for the spectrum

    spectrum : array-like
        the spectral values, corresponding to each energy interval

    spectrum_error : array-like
        the error bars corresponding to spectrum

    """

    def __init__(self, events, freq_interval, energy_spec, ref_band=None,
                 bin_time=1, use_pi=False, segment_size=None, events2=None):

        self.events1 = events
        self.events2 = assign_value_if_none(events2, events)
        self.freq_interval = freq_interval
        self.use_pi = use_pi
        self.bin_time = bin_time
        if isinstance(energy_spec, tuple):
            energies = _decode_energy_specification(energy_spec)
        else:
            energies = np.asarray(energy_spec)

        self.energy_intervals = list(zip(energies[0: -1], energies[1:]))

        self.ref_band = np.asarray(assign_value_if_none(ref_band,
                                                        [0, np.inf]))

        if len(self.ref_band.shape) <= 1:
            self.ref_band = np.asarray([self.ref_band])

        self.segment_size = segment_size

        if len(events.time) == 0:
            simon("There are no events in your event list!" +
                  "Can't make a spectrum!")
            self.spectrum = 0
            self.spectrum_error = 0
        else:
            self.spectrum, self.spectrum_error = self._spectrum_function()

    def _get_events_from_energy_range(self, events, erange):
        energies = events.energy
        mask = (energies >= erange[0]) & (energies < erange[1])

        return events.apply_mask(mask)

    def _get_times_from_energy_range(self, events, erange):
        energies = events.energy
        mask = (energies >= erange[0]) & (energies < erange[1])
        return events.time[mask]

    def _decide_ref_intervals(self, channel_band, ref_band):
        """
        Ensures that the ``channel_band`` (i.e. the band of interest) is
        not contained within the ``ref_band`` (i.e. the reference band)

        Parameters
        ----------
        channel_band : iterable of type ``[elow, ehigh]``
            The lower/upper limits of the energies to be contained in the band
            of interest

        ref_band : iterable
            The lower/upper limits of the energies in the reference band

        Returns
        -------
        ref_intervals : iterable
            The channels that are both in the reference band in not in the
            bands of interest
        """
        channel_band = np.asarray(channel_band)
        ref_band = np.asarray(ref_band)
        if len(ref_band.shape) <= 1:
            ref_band = np.asarray([ref_band])
        if check_separate(ref_band, [channel_band]):
            return np.asarray(ref_band)
        not_channel_band = [[0, channel_band[0]],
                            [channel_band[1], np.max([np.max(ref_band),
                                                      channel_band[1] + 1])]]

        return cross_two_gtis(ref_band, not_channel_band)

    def _construct_lightcurves(self, channel_band, tstart=None, tstop=None,
                               exclude=True, only_base=False):
        """
        Construct light curves from event data, for each band of interest.

        Parameters
        ----------
        channel_band : iterable of type ``[elow, ehigh]``
            The lower/upper limits of the energies to be contained in the band
            of interest

        tstart : float, optional, default ``None``
            A common start time (if start of observation is different from
            the first recorded event)

        tstop : float, optional, default ``None``
            A common stop time (if start of observation is different from
            the first recorded event)

        exclude : bool, optional, default ``True``
            if ``True``, exclude the band of interest from the reference band

        only_base : bool, optional, default ``False``
            if ``True``, only return the light curve of the channel of interest, not
            that of the reference band

        Returns
        -------
        base_lc : :class:`Lightcurve` object
            The light curve of the channels of interest

        ref_lc : :class:`Lightcurve` object (only returned if ``only_base`` is ``False``)
            The reference light curve for comparison with ``base_lc``
        """
        if self.use_pi:
            energies1 = self.events1.pi
            energies2 = self.events2.pi
        else:
            energies2 = self.events2.energy
            energies1 = self.events1.energy

        gti = cross_two_gtis(self.events1.gti, self.events2.gti)

        tstart = assign_value_if_none(tstart, gti[0, 0])
        tstop = assign_value_if_none(tstop, gti[-1, -1])

        good = (energies1 >= channel_band[0]) & (energies1 < channel_band[1])
        base_lc = Lightcurve.make_lightcurve(self.events1.time[good],
                                             self.bin_time,
                                             tstart=tstart,
                                             tseg=tstop - tstart,
                                             gti=gti,
                                             mjdref=self.events1.mjdref)

        if only_base:
            return base_lc

        if exclude:
            ref_intervals = self._decide_ref_intervals(channel_band,
                                                       self.ref_band)
        else:
            ref_intervals = self.ref_band

        ref_lc = Lightcurve(base_lc.time, np.zeros_like(base_lc.counts),
                            gti=base_lc.gti, mjdref=base_lc.mjdref,
                            dt=base_lc.dt,
                            err_dist=base_lc.err_dist, skip_checks=True)

        for i in ref_intervals:
            good = (energies2 >= i[0]) & (energies2 < i[1])
            new_lc = Lightcurve.make_lightcurve(self.events2.time[good],
                                                self.bin_time,
                                                tstart=tstart,
                                                tseg=tstop - tstart,
                                                gti=base_lc.gti,
                                                mjdref=self.events2.mjdref)
            ref_lc = ref_lc + new_lc

        ref_lc.err_dist = base_lc.err_dist
        return base_lc, ref_lc

    @abstractmethod
    def _spectrum_function(self):
        pass


class RmsSpectrum(VarEnergySpectrum):
    """Calculate the rms-Energy spectrum.

    For each energy interval, calculate the power density spectrum in
    fractional r.m.s. normalization. If ``events2`` is specified, the cospectrum
    is used instead of the PDS.

    Parameters
    ----------
    events : :class:`stingray.events.EventList` object
        event list

    freq_interval : ``[f0, f1]``, list of float
        the frequency range over which calculating the variability quantity

    energy_spec : list or tuple ``(emin, emax, N, type)``
        if a ``list`` is specified, this is interpreted as a list of bin edges;
        if a ``tuple`` is provided, this will encode the minimum and maximum
        energies, the number of intervals, and ``lin`` or ``log``.

    Other Parameters
    ----------------
    ref_band : ``[emin, emax]``, float; default ``None``
        minimum and maximum energy of the reference band. If ``None``, the
        full band is used.

    use_pi : bool, default ``False``
        Use channel instead of energy

    events2 : :class:`stingray.events.EventList` object
        event list for the second channel, if not the same. Useful if the
        reference band has to be taken from another detector.

    Attributes
    ----------
    events1 : array-like
        list of events used to produce the spectrum

    events2 : array-like
        if the spectrum requires it, second list of events

    freq_interval : array-like
        interval of frequencies used to calculate the spectrum

    energy_intervals : ``[[e00, e01], [e10, e11], ...]``
        energy intervals used for the spectrum

    spectrum : array-like
        the spectral values, corresponding to each energy interval

    spectrum_error : array-like
        the errorbars corresponding to spectrum
    """

    def _spectrum_function(self):
        events1 = self.events1
        events2 = self.events2
        common_gti = events1.gti
        if events2 is None or events2 is events1:
            events2 = events1
            same_events = True
        else:
            common_gti = cross_two_gtis(events1.gti, events2.gti)
            same_events = False

        spec = np.zeros(len(self.energy_intervals))
        spec_err = np.zeros_like(spec)

        N = np.rint(self.segment_size / self.bin_time)
        delta_nu = 1 / self.segment_size
        freq = fftfreq(int(N), self.bin_time)
        freq = freq[freq > 0]
        good = (freq >= self.freq_interval[0]) & (freq < self.freq_interval[1])

        Mave = np.count_nonzero(good)
        delta_nu_after_mean = delta_nu * Mave

        f = (self.freq_interval[0] + self.freq_interval[1]) / 2

        for i, eint in enumerate(show_progress(self.energy_intervals)):
            sub_events = self._get_times_from_energy_range(events1, eint)
            countrate_sub = sub_events.size / get_total_gti_length(common_gti,
                                                                   minlen=self.segment_size)
            Psnoise = poisson_level(countrate_sub, norm="abs")

            if not same_events:
                ref_events = self._get_times_from_energy_range(events2, eint)
                countrate_ref = ref_events.size / get_total_gti_length(
                    common_gti,
                    minlen=self.segment_size)
                Prnoise = poisson_level(countrate_ref, norm="abs")
                _, cross, N, M, mean = avg_cs_from_events(
                    sub_events, ref_events, common_gti, self.segment_size,
                    self.bin_time, silent=True, norm="abs")
                Pmean = np.mean(cross[good])
                Pnoise = 0
                rmsnoise = np.sqrt(delta_nu_after_mean * np.sqrt(Psnoise*Prnoise))
                meanrate = mean / self.bin_time
            else:
                _, Ps, N, M, mean = avg_pds_from_events(sub_events, common_gti,
                                                  self.segment_size, self.bin_time,
                                                  silent=True, norm="abs")
                Pmean = np.mean(Ps[good])
                Pnoise = Psnoise
                rmsnoise = np.sqrt(delta_nu_after_mean * Pnoise)

                meanrate = mean / self.bin_time

            # Assume coherence 1
            # rmsnoise = np.sqrt(delta_nu_after_mean * Pnoise)

            rms = np.sqrt(np.abs(Pmean - Pnoise) * delta_nu_after_mean)

            num = rms**4 + rmsnoise**4 + 2 * rms * rmsnoise
            den = 4 * M * Mave * rms**2

            rms_err = np.sqrt(num / den)

            spec[i] = rms / meanrate
            spec_err[i] = rms_err / meanrate

        return spec, spec_err


RmsEnergySpectrum = RmsSpectrum


class ExcessVarianceSpectrum(VarEnergySpectrum):
    """Calculate the Excess Variance spectrum.

    For each energy interval, calculate the excess variance in the specified
    frequency range.

    Parameters
    ----------
    events : :class:`stingray.events.EventList` object
        event list

    freq_interval : ``[f0, f1]``, list of float
        the frequency range over which calculating the variability quantity

    energy_spec : list or tuple ``(emin, emax, N, type)``
        if a list is specified, this is interpreted as a list of bin edges;
        if a tuple is provided, this will encode the minimum and maximum
        energies, the number of intervals, and ``lin`` or ``log``.

    Other Parameters
    ----------------
    ref_band : ``[emin, emax]``, floats; default ``None``
        minimum and maximum energy of the reference band. If ``None``, the
        full band is used.

    use_pi : bool, default ``False``
        Use channel instead of energy

    Attributes
    ----------
    events1 : array-like
        list of events used to produce the spectrum

    freq_interval : array-like
        interval of frequencies used to calculate the spectrum

    energy_intervals : ``[[e00, e01], [e10, e11], ...]``
        energy intervals used for the spectrum

    spectrum : array-like
        the spectral values, corresponding to each energy interval

    spectrum_error : array-like
        the errorbars corresponding to spectrum
    """

    def __init__(self, events, freq_interval, energy_spec,
                 bin_time=1, use_pi=False, segment_size=None,
                 normalization='fvar'):

        self.normalization = normalization
        accepted_normalizations = ['fvar', 'none']
        if normalization not in accepted_normalizations:
            raise ValueError('The normalization of excess variance must be '
                             'one of {}'.format(accepted_normalizations))

        VarEnergySpectrum.__init__(self, events, freq_interval, energy_spec,
                                   bin_time=bin_time, use_pi=use_pi,
                                   segment_size=segment_size)

    def _spectrum_function(self):
        spec = np.zeros(len(self.energy_intervals))
        spec_err = np.zeros_like(spec)
        for i, eint in enumerate(self.energy_intervals):
            lc = self._construct_lightcurves(eint, exclude=False,
                                             only_base=True)

            spec[i], spec_err[i] = excess_variance(lc, self.normalization)

        return spec, spec_err


class CountSpectrum(VarEnergySpectrum):
    """Calculate the covariance spectrum.

    For each energy interval, calculate the covariance between two bands.
    If ``events2`` is specified, the energy bands are chosen from this second
    event list, while the reference band from ``events``.

    Parameters
    ----------
    events : :class:`stingray.events.EventList` object
        event list

    freq_interval : ``[f0, f1]``, list of float
        the frequency range over which calculating the variability quantity

    energy_spec : list or tuple ``(emin, emax, N, type)``
        if a ``list`` is specified, this is interpreted as a list of bin edges;
        if a ``tuple`` is provided, this will encode the minimum and maximum
        energies, the number of intervals, and ``lin`` or ``log``.

    Other Parameters
    ----------------
    ref_band : ``[emin, emax]``, float; default ``None``
        minimum and maximum energy of the reference band. If ``None``, the
        full band is used.

    use_pi : bool, default ``False``
        Use channel instead of energy

    events2 : :class:`stingray.events.EventList` object
        event list for the second channel, if not the same. Useful if the
        reference band has to be taken from another detector.

    Attributes
    ----------
    events1 : array-like
        list of events used to produce the spectrum

    events2 : array-like
        if the spectrum requires it, second list of events

    freq_interval : array-like
        interval of frequencies used to calculate the spectrum

    energy_intervals : ``[[e00, e01], [e10, e11], ...]``
        energy intervals used for the spectrum

    spectrum : array-like
        the spectral values, corresponding to each energy interval

    spectrum_error : array-like
        the errorbars corresponding to spectrum
    """

    def __init__(self, events, energy_spec, ref_band=None,
                 bin_time=1, use_pi=False, segment_size=None, events2=None):

        VarEnergySpectrum.__init__(self, events, None, energy_spec,
                                   bin_time=bin_time, use_pi=use_pi,
                                   ref_band=ref_band,
                                   segment_size=segment_size, events2=events2)

    def _spectrum_function(self):
        events = self.events1
        spec = np.zeros(len(self.energy_intervals))
        spec_err = np.zeros_like(spec)

        for i, eint in show_progress(enumerate(self.energy_intervals)):
            sub_events = self._get_times_from_energy_range(events, eint)

            sp = sub_events.size
            spec[i] = sp
            spec_err[i] = np.sqrt(sp)

        return spec, spec_err


class LagSpectrum(VarEnergySpectrum):
    """Calculate the covariance spectrum.

    For each energy interval, calculate the covariance between two bands.
    If ``events2`` is specified, the energy bands are chosen from this second
    event list, while the reference band from ``events``.

    Parameters
    ----------
    events : :class:`stingray.events.EventList` object
        event list

    freq_interval : ``[f0, f1]``, list of float
        the frequency range over which calculating the variability quantity

    energy_spec : list or tuple ``(emin, emax, N, type)``
        if a ``list`` is specified, this is interpreted as a list of bin edges;
        if a ``tuple`` is provided, this will encode the minimum and maximum
        energies, the number of intervals, and ``lin`` or ``log``.

    Other Parameters
    ----------------
    ref_band : ``[emin, emax]``, float; default ``None``
        minimum and maximum energy of the reference band. If ``None``, the
        full band is used.

    use_pi : bool, default ``False``
        Use channel instead of energy

    events2 : :class:`stingray.events.EventList` object
        event list for the second channel, if not the same. Useful if the
        reference band has to be taken from another detector.

    Attributes
    ----------
    events1 : array-like
        list of events used to produce the spectrum

    events2 : array-like
        if the spectrum requires it, second list of events

    freq_interval : array-like
        interval of frequencies used to calculate the spectrum

    energy_intervals : ``[[e00, e01], [e10, e11], ...]``
        energy intervals used for the spectrum

    spectrum : array-like
        the spectral values, corresponding to each energy interval

    spectrum_error : array-like
        the errorbars corresponding to spectrum
    """
    # events, freq_interval, energy_spec, ref_band = None
    def __init__(self, events, freq_interval, energy_spec, ref_band=None,
                 bin_time=1, use_pi=False, segment_size=None, events2=None):

        VarEnergySpectrum.__init__(self, events, freq_interval,
                                   energy_spec=energy_spec,
                                   bin_time=bin_time, use_pi=use_pi,
                                   ref_band=ref_band,
                                   segment_size=segment_size, events2=events2)

    def _spectrum_function(self):
        events1 = self.events1
        events2 = self.events2
        common_gti = events1.gti
        if events2 is None or events2 is events1:
            events2 = events1
            same_events = True
        else:
            common_gti = cross_two_gtis(events1.gti, events2.gti)
            same_events = False

        spec = np.zeros(len(self.energy_intervals)) + np.nan
        spec_err = np.zeros_like(spec) + np.nan

        ref_events = self._get_times_from_energy_range(events2,
                                                       self.ref_band[0])
        countrate_ref = ref_events.size / get_total_gti_length(common_gti,
                                                               minlen=self.segment_size)
        Prnoise = poisson_level(countrate_ref, norm="abs")
        freq, Pr, N, M, mean = avg_pds_from_events(ref_events, common_gti,
                                             self.segment_size, self.bin_time,
                                             silent=True, norm="abs")

        good = (freq >= self.freq_interval[0]) & (freq < self.freq_interval[1])
        Prmean = np.mean(Pr[good])
        Mave = np.count_nonzero(good)
        Mtot = Mave * M

        f = (self.freq_interval[0] + self.freq_interval[1]) / 2
        import matplotlib.pyplot as plt
        for i, eint in enumerate(show_progress(self.energy_intervals)):
            sub_events = self._get_times_from_energy_range(events1, eint)
            countrate_sub = sub_events.size / get_total_gti_length(common_gti,
                                                                   minlen=self.segment_size)
            Psnoise = poisson_level(countrate_sub, norm="abs")

            _, cross, _, _, _ = avg_cs_from_events(sub_events, ref_events,
                                                common_gti, self.segment_size,
                                                self.bin_time, silent=True,
                                                norm="abs")
            _, Ps, _, _, _ = avg_pds_from_events(sub_events, common_gti,
                                              self.segment_size, self.bin_time,
                                              silent=True, norm="abs")

            if cross is None or Ps is None:
                continue

            Cmean = np.mean(cross[good])
            Psmean = np.mean(Ps[good])

            common_ref = same_events and len(
                cross_two_gtis([eint], self.ref_band)) > 0

            _, _, phi_e, _ = error_on_cross_spectrum(
                Cmean, Psmean, Prmean, Mtot, Psnoise, Prnoise,
                common_ref=common_ref)

            lag = np.mean((np.angle(cross) / (2 * np.pi * freq))[good])
            lag_e = phi_e / (2 * np.pi * f)
            spec[i] = lag
            spec_err[i] = lag_e

        return spec, spec_err


LagEnergySpectrum = LagSpectrum


class ComplexCovarianceSpectrum(VarEnergySpectrum):
    """Calculate the covariance spectrum.

    For each energy interval, calculate the covariance between two bands.
    If ``events2`` is specified, the energy bands are chosen from this second
    event list, while the reference band from ``events``.

    Parameters
    ----------
    events : :class:`stingray.events.EventList` object
        event list

    freq_interval : ``[f0, f1]``, list of float
        the frequency range over which calculating the variability quantity

    energy_spec : list or tuple ``(emin, emax, N, type)``
        if a ``list`` is specified, this is interpreted as a list of bin edges;
        if a ``tuple`` is provided, this will encode the minimum and maximum
        energies, the number of intervals, and ``lin`` or ``log``.

    Other Parameters
    ----------------
    ref_band : ``[emin, emax]``, float; default ``None``
        minimum and maximum energy of the reference band. If ``None``, the
        full band is used.

    use_pi : bool, default ``False``
        Use channel instead of energy

    events2 : :class:`stingray.events.EventList` object
        event list for the second channel, if not the same. Useful if the
        reference band has to be taken from another detector.

    Attributes
    ----------
    events1 : array-like
        list of events used to produce the spectrum

    events2 : array-like
        if the spectrum requires it, second list of events

    freq_interval : array-like
        interval of frequencies used to calculate the spectrum

    energy_intervals : ``[[e00, e01], [e10, e11], ...]``
        energy intervals used for the spectrum

    spectrum : array-like
        the spectral values, corresponding to each energy interval

    spectrum_error : array-like
        the errorbars corresponding to spectrum
    """

    def __init__(self, events, energy_spec, ref_band=None,
                 freq_interval=[0, 1],
                 bin_time=1, use_pi=False, segment_size=None, events2=None,
                 norm="abs",
                 return_complex=False):

        self.norm = norm
        self.return_complex = return_complex
        VarEnergySpectrum.__init__(self, events, freq_interval=freq_interval,
                                   energy_spec=energy_spec,
                                   bin_time=bin_time, use_pi=use_pi,
                                   ref_band=ref_band,
                                   segment_size=segment_size, events2=events2)

    def _spectrum_function(self):
        events1 = self.events1
        common_gti = events1.gti
        if self.events2 is None or self.events2 is self.events1:
            events2 = self.events1
            same_events = True
        else:
            common_gti = cross_two_gtis(events1.gti, events2.gti)
            same_events = False

        if self.return_complex:
            dtype = complex
        else:
            dtype = float
        spec = np.zeros(len(self.energy_intervals), dtype=dtype) + np.nan
        spec_err = np.zeros_like(spec) + np.nan
        df = 1 / self.segment_size

        ref_events = self._get_times_from_energy_range(events2,
                                                       self.ref_band[0])
        countrate_ref = ref_events.size / get_total_gti_length(common_gti,
                                                               minlen=self.segment_size)
        Prnoise = poisson_level(countrate_ref, norm=self.norm)

        freq, Pr, N, M, mean = avg_pds_from_events(ref_events, common_gti,
                                             self.segment_size, self.bin_time,
                                             silent=True, norm=self.norm)

        good = (freq >= self.freq_interval[0]) & (freq < self.freq_interval[1])
        Mave = np.count_nonzero(good)
        Prmean = np.mean(Pr[good])

        Mtot = M * Mave
        delta_nu = Mave * df

        for i, eint in enumerate(show_progress(self.energy_intervals)):
            sub_events = self._get_times_from_energy_range(events1, eint)
            countrate_sub = sub_events.size / get_total_gti_length(common_gti,
                                                                   minlen=self.segment_size)
            Psnoise = poisson_level(countrate_sub, norm=self.norm)

            _, cross, _, _, _ = avg_cs_from_events(sub_events, ref_events,
                                                common_gti, self.segment_size,
                                                self.bin_time, silent=True,
                                                norm=self.norm)
            _, Ps, _, _, _ = avg_pds_from_events(sub_events, common_gti,
                                              self.segment_size, self.bin_time,
                                              silent=True, norm=self.norm)
            if cross is None or Ps is None:
                continue

            common_ref = same_events and len(
                cross_two_gtis([eint], self.ref_band)) > 0
            if common_ref:
                noise_term = Psnoise
                if self.norm == "frac":
                    noise_term *= countrate_sub / countrate_ref
                cross -= Psnoise
            Cmean = np.mean(cross[good])
            Cmean_real = np.abs(Cmean)

            if not self.return_complex:
                Cmean = Cmean_real

            Psmean = np.mean(Ps[good])
            _, _, _, Ce = error_on_cross_spectrum(
                Cmean_real, Psmean, Prmean, Mtot, Psnoise, Prnoise,
                common_ref=common_ref)

            cov, cov_e = cross_to_covariance(np.asarray([Cmean, Ce]), Prmean,
                                             Prnoise, delta_nu)

            spec[i] = cov
            spec_err[i] = cov_e

        return spec, spec_err


class CovarianceSpectrum(ComplexCovarianceSpectrum):
    """Calculate the covariance spectrum.

    For each energy interval, calculate the covariance between two bands.
    If ``events2`` is specified, the energy bands are chosen from this second
    event list, while the reference band from ``events``.

    Parameters
    ----------
    events : :class:`stingray.events.EventList` object
        event list

    freq_interval : ``[f0, f1]``, list of float
        the frequency range over which calculating the variability quantity

    energy_spec : list or tuple ``(emin, emax, N, type)``
        if a ``list`` is specified, this is interpreted as a list of bin edges;
        if a ``tuple`` is provided, this will encode the minimum and maximum
        energies, the number of intervals, and ``lin`` or ``log``.

    Other Parameters
    ----------------
    ref_band : ``[emin, emax]``, float; default ``None``
        minimum and maximum energy of the reference band. If ``None``, the
        full band is used.

    use_pi : bool, default ``False``
        Use channel instead of energy

    events2 : :class:`stingray.events.EventList` object
        event list for the second channel, if not the same. Useful if the
        reference band has to be taken from another detector.

    Attributes
    ----------
    events1 : array-like
        list of events used to produce the spectrum

    events2 : array-like
        if the spectrum requires it, second list of events

    freq_interval : array-like
        interval of frequencies used to calculate the spectrum

    energy_intervals : ``[[e00, e01], [e10, e11], ...]``
        energy intervals used for the spectrum

    spectrum : array-like
        the spectral values, corresponding to each energy interval

    spectrum_error : array-like
        the errorbars corresponding to spectrum
    """

    def __init__(self, events, energy_spec, ref_band=None,
                 freq_interval=[0, 1],
                 bin_time=1, use_pi=False, segment_size=None, events2=None,
                 norm="abs"):
        ComplexCovarianceSpectrum.__init__(self, events,
                                           freq_interval=freq_interval,
                                           energy_spec=energy_spec,
                                           bin_time=bin_time, use_pi=use_pi,
                                           norm=norm,
                                           ref_band=ref_band,
                                           return_complex=False,
                                           segment_size=segment_size,
                                           events2=events2)
