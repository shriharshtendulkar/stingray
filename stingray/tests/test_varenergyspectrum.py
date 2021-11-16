from multiprocessing import Event
import os
import numpy as np
from stingray.events import EventList
from stingray.varenergyspectrum import VarEnergySpectrum
from stingray.varenergyspectrum import ComplexCovarianceSpectrum, CovarianceSpectrum
from stingray.varenergyspectrum import RmsSpectrum, RmsEnergySpectrum, CountSpectrum
from stingray.varenergyspectrum import LagSpectrum, LagEnergySpectrum
from stingray.varenergyspectrum import ExcessVarianceSpectrum
from stingray.lightcurve import Lightcurve

from astropy.tests.helper import pytest

np.random.seed(20150907)
curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

class DummyVarEnergy(VarEnergySpectrum):
    def _spectrum_function(self):
        return None, None


class TestExcVarEnergySpectrum(object):
    @classmethod
    def setup_class(cls):
        from ..simulator import Simulator

        simulator = Simulator(0.1, 10000, rms=0.2, mean=200)
        test_lc = simulator.simulate(1)
        cls.test_ev1, cls.test_ev2 = EventList(), EventList()
        cls.test_ev1.simulate_times(test_lc)
        cls.test_ev1.energy = np.random.uniform(0.3, 12, len(cls.test_ev1.time))

    def test_allocate(self):
        _ = ExcessVarianceSpectrum(
            self.test_ev1, [0.0, 100], (0.3, 12, 5, "lin"), bin_time=1, segment_size=100
        )

    def test_invalid_norm(self):
        with pytest.raises(ValueError):
            _ = ExcessVarianceSpectrum(
                self.test_ev1, [0.0, 100], (0.3, 12, 5, "lin"), bin_time=1, segment_size=100,
                normalization="asdfghjkl"
            )


class TestVarEnergySpectrum(object):
    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 100.0
        nphot = 1000
        alltimes = np.random.uniform(tstart, tend, nphot)
        alltimes.sort()
        cls.events = EventList(
            alltimes, energy=np.random.uniform(0.3, 12, nphot), gti=[[tstart, tend]]
        )
        cls.vespec = DummyVarEnergy(
            cls.events, [0.0, 10000], (0.5, 5, 10, "lin"), [0.3, 10], bin_time=0.1
        )
        cls.vespeclog = DummyVarEnergy(cls.events, [0.0, 10000], (0.5, 5, 10, "log"), [0.3, 10])

    def test_no_spectrum_func_raises(self):
        with pytest.raises(TypeError):
            ref_int = VarEnergySpectrum(self.events, [0.0, 10000], (0.5, 5, 10, "log"), [0.3, 10])

    def test_intervals_overlapping(self):
        ref_int = self.vespec._decide_ref_intervals([0.5, 6], [0.3, 10])
        np.testing.assert_allclose(ref_int, [[0.3, 0.5], [6, 10]])
        ref_int = self.vespec._decide_ref_intervals([0.5, 11], [0.3, 10])
        np.testing.assert_allclose(ref_int, [[0.3, 0.5]])

    def test_intervals_non_overlapping(self):
        ref_int = self.vespec._decide_ref_intervals([6, 11], [0.3, 5])
        np.testing.assert_allclose(ref_int, [[0.3, 5]])

    def test_ref_band_none(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54], energy=[0, 0, 0, 0, 1, 1], gti=[[0, 0.65]]
        )
        vespec = DummyVarEnergy(events, [0.0, 10000], (0, 1, 2, "lin"), bin_time=0.1)
        assert np.allclose(vespec.ref_band, np.array([[0, np.inf]]))

    def test_energy_spec_wrong_list_not_tuple(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54], energy=[0, 0, 0, 0, 1, 1], gti=[[0, 0.65]]
        )
        # Test using a list instead of tuple
        # with pytest.raises(ValueError):
        vespec = DummyVarEnergy(events, [0.0, 10000], [0, 1, 2, "lin"], bin_time=0.1)

    def test_energy_spec_wrong_str(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54], energy=[0, 0, 0, 0, 1, 1], gti=[[0, 0.65]]
        )
        # Test using a list instead of tuple
        with pytest.raises(ValueError):
            vespec = DummyVarEnergy(events, [0.0, 10000], (0, 1, 2, "xxx"), bin_time=0.1)

    def test_construct_lightcurves(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54], energy=[0, 0, 0, 0, 1, 1], gti=[[0, 0.65]]
        )
        vespec = DummyVarEnergy(events, [0.0, 10000], (0, 1, 2, "lin"), [0.5, 1.1], bin_time=0.1)
        base_lc, ref_lc = vespec._construct_lightcurves([0, 0.5], tstart=0, tstop=0.65)
        np.testing.assert_allclose(base_lc.counts, [1, 0, 2, 1, 0, 0])
        np.testing.assert_allclose(ref_lc.counts, [0, 0, 0, 0, 1, 1])

    def test_construct_lightcurves_no_exclude(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54], energy=[0, 0, 0, 0, 1, 1], gti=[[0, 0.65]]
        )

        vespec = DummyVarEnergy(events, [0.0, 10000], (0, 1, 2, "lin"), [0, 0.5], bin_time=0.1)
        base_lc, ref_lc = vespec._construct_lightcurves(
            [0, 0.5], tstart=0, tstop=0.65, exclude=False
        )
        np.testing.assert_equal(base_lc.counts, ref_lc.counts)

    def test_construct_lightcurves_pi(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54], pi=np.asarray([0, 0, 0, 0, 1, 1]), gti=[[0, 0.65]]
        )
        vespec = DummyVarEnergy(
            events, [0.0, 10000], (0, 1, 2, "lin"), [0.5, 1.1], use_pi=True, bin_time=0.1
        )
        base_lc, ref_lc = vespec._construct_lightcurves([0, 0.5], tstart=0, tstop=0.65)
        np.testing.assert_allclose(base_lc.counts, [1, 0, 2, 1, 0, 0])
        np.testing.assert_allclose(ref_lc.counts, [0, 0, 0, 0, 1, 1])


class TestCountSpectrum(object):
    @classmethod
    def setup_class(cls):

        cls.times = [0.1, 2, 4, 5.5]
        cls.energy = [3, 5, 2, 4]

        cls.events = EventList(time=cls.times, energy=cls.energy, pi=cls.energy, gti=[[0, 6.]])

    @pytest.mark.parametrize("use_pi", [False, True])
    def test_counts(self, use_pi):
        ctsspec = CountSpectrum(self.events, [1.5, 3.5, 6.5], use_pi=use_pi)
        assert np.allclose(ctsspec.spectrum, 2)


class TestRmsAndCovSpectrum(object):
    @classmethod
    def setup_class(cls):
        from ..simulator import Simulator

        cls.bin_time = 0.01

        data = np.load(os.path.join(datadir, "sample_variable_lc.npy"))
        flux = data / 50
        times = np.arange(data.size) * cls.bin_time

        test_lc = Lightcurve(times, flux, err_dist="gauss", dt=cls.bin_time, skip_checks=True)

        cls.test_ev1, cls.test_ev2 = EventList(), EventList()
        cls.test_ev1.simulate_times(test_lc)
        cls.test_ev2.simulate_times(test_lc)
        cls.test_ev1.energy = np.random.uniform(0.3, 12, len(cls.test_ev1.time))
        cls.test_ev2.energy = np.random.uniform(0.3, 12, len(cls.test_ev2.time))

    def test_create_complexcovariance(self):
        _ = ComplexCovarianceSpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=100,
            norm="abs",
            events2=self.test_ev2
        )

    @pytest.mark.parametrize("norm", ["frac", "abs"])
    def test_correct_rms_values_vs_cross(self, norm):
        """The rms calculated with independent event lists (from the cospectrum)
        is equivalent to the one calculated with one event list (from the PDS)"""

        rmsspec_cross = RmsEnergySpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=9999,
            events2=self.test_ev2,
            norm=norm,
        )
        rmsspec_pds = RmsEnergySpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=9999,
            norm=norm,
        )
        pds = rmsspec_pds.spectrum
        cross = rmsspec_cross.spectrum
        err = rmsspec_pds.spectrum_error
        cerr = rmsspec_cross.spectrum_error
        assert np.allclose(err, cerr, rtol=0.2)
        assert np.allclose(pds, cross, atol=3 * err)

        if norm == "frac":
            assert np.allclose(pds, 0.20, atol=3 * err)

    @pytest.mark.parametrize("norm", ["frac", "abs"])
    def test_correct_cov_values_vs_cross(self, norm):
        """The rms calculated with independent event lists (from the cospectrum)
        is equivalent to the one calculated with one event list (from the PDS)"""
        covar = CovarianceSpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=100,
            norm=norm
        )

        covar_cross = CovarianceSpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=100,
            norm=norm,
            events2=self.test_ev2
        )

        cov = covar.spectrum
        rms = covar_cross.spectrum
        coverr = covar_cross.spectrum_error

        assert np.allclose(cov, rms, atol=3 * coverr)

    @pytest.mark.parametrize("norm", ["frac", "abs"])
    def test_correct_rms_values_vs_cov(self, norm):
        """The rms calculated with independent event lists (from the cospectrum)
        is equivalent to the one calculated with one event list (from the PDS)"""
        covar = CovarianceSpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=100,
            norm=norm,
            events2=self.test_ev2
        )
        rmsspec = RmsSpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=100,
            norm=norm,
            events2=self.test_ev2
        )

        cov = covar.spectrum
        rms = rmsspec.spectrum
        coverr = covar.spectrum_error

        assert np.allclose(cov, rms, atol=3 * coverr)

    def test_cov_invalid_evlist_warns(self):
        ev = EventList(time=[], energy=[], gti=self.test_ev1.gti)
        with pytest.warns(UserWarning) as record:
            rms = CovarianceSpectrum(
                ev, [0.0, 100], (0.3, 12, 5, "lin"), bin_time=0.01, segment_size=100
            )
        assert np.all(np.isnan(rms.spectrum))
        assert np.all(np.isnan(rms.spectrum_error))

    def test_rms_invalid_evlist_warns(self):
        ev = EventList(time=[], energy=[], gti=self.test_ev1.gti)
        with pytest.warns(UserWarning) as record:
            rms = RmsEnergySpectrum(
                ev,
                [0.0, 100],
                (0.3, 12, 5, "lin"),
                bin_time=0.01,
                segment_size=100,
                events2=self.test_ev2,
            )
        assert np.all(np.isnan(rms.spectrum))
        assert np.all(np.isnan(rms.spectrum_error))


class TestLagEnergySpectrum(object):
    @classmethod
    def setup_class(cls):
        from ..simulator import Simulator

        dt = 0.01
        cls.time_lag = 5
        data = np.load(os.path.join(datadir, "sample_variable_lc.npy"))
        flux = data
        times = np.arange(data.size) * dt
        maxfreq = 0.25 / cls.time_lag
        roll_amount = int(cls.time_lag // dt)
        good = slice(roll_amount, roll_amount + int(200 / dt))

        rolled_flux = np.array(np.roll(flux, roll_amount))
        times, flux, rolled_flux = times[good], flux[good], rolled_flux[good]
        test_lc1 = Lightcurve(times, flux, err_dist="gauss", dt=dt, skip_checks=True)
        test_lc2 = Lightcurve(test_lc1.time, rolled_flux, err_dist=test_lc1.err_dist, dt=dt)
        test_ev1, test_ev2 = EventList(), EventList()
        test_ev1.simulate_times(test_lc1)
        test_ev2.simulate_times(test_lc2)

        test_ev1.energy = np.random.uniform(0.3, 9, len(test_ev1.time))
        test_ev2.energy = np.random.uniform(9, 12, len(test_ev2.time))

        cls.lag = LagEnergySpectrum(
            test_ev1,
            [0.0, maxfreq],
            (0.3, 9, 1, "lin"),
            [9, 12],
            bin_time=dt / 2,
            segment_size=199,
            events2=test_ev2,
        )

        # Make single event list
        test_ev = test_ev1.join(test_ev2)

        cls.lag_same = LagEnergySpectrum(
            test_ev,
            [0.0, maxfreq],
            (0.3, 9, 1, "lin"),
            [9, 12],
            bin_time=dt / 2,
            segment_size=199,
        )

    def test_lagspectrum_values_and_errors(self):
        assert np.all(np.abs(self.lag.spectrum - self.time_lag) < 3 * self.lag.spectrum_error)

    def test_lagspectrum_values_and_errors_same(self):
        assert np.all(np.abs(self.lag_same.spectrum - self.time_lag) < 3 * self.lag.spectrum_error)

    def test_lagspectrum_invalid_warns(self):
        ev = EventList(time=[], energy=[], gti=self.lag.events1.gti)
        with pytest.warns(UserWarning) as record:
            lag = LagSpectrum(
                ev,
                [0.0, 0.5],
                (0.3, 9, 4, "lin"),
                [9, 12],
                bin_time=0.1,
                segment_size=30,
                events2=self.lag.events2,
            )

        assert np.all(np.isnan(lag.spectrum))
        assert np.all(np.isnan(lag.spectrum_error))
