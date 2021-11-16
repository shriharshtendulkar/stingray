import os
import numpy as np
import pytest
import warnings
import matplotlib.pyplot as plt
import scipy.special
from astropy.io import fits
from stingray import Lightcurve
from stingray import Crossspectrum, AveragedCrossspectrum, coherence, time_lag
from stingray.crossspectrum import  cospectra_pvalue, normalize_crossspectrum
from stingray import StingrayError
from ..simulator import Simulator

from stingray.events import EventList
import copy

np.random.seed(20160528)
curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


def avg_cdf_two_spectra(x):

    prefac = 0.25

    if x >= 0:
        fac1 = 2 * scipy.special.gamma(2) - scipy.special.gammaincc(2, 2 * x)
        fac2 = 2. * scipy.special.gamma(1) - scipy.special.gammaincc(1, 2 * x)
    else:
        fac1 = scipy.special.gammaincc(2, -2 * x)
        fac2 = scipy.special.gammaincc(1, -2 * x)

    return prefac * (fac1 + fac2)

class TestClassicalPvalue(object):

    def test_pval_returns_float_when_float_input(self):
        power = 1.0
        nspec = 1.0
        pval = cospectra_pvalue(power, nspec)
        assert isinstance(pval, float)

    def test_pval_returns_iterable_when_iterable_input(self):
        power = [0, 1, 2]
        nspec = 1.0
        pval = cospectra_pvalue(power, nspec)
        assert isinstance(pval, np.ndarray)
        assert len(pval) == len(power)

    def test_pval_fails_if_single_power_infinite(self):
        power = np.inf
        nspec = 1
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_single_power_nan(self):
        power = np.nan
        nspec = 1
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_multiple_powers_nan(self):
        power = [1, np.nan, 2.0]
        nspec = 1
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_multiple_powers_inf(self):
        power = [1, 2.0, np.inf]
        nspec = 1
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_nspec_zero(self):
        power = 1.0
        nspec = 0
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_nspec_negative(self):
        power = 1.0
        nspec = -10
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_nspec_not_integer(self):
        power = 1.0
        nspec = 1.5
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_single_spectrum(self):
        # the Laplace distribution is symmetric around
        # 0, so a power of 0 should return p=0.5
        power = 0.0
        nspec = 1
        assert cospectra_pvalue(power, nspec) == 0.5

    def test_single_spectrum_with_positive_power(self):
        """
        Because the Laplace distribution is always symmetric
        around zero, let's do a second version where I look
        for a different number.
        """
        power = 0.69314718055
        nspec = 1
        assert np.isclose(cospectra_pvalue(power, nspec), 0.25)

    def test_two_averaged_spectra(self):
        """
        For nspec=2, I can derive this by hand:
        """
        power = 1.0
        nspec = 2
        manual_pval = 1.0 - avg_cdf_two_spectra(power)
        assert np.isclose(cospectra_pvalue(power, nspec), manual_pval)

    def test_sixty_spectra(self):
        power = 1.0
        nspec = 60
        gauss = scipy.stats.norm(0, np.sqrt(2/(nspec+1)))
        pval_theory = gauss.sf(power)
        assert np.isclose(cospectra_pvalue(power, nspec), pval_theory)


class TestAveragedCrossspectrumEvents(object):

    def setup_class(self):
        tstart = 0.0
        tend = 1.0
        self.dt = np.longdouble(0.0001)
        segment_size = 1
        self.segment_size = segment_size
        N = np.rint(segment_size / self.dt).astype(int)
        # adjust dt
        self.dt = segment_size / N

        times1 = np.sort(np.random.uniform(tstart, tend, 1000000))
        times2 = np.sort(np.random.uniform(tstart, tend, 1000000))
        gti = np.array([[tstart, tend]])

        self.events1 = EventList(times1, gti=gti)
        self.events2 = EventList(times2, gti=gti)

        self.cs = Crossspectrum(self.events1, self.events2, dt=self.dt)

        self.acs = AveragedCrossspectrum(self.events1.to_lc(self.dt),
                                         self.events2.to_lc(self.dt),
                                         segment_size=segment_size, dt=self.dt, norm='none')
        self.lc1, self.lc2 = self.events1, self.events2

    def test_from_events_works(self):
        lccs = AveragedCrossspectrum.from_events(self.events1, self.events2,
                                                 segment_size=1, dt=self.dt, norm='none',
                                                 use_common_mean=False)
        power1 = lccs.power.real
        power2 = self.acs.power.real
        assert np.allclose(power1, power2, rtol=0.01)

    def test_from_time_array_works_with_memmap(self):
        with fits.open(os.path.join(datadir, "monol_testA.evt"), memmap=True) as hdul:
            times1 = hdul[1].data["TIME"]

            gti = np.array([[hdul[2].data["START"][0], hdul[2].data["STOP"][0]]])

            times2 = np.random.uniform(gti[0, 0], gti[0, 1], 1000)

            _ = AveragedCrossspectrum.from_time_array(
                times1, times2, segment_size=128, dt=self.dt, gti=gti, norm='none',
                use_common_mean=False)

    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_from_lc_works(self, norm):
        pds = AveragedCrossspectrum.from_lightcurve(
            self.events1.to_lc(self.dt), self.events2.to_lc(self.dt),
            segment_size=self.segment_size, norm=norm)
        pds_ev = AveragedCrossspectrum.from_events(
            self.events1, self.events2,
            segment_size=self.segment_size, dt=self.dt, norm=norm)
        for attr in ["power", "freq", "m", "n", "nphots1", "nphots2", "segment_size"]:
            assert np.allclose(getattr(pds, attr), getattr(pds_ev, attr))

    def test_it_works_with_events(self):
        lc1 = self.events1.to_lc(self.dt)
        lc2 = self.events2.to_lc(self.dt)
        lccs = Crossspectrum(lc1, lc2)
        assert np.allclose(lccs.power, self.cs.power)

    def test_no_segment_size(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, dt=self.dt)

    def test_init_with_norm_not_str(self):
        with pytest.raises(TypeError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1,
                                       norm=1, dt=self.dt)

    def test_init_with_invalid_norm(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1,
                                       norm='frabs', dt=self.dt)

    def test_init_with_inifite_segment_size(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2,
                                       segment_size=np.inf, dt=self.dt)

    def test_coherence(self):
        with pytest.warns(UserWarning) as w:
            coh = self.acs.coherence()
        assert len(coh[0]) == 4999
        assert len(coh[1]) == 4999

    def test_failure_when_normalization_not_recognized(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2,
                                       segment_size=1,
                                       norm="wrong", dt=self.dt)

    def test_failure_when_power_type_not_recognized(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2,
                                       segment_size=1,
                                       power_type="wrong", dt=self.dt)

    def test_rebin(self):
        new_cs = self.acs.rebin(df=1.5)
        assert new_cs.df == 1.5
        new_cs.time_lag()

    def test_rebin_factor(self):
        new_cs = self.acs.rebin(f=1.5)
        assert new_cs.df == self.acs.df * 1.5
        new_cs.time_lag()

    def test_rebin_log(self):
        # For now, just verify that it doesn't crash
        new_cs = self.acs.rebin_log(f=0.1)
        assert type(new_cs) == type(self.acs)
        new_cs.time_lag()

    def test_rebin_log_returns_complex_values(self):
        # For now, just verify that it doesn't crash
        new_cs = self.acs.rebin_log(f=0.1)
        assert np.iscomplexobj(new_cs.power[0])

    def test_rebin_log_returns_complex_errors(self):
        # For now, just verify that it doesn't crash

        new_cs = self.acs.rebin_log(f=0.1)
        assert np.iscomplexobj(new_cs.power_err[0])


class TestCoherenceFunction(object):

    def setup_class(self):
        self.lc1 = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        self.lc2 = Lightcurve([1, 2, 3, 4, 5], [4, 8, 1, 9, 11])

    def test_coherence_runs(self):
        with pytest.warns(UserWarning) as record:
            coh = coherence(self.lc1, self.lc2)

    def test_coherence_fails_if_data1_not_lc(self):
        data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]])

        with pytest.raises(TypeError):
            coh = coherence(self.lc1, data)

    def test_coherence_fails_if_data2_not_lc(self):
        data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]])

        with pytest.raises(TypeError):
            coh = coherence(data, self.lc2)

    def test_coherence_computes_correctly(self):
        with pytest.warns(UserWarning) as record:
            coh = coherence(self.lc1, self.lc2)

        assert len(coh) == 2
        assert np.abs(np.mean(coh)) < 1


class TestTimelagFunction(object):

    def setup_class(self):
        self.lc1 = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        self.lc2 = Lightcurve([1, 2, 3, 4, 5], [4, 8, 1, 9, 11])

    def test_time_lag_runs(self):
        with pytest.warns(UserWarning) as record:
            lag = time_lag(self.lc1, self.lc2)

    def test_time_lag_fails_if_data1_not_lc(self):
        data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]])

        with pytest.raises(TypeError):
            lag = time_lag(self.lc1, data)

    def test_time_lag_fails_if_data2_not_lc(self):
        data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]])

        with pytest.raises(TypeError):
            lag = time_lag(data, self.lc2)

    def test_time_lag_computes_correctly(self):
        with pytest.warns(UserWarning) as record:
            lag = time_lag(self.lc1, self.lc2)

        assert np.max(lag) <= np.pi
        assert np.min(lag) >= -np.pi


class TestCoherence(object):

    def test_coherence(self):
        lc1 = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        lc2 = Lightcurve([1, 2, 3, 4, 5], [4, 8, 1, 9, 11])

        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(lc1, lc2)
            coh = cs.coherence()

        assert len(coh) == 2
        assert np.abs(np.mean(coh)) < 1

    def test_high_coherence(self):
        t = np.arange(1280)
        a = np.random.poisson(100, len(t))
        lc = Lightcurve(t, a)
        lc2 = Lightcurve(t, copy.copy(a))

        with pytest.warns(UserWarning) as record:
            c = AveragedCrossspectrum(lc, lc2, 128)
            coh, _ = c.coherence()

        np.testing.assert_almost_equal(np.mean(coh).real, 1.0)



class TestNormalization(object):

    def setup_class(self):
        tstart = 0.0
        self.tseg = 100000.0
        dt = 1

        time = np.arange(tstart + 0.5 * dt, self.tseg + 0.5 * dt, dt)

        np.random.seed(100)
        counts1 = np.random.poisson(10000, size=time.shape[0])
        counts1_norm = counts1 / 13.4
        counts1_norm_err = np.std(counts1) / 13.4
        self.lc1_norm = \
            Lightcurve(time, counts1_norm, gti=[[tstart, self.tseg]], dt=dt,
                       err_dist='gauss', err=np.zeros_like(counts1_norm) + counts1_norm_err)
        self.lc1 = Lightcurve(time, counts1, gti=[[tstart, self.tseg]], dt=dt)
        self.rate1 = np.mean(counts1) / dt  # mean count rate (counts/sec) of light curve 1

        with pytest.warns(UserWarning) as record:
            self.cs = Crossspectrum(self.lc1, self.lc1, norm="none")

        with pytest.warns(UserWarning) as record:
            self.cs_norm = Crossspectrum(self.lc1_norm, self.lc1_norm, norm="none")

    @pytest.mark.parametrize('power_type', ['all', 'real', 'absolute'])
    def test_norm_abs(self, power_type):
        # Testing for a power spectrum of lc1
        self.cs.norm = 'abs'
        # New lc with the same absolute variance, but mean-subtracted
        norm_lc_sub = copy.deepcopy(self.lc1)
        norm_lc_sub.counts = norm_lc_sub.counts - np.mean(norm_lc_sub.counts)
        norm_lc_sub.err_dist = 'gauss'
        cs = Crossspectrum(norm_lc_sub, norm_lc_sub, norm="none")
        cs.norm = 'abs'
        cs.power_type = power_type
        self.cs.power_type = power_type

        power = self.cs._normalize_crossspectrum(self.cs.unnorm_power, self.tseg)
        power_norm = cs._normalize_crossspectrum(cs.unnorm_power, self.tseg)
        abs_noise = 2. * self.rate1  # expected Poisson noise level
        assert np.isclose(np.mean(power[1:]), abs_noise, rtol=0.01)
        assert np.allclose(power[1:], power_norm[1:], atol=0.5)

    @pytest.mark.parametrize('power_type', ['all', 'real', 'absolute'])
    def test_norm_leahy(self, power_type):

        self.cs.norm = 'leahy'
        self.cs_norm.norm = 'leahy'
        self.cs.power_type = power_type
        self.cs_norm.power_type = power_type

        power = self.cs._normalize_crossspectrum(self.cs.unnorm_power, self.tseg)
        power_norm = self.cs_norm._normalize_crossspectrum(self.cs_norm.unnorm_power, self.tseg)

        assert np.allclose(power[1:], power_norm[1:], atol=0.5)
        leahy_noise = 2.0  # expected Poisson noise level
        assert np.isclose(np.mean(power[1:]), leahy_noise, rtol=0.02)

    @pytest.mark.parametrize('power_type', ['all', 'real', 'absolute'])
    def test_norm_frac(self, power_type):
        self.cs.norm = 'frac'
        self.cs_norm.norm = 'frac'
        self.cs.power_type = power_type
        self.cs_norm.power_type = power_type
        power = self.cs._normalize_crossspectrum(self.cs.unnorm_power, self.tseg)
        power_norm = self.cs_norm._normalize_crossspectrum(self.cs_norm.unnorm_power, self.tseg)

        assert np.allclose(power[1:], power_norm[1:])
        norm = 2. / self.rate1
        assert np.isclose(np.mean(power[1:]), norm, rtol=0.1)

    def test_failure_when_normalization_not_recognized(self):
        with pytest.raises(ValueError):
            power = normalize_crossspectrum(self.cs.power, self.lc1.tseg, self.lc1.n,
                                            self.cs.nphots1, self.cs.nphots2,
                                            norm="wrong")
        self.cs.norm = 'asdgfasdfa'
        self.cs_norm.norm = 'adfafaf'
        with pytest.raises(ValueError):
            power = self.cs._normalize_crossspectrum(self.cs.unnorm_power,
                                                     self.tseg)
        with pytest.raises(ValueError):
            power = self.cs_norm._normalize_crossspectrum(self.cs.unnorm_power,
                                                          self.tseg)

    def test_failure_wrong_power_type(self):
        self.cs.power_type = 'asdgfasdfa'
        self.cs_norm.power_type = 'adfafaf'
        self.cs.norm = 'leahy'
        self.cs_norm.norm = 'leahy'

        with pytest.raises(ValueError):
            power = self.cs._normalize_crossspectrum(self.cs.unnorm_power,
                                                     self.tseg)
        with pytest.raises(ValueError):
            power = self.cs_norm._normalize_crossspectrum(self.cs.unnorm_power,
                                                          self.tseg)


class TestCrossspectrum(object):

    def setup_class(self):
        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.arange(tstart + 0.5 * dt, tend + 0.5 * dt, dt)

        counts1 = np.random.poisson(0.01, size=time.shape[0])
        counts2 = np.random.negative_binomial(1, 0.09, size=time.shape[0])
        self.lc1 = Lightcurve(time, counts1, gti=[[tstart, tend]], dt=dt)
        self.lc2 = Lightcurve(time, counts2, gti=[[tstart, tend]], dt=dt)
        self.rate1 = 100.  # mean count rate (counts/sec) of light curve 1

        with pytest.warns(UserWarning) as record:
            self.cs = Crossspectrum(self.lc1, self.lc2)

    def test_lc_keyword_deprecation(self):
        cs1 = Crossspectrum(self.lc1, self.lc2)
        with pytest.warns(DeprecationWarning) as record:
            cs2 = Crossspectrum(lc1=self.lc1, lc2=self.lc2)
        assert np.any(['lcN keywords' in r.message.args[0]
                       for r in record])
        assert np.allclose(cs1.power, cs2.power)
        assert np.allclose(cs1.freq, cs2.freq)

    def test_make_empty_crossspectrum(self):
        cs = Crossspectrum()
        assert cs.freq is None
        assert cs.power is None
        assert cs.df is None
        assert cs.nphots1 is None
        assert cs.nphots2 is None
        assert cs.m == 1
        assert cs.n is None
        assert cs.power_err is None

    def test_init_with_one_lc_none(self):
        with pytest.raises(TypeError):
            cs = Crossspectrum(self.lc1)

    def test_init_with_multiple_gti(self):
        gti = np.array([[0.0, 0.2], [0.6, 1.0]])
        with pytest.raises(TypeError):
            cs = Crossspectrum(self.lc1, self.lc2, gti=gti)

    def test_init_with_norm_not_str(self):
        with pytest.raises(TypeError):
            cs = Crossspectrum(norm=1)

    def test_init_with_invalid_norm(self):
        with pytest.raises(ValueError):
            cs = Crossspectrum(norm='frabs')

    def test_init_with_wrong_lc1_instance(self):
        lc_ = {"a":1, "b":2}
        with pytest.raises(TypeError):
            cs = Crossspectrum(lc_, self.lc2)

    def test_init_with_wrong_lc2_instance(self):
        lc_ =  {"a":1, "b":2}
        with pytest.raises(TypeError):
            cs = Crossspectrum(self.lc1, lc_)

    def test_make_crossspectrum_diff_lc_counts_shape(self):
        counts = np.array([1] * 10001)
        time = np.linspace(0.0, 1.0001, 10001)
        lc_ = Lightcurve(time, counts)
        with pytest.raises(StingrayError):
            cs = Crossspectrum(self.lc1, lc_)

    def test_make_crossspectrum_diff_lc_stat(self):
        lc_ = copy.copy(self.lc1)
        lc_.err_dist = 'gauss'

        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, lc_)
        assert np.any(["different statistics" in r.message.args[0]
                       for r in record])

    def test_make_crossspectrum_bad_lc_stat(self):
        lc1 = copy.copy(self.lc1)
        lc1.err_dist = 'gauss'
        lc2 = copy.copy(self.lc1)
        lc2.err_dist = 'gauss'

        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(lc1, lc2)
        assert np.any(["is not poisson" in r.message.args[0]
                       for r in record])

    def test_make_crossspectrum_diff_dt(self):
        counts = np.array([1] * 10000)
        time = np.linspace(0.0, 2.0, 10000)
        lc_ = Lightcurve(time, counts)
        with pytest.raises(StingrayError):
            cs = Crossspectrum(self.lc1, lc_)

    def test_rebin_smaller_resolution(self):
        # Original df is between 0.9 and 1.0
        with pytest.raises(ValueError):
            new_cs = self.cs.rebin(df=0.1)

    def test_rebin(self):
        new_cs = self.cs.rebin(df=1.5)
        assert new_cs.df == 1.5
        new_cs.time_lag()

    def test_rebin_factor(self):
        new_cs = self.cs.rebin(f=1.5)
        assert new_cs.df == self.cs.df * 1.5
        new_cs.time_lag()

    def test_rebin_log(self):
        # For now, just verify that it doesn't crash
        new_cs = self.cs.rebin_log(f=0.1)
        assert type(new_cs) == type(self.cs)
        new_cs.time_lag()

    def test_norm_abs(self):
        # Testing for a power spectrum of lc1
        cs = Crossspectrum(self.lc1, self.lc1, norm='abs')
        assert len(cs.power) == 4999
        assert cs.norm == 'abs'
        abs_noise = 2. * self.rate1  # expected Poisson noise level
        assert np.isclose(np.mean(cs.power[1:]), abs_noise)

    def test_norm_leahy(self):
        # with pytest.warns(UserWarning) as record:
        cs = Crossspectrum(self.lc1, self.lc1,
                           norm='leahy')
        assert len(cs.power) == 4999
        assert cs.norm == 'leahy'
        leahy_noise = 2.0  # expected Poisson noise level
        assert np.isclose(np.mean(cs.power[1:]), leahy_noise, rtol=0.02)

    def test_norm_frac(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc1, norm='frac')
        assert len(cs.power) == 4999
        assert cs.norm == 'frac'
        norm = 2. / self.rate1
        assert np.isclose(np.mean(cs.power[1:]), norm, rtol=0.2)

    def test_norm_abs(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm='abs')
        assert len(cs.power) == 4999
        assert cs.norm == 'abs'

    def test_failure_when_normalization_not_recognized(self):
        with pytest.raises(ValueError):
            cs = Crossspectrum(self.lc1, self.lc2, norm='wrong')

    def test_coherence(self):
        coh = self.cs.coherence()
        assert len(coh) == 4999
        assert np.abs(coh[0]) < 1

    def test_timelag(self):
        time_lag = self.cs.time_lag()
        assert np.max(time_lag) <= np.pi
        assert np.min(time_lag) >= -np.pi

    def test_nonzero_err(self):
        assert np.all(self.cs.power_err > 0)

    def test_timelag_error(self):
        class Child(Crossspectrum):
            def __init__(self):
                pass

        obj = Child()
        with pytest.raises(AttributeError):
            lag = obj.time_lag()

    def test_plot_simple(self):
        self.cs.plot()
        assert plt.fignum_exists('crossspectrum')

    def test_rebin_error(self):
        cs = Crossspectrum()
        with pytest.raises(ValueError):
            cs.rebin()

    def test_classical_significances_runs(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm='leahy')
        cs.classical_significances()

    def test_classical_significances_fails_in_rms(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm='frac')
        with pytest.raises(ValueError):
            cs.classical_significances()

    def test_classical_significances_threshold(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm='leahy')

        # change the powers so that just one exceeds the threshold
        cs.power = np.zeros_like(cs.power) + 2.0

        index = 1
        cs.power[index] = 10.0

        threshold = 0.01

        pval = cs.classical_significances(threshold=threshold,
                                          trial_correction=False)
        assert pval[0, 0] < threshold
        assert pval[1, 0] == index

    def test_classical_significances_trial_correction(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm='leahy')
        # change the powers so that just one exceeds the threshold
        cs.power = np.zeros_like(cs.power) + 2.0
        index = 1
        cs.power[index] = 10.0
        threshold = 0.01
        pval = cs.classical_significances(threshold=threshold,
                                          trial_correction=True)
        assert np.size(pval) == 0


    def test_classical_significances_with_logbinned_psd(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm='leahy')
        cs_log = cs.rebin_log()
        pval = cs_log.classical_significances(threshold=1.1,
                                              trial_correction=False)

        assert len(pval[0]) == len(cs_log.power)

    def test_pvals_is_numpy_array(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='leahy')
        # change the powers so that just one exceeds the threshold
        cs.power = np.zeros_like(cs.power) + 2.0

        index = 1
        cs.power[index] = 10.0

        threshold = 1.0

        pval = cs.classical_significances(threshold=threshold,
                                          trial_correction=True)

        assert isinstance(pval, np.ndarray)
        assert pval.shape[0] == 2

    def test_fullspec(self):
        csT = Crossspectrum(self.lc1, self.lc2, fullspec=True)
        assert csT.fullspec == True
        assert self.cs.fullspec == False
        assert csT.n == self.cs.n
        assert csT.n == len(csT.power)
        assert self.cs.n != len(self.cs.power)
        assert len(csT.power) >= len(self.cs.power)
        assert len(csT.power) == len(self.lc1)
        assert csT.freq[csT.n//2] <= 0.

class TestAveragedCrossspectrum(object):

    def setup_class(self):
        tstart = 0.0
        tend = 1.0
        dt = np.longdouble(0.0001)

        time = np.arange(tstart + 0.5 * dt, tend + 0.5 * dt, dt)

        counts1 = np.random.poisson(0.01, size=time.shape[0])
        counts2 = np.random.negative_binomial(1, 0.09, size=time.shape[0])

        self.lc1 = Lightcurve(time, counts1, gti=[[tstart, tend]], dt=dt)
        self.lc2 = Lightcurve(time, counts2, gti=[[tstart, tend]], dt=dt)

        with pytest.warns(UserWarning) as record:
            self.cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1)

    def test_save_all(self):
        cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1,
                                   save_all=True)
        assert hasattr(cs, 'cs_all')

    def test_lc_keyword_deprecation(self):
        cs1 = AveragedCrossspectrum(data1=self.lc1, data2=self.lc2,
                                    segment_size=1)
        with pytest.warns(DeprecationWarning) as record:
            cs2 = AveragedCrossspectrum(lc1=self.lc1, lc2=self.lc2,
                                        segment_size=1)
        assert np.any(['lcN keywords' in r.message.args[0]
                       for r in record])
        assert np.allclose(cs1.power, cs2.power)
        assert np.allclose(cs1.freq, cs2.freq)

    def test_make_empty_crossspectrum(self):
        cs = AveragedCrossspectrum()
        assert cs.freq is None
        assert cs.power is None
        assert cs.df is None
        assert cs.nphots1 is None
        assert cs.nphots2 is None
        assert cs.m == 1
        assert cs.n is None
        assert cs.power_err is None

    def test_no_counts_warns(self):
        newlc = copy.deepcopy(self.lc1)
        newlc.counts[:newlc.counts.size // 2] = \
            0 * newlc.counts[:newlc.counts.size // 2]
        with pytest.warns(UserWarning) as record:
            ps = AveragedCrossspectrum(newlc, self.lc2, segment_size=0.2)

        assert np.any(["No counts in "
                       in r.message.args[0] for r in record])

    def test_no_segment_size(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2)

    def test_invalid_type_attribute(self):
        with pytest.raises(ValueError):
            cs_test = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1)
            cs_test.type = 'invalid_type'
            assert AveragedCrossspectrum._make_crossspectrum(cs_test,
                                                             self.lc1,
                                                             self.lc2)

    def test_invalid_type_attribute_with_multiple_lcs(self):
        with pytest.warns(UserWarning) as record:
            acs_test = AveragedCrossspectrum([self.lc1, self.lc2],
                                             [self.lc2, self.lc1],
                                             segment_size=1)
        acs_test.type = 'invalid_type'
        with pytest.raises(ValueError) as excinfo:
            assert AveragedCrossspectrum._make_crossspectrum(acs_test,
                                                             [self.lc1,
                                                              self.lc2],
                                                             [self.lc2,
                                                                  self.lc1])
        assert "Type of spectrum not recognized" in str(excinfo.value)

    def test_different_dt(self):
        time1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        counts1_test = np.random.poisson(0.01, size=len(time1))
        test_lc1 = Lightcurve(time1, counts1_test)

        time2 = [2, 4, 6, 8, 10]
        counts2_test = np.random.negative_binomial(1, 0.09, size=len(time2))
        test_lc2 = Lightcurve(time2, counts2_test)

        assert test_lc1.tseg == test_lc2.tseg

        assert test_lc1.dt != test_lc2.dt

        with pytest.raises(ValueError):
            assert AveragedCrossspectrum(test_lc1, test_lc2, segment_size=1)

    def test_different_tseg(self):
        time2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        counts2_test = np.random.poisson(1000, size=len(time2))
        test_lc2 = Lightcurve(time2, counts2_test)

        time1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        counts1_test = np.random.poisson(1000, size=len(time1))
        test_lc1 = Lightcurve(time1, counts1_test)

        assert test_lc2.dt == test_lc1.dt

        assert test_lc2.tseg != test_lc1.tseg

        with pytest.warns(UserWarning) as record:
            AveragedCrossspectrum(test_lc1, test_lc2, segment_size=5)
            assert np.any(["same tseg" in r.message.args[0]
                           for r in record])

    def test_with_zero_counts(self):
        nbins = 100
        x = np.linspace(0, 10, nbins)
        ycounts1 = np.random.normal(loc=10, scale=0.5, size=int(0.4*nbins))
        ycounts2 = np.random.normal(loc=10, scale=0.5, size=int(0.4*nbins))

        yzero = np.zeros(int(0.6*nbins))
        y1 = np.hstack([ycounts1, yzero])
        y2 = np.hstack([ycounts2, yzero])

        lc1 = Lightcurve(x, y1)
        lc2 = Lightcurve(x, y2)

        with pytest.warns(UserWarning) as record:
            acs = AveragedCrossspectrum(
                lc1, lc2, segment_size=5.0, norm="leahy")
        assert acs.m == 1
        assert np.any(["No counts in interval" in r.message.args[0]
                       for r in record])


    def test_rebin_with_invalid_type_attribute(self):
        new_df = 2

        with pytest.warns(UserWarning) as record:
            aps = AveragedCrossspectrum(lc1=self.lc1, lc2=self.lc2,
                                        segment_size=1, norm='leahy')
        aps.type = 'invalid_type'
        with pytest.raises(ValueError) as excinfo:
            assert aps.rebin(df=new_df, method=aps.type)
        assert "Method for summing or averaging not recognized. " in str(excinfo.value)

    def test_rebin_with_valid_type_attribute(self):
        new_df = 2
        with pytest.warns(UserWarning) as record:
            aps = AveragedCrossspectrum(self.lc1, self.lc2,
                                        segment_size=1, norm='leahy')
        assert aps.rebin(df=new_df)

    def test_init_with_norm_not_str(self):
        with pytest.raises(TypeError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1,
                                       norm=1)

    def test_init_with_invalid_norm(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1,
                                       norm='frabs')

    def test_init_with_inifite_segment_size(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=np.inf)

    def test_with_iterable_of_lightcurves(self):
        def iter_lc(lc, n):
            "Generator of n parts of lc."
            t0 = int(len(lc) / n)
            t = t0
            i = 0
            while (True):
                lc_seg = lc[i:t]
                yield lc_seg
                if t + t0 > len(lc):
                    break
                else:
                    i, t = t, t + t0

        with pytest.warns(UserWarning) as record:
            cs = AveragedCrossspectrum(iter_lc(self.lc1, 1), iter_lc(self.lc2, 1),
                                   segment_size=1)


    def test_with_multiple_lightcurves_variable_length(self):
        gti = [[0, 0.05], [0.05, 0.5], [0.555, 1.0]]
        lc1 = copy.deepcopy(self.lc1)
        lc1.gti = gti
        lc2 = copy.deepcopy(self.lc2)
        lc2.gti = gti

        lc1_split = lc1.split_by_gti()
        lc2_split = lc2.split_by_gti()

        cs = AveragedCrossspectrum(lc1_split, lc2_split, segment_size=0.05,
                                   norm="leahy", silent=True)


    def test_coherence(self):
        with warnings.catch_warnings(record=True) as w:
            coh = self.cs.coherence()

            assert len(coh[0]) == 4999
            assert len(coh[1]) == 4999
            assert issubclass(w[-1].category, UserWarning)

    def test_failure_when_normalization_not_recognized(self):
        with pytest.raises(ValueError):
            self.cs = AveragedCrossspectrum(self.lc1, self.lc2,
                                            segment_size=1,
                                            norm="wrong")

    def test_failure_when_power_type_not_recognized(self):
        with pytest.raises(ValueError):
            self.cs = AveragedCrossspectrum(self.lc1, self.lc2,
                                            segment_size=1,
                                            power_type="wrong")

    def test_normalize_crossspectrum(self):
        cs1 = Crossspectrum(self.lc1, self.lc2, norm="leahy")
        cs2 = Crossspectrum(self.lc1, self.lc2, norm="leahy",
                            power_type="all")
        cs3 = Crossspectrum(self.lc1, self.lc2, norm="leahy",
                            power_type="real")
        cs4 = Crossspectrum(self.lc1, self.lc2, norm="leahy",
                            power_type="absolute")
        assert np.allclose(cs1.power.real, cs3.power)
        assert np.all(np.isclose(np.abs(cs2.power), cs4.power, atol=0.0001))

    def test_rebin(self):
        with warnings.catch_warnings(record=True) as w:
            new_cs = self.cs.rebin(df=1.5)
        assert new_cs.df == 1.5
        new_cs.time_lag()

    def test_rebin_factor(self):
        with warnings.catch_warnings(record=True) as w:
            new_cs = self.cs.rebin(f=1.5)
        assert new_cs.df == self.cs.df * 1.5
        new_cs.time_lag()

    def test_rebin_log(self):
        # For now, just verify that it doesn't crash
        with warnings.catch_warnings(record=True) as w:
            new_cs = self.cs.rebin_log(f=0.1)
        assert type(new_cs) == type(self.cs)
        new_cs.time_lag()

    def test_rebin_log_returns_complex_values(self):
        # For now, just verify that it doesn't crash
        with warnings.catch_warnings(record=True) as w:
            new_cs = self.cs.rebin_log(f=0.1)
        assert np.iscomplexobj(new_cs.power[0])

    def test_rebin_log_returns_complex_errors(self):
        # For now, just verify that it doesn't crash
        with warnings.catch_warnings(record=True) as w:
            new_cs = self.cs.rebin_log(f=0.1)
        assert np.iscomplexobj(new_cs.power_err[0])

    def test_timelag(self):
        dt = 0.1
        simulator = Simulator(dt, 10000, rms=0.2, mean=1000)
        test_lc1 = simulator.simulate(2)
        test_lc1.counts -= np.min(test_lc1.counts)

        with pytest.warns(UserWarning):
            test_lc1 = Lightcurve(test_lc1.time,
                                  test_lc1.counts,
                                  err_dist=test_lc1.err_dist,
                                  dt=dt)
            test_lc2 = Lightcurve(test_lc1.time,
                                  np.array(np.roll(test_lc1.counts, 2)),
                                  err_dist=test_lc1.err_dist,
                                  dt=dt)

        with warnings.catch_warnings(record=True) as w:
            cs = AveragedCrossspectrum(test_lc1, test_lc2, segment_size=5,
                                       norm="none")

            time_lag, time_lag_err = cs.time_lag()

        assert np.all(np.abs(time_lag[:6] - 0.1) < 3 * time_lag_err[:6])

    def test_errorbars(self):
        time = np.arange(10000) * 0.1
        test_lc1 = Lightcurve(time, np.random.poisson(200, 10000))
        test_lc2 = Lightcurve(time, np.random.poisson(200, 10000))

        with warnings.catch_warnings(record=True) as w:
            cs = AveragedCrossspectrum(test_lc1, test_lc2, segment_size=10,
                                       norm="leahy")

        assert np.allclose(cs.power_err, np.sqrt(2 / cs.m))

    def test_classical_significances(self):
        time = np.arange(10000) * 0.1
        np.random.seed(62)
        test_lc1 = Lightcurve(time, np.random.poisson(200, 10000))
        test_lc2 = Lightcurve(time, np.random.poisson(200, 10000))
        with warnings.catch_warnings(record=True) as w:

            cs = AveragedCrossspectrum(test_lc1, test_lc2, segment_size=10,
                                       norm="leahy")
        maxpower = np.max(cs.power)
        assert np.all(np.isfinite(cs.classical_significances(threshold = maxpower/2.)))
