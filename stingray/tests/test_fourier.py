import pytest
from stingray.fourier import *


def test_norm():
    mean = var = 100000
    N = 1000000
    dt = 0.2
    meanrate = mean / dt
    lc = np.random.poisson(mean, N)
    pds = np.abs(fft(lc)) ** 2
    freq = fftfreq(N, dt)
    good = slice(1, N // 2)

    pdsabs = normalize_abs(pds, dt, lc.size)
    pdsfrac = normalize_frac(pds, dt, lc.size, mean)
    pois_abs = poisson_level(meanrate=meanrate, norm="abs")
    pois_frac = poisson_level(meanrate=meanrate, norm="frac")

    assert np.isclose(pdsabs[good].mean(), pois_abs, rtol=0.01)
    assert np.isclose(pdsfrac[good].mean(), pois_frac, rtol=0.01)


class TestFourier(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 1
        cls.times = np.sort(np.random.uniform(0, 1000, 1000))
        cls.gti = np.asarray([[0, 1000]])
        cls.counts, bins = np.histogram(cls.times, bins=np.linspace(0, 1000, 1001))
        cls.bin_times = (bins[:-1] + bins[1:]) / 2
        cls.segment_size = 10.0
        cls.times2 = np.sort(np.random.uniform(0, 1000, 1000))
        cls.counts2, _ = np.histogram(cls.times2, bins=np.linspace(0, 1000, 1001))

    def test_ctrate_events(self):
        assert get_total_ctrate(self.times, self.gti, self.segment_size) == 1.0

    def test_ctrate_counts(self):
        assert get_total_ctrate(self.bin_times, self.gti, self.segment_size, self.counts) == 1.0

    def test_fts_from_segments_cts_and_events_are_equal(self):
        N = 10
        fts_evts = [
            (f, n) for (f, n) in get_fts_from_segments(self.times, self.gti, self.segment_size, N=N)
        ]
        fts_cts = [
            (f, n)
            for (f, n) in get_fts_from_segments(
                self.bin_times, self.gti, self.segment_size, counts=self.counts
            )
        ]
        for (fe, ne), (fc, nc) in zip(fts_evts, fts_cts):
            assert np.allclose(fe, fc)
            assert ne == nc

    @pytest.mark.parametrize("use_common_mean", [True, False])
    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_pds_cts_and_events_are_equal(self, norm, use_common_mean):
        out_ev = avg_pds_from_events(
            self.times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            fullspec=False,
            silent=False,
            power_type="all",
            counts=None,
        )
        out_ct = avg_pds_from_events(
            self.bin_times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            fullspec=False,
            silent=False,
            power_type="all",
            counts=self.counts,
        )
        for oe, oc in zip(out_ev, out_ct):
            if isinstance(oe, Iterable):
                assert np.allclose(oe, oc)
            else:
                assert np.isclose(oe, oc)

    @pytest.mark.parametrize("use_common_mean", [True, False])
    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_cs_cts_and_events_are_equal(self, norm, use_common_mean):
        out_ev = avg_cs_from_events(
            self.times,
            self.times2,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            fullspec=False,
            silent=False,
            power_type="all",
        )
        out_ct = avg_cs_from_events(
            self.bin_times,
            self.bin_times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            fullspec=False,
            silent=False,
            power_type="all",
            counts1=self.counts,
            counts2=self.counts2,
        )
        for oe, oc in zip(out_ev, out_ct):
            if isinstance(oe, Iterable):
                assert np.allclose(oe, oc)
            else:
                assert np.isclose(oe, oc)


class TestNorms(object):
    @classmethod
    def setup_class(cls):
        cls.mean = cls.var = 100000
        cls.N = 800000
        cls.dt = 0.2
        cls.df = 1 / (cls.N * cls.dt)
        freq = fftfreq(cls.N, cls.dt)
        good = freq > 0
        cls.good = good
        cls.meanrate = cls.mean / cls.dt
        cls.lc = np.random.poisson(cls.mean, cls.N)
        cls.pds = (np.abs(np.fft.fft(cls.lc)) ** 2)[good]
        cls.lc_bksub = cls.lc - cls.mean
        cls.pds_bksub = (np.abs(np.fft.fft(cls.lc_bksub)) ** 2)[good]
        cls.lc_renorm = cls.lc / cls.mean
        cls.pds_renorm = (np.abs(np.fft.fft(cls.lc_renorm)) ** 2)[good]
        cls.lc_renorm_bksub = cls.lc_renorm - 1
        cls.pds_renorm_bksub = (np.abs(np.fft.fft(cls.lc_renorm_bksub)) ** 2)[good]

    def test_leahy_bksub_var_vs_standard(self):
        """Test that the Leahy norm. does not change with background-subtracted lcs"""
        leahyvar = normalize_leahy_from_variance(self.pds_bksub, np.var(self.lc_bksub), self.N)
        leahy = 2 * self.pds / np.sum(self.lc)
        ratio = np.mean(leahyvar / leahy)
        assert np.isclose(ratio, 1, rtol=0.01)

    def test_abs_bksub(self):
        """Test that the abs rms normalization does not change with background-subtracted lcs"""
        ratio = normalize_abs(self.pds_bksub, self.dt, self.N) / normalize_abs(
            self.pds, self.dt, self.N
        )
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_frac_renorm_constant(self):
        """Test that the fractional rms normalization is equivalent when renormalized"""
        ratio = normalize_frac(self.pds_renorm, self.dt, self.N, 1) / normalize_frac(
            self.pds, self.dt, self.N, self.mean
        )
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_frac_to_abs_ctratesq(self):
        """Test that fractional rms normalization x ctrate**2 is equivalent to abs renormalized"""
        ratio = (
            normalize_frac(self.pds, self.dt, self.N, self.mean)
            / normalize_abs(self.pds, self.dt, self.N)
            * self.meanrate ** 2
        )
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_total_variance(self):
        """Test that the total variance of the unnormalized pds is the same as
        the variance from the light curve
        Attention: VdK defines the variance as sum (x - x0)**2.
        The usual definition is divided by 'N'
        """
        vdk_total_variance = np.sum((self.lc - self.mean) ** 2)
        ratio = np.mean(self.pds) / vdk_total_variance
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level(self, norm):
        pdsnorm = normalize_crossspectrum(self.pds, self.dt, self.N, self.mean, norm=norm)

        assert np.isclose(
            pdsnorm.mean(), poisson_level(meanrate=self.meanrate, norm=norm), rtol=0.01
        )

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level_real(self, norm):
        pdsnorm = normalize_crossspectrum(
            self.pds, self.dt, self.N, self.mean, norm=norm, power_type="real"
        )

        assert np.isclose(
            pdsnorm.mean(), poisson_level(meanrate=self.meanrate, norm=norm), rtol=0.01
        )

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level_absolute(self, norm):
        pdsnorm = normalize_crossspectrum(
            self.pds, self.dt, self.N, self.mean, norm=norm, power_type="abs"
        )

        assert np.isclose(
            pdsnorm.mean(), poisson_level(meanrate=self.meanrate, norm=norm), rtol=0.01
        )

    def test_normalize_with_variance(self):
        pdsnorm = normalize_crossspectrum(
            self.pds, self.dt, self.N, self.mean, variance=self.var, norm="leahy"
        )
        assert np.isclose(pdsnorm.mean(), 2, rtol=0.01)

    def test_normalize_none(self):
        pdsnorm = normalize_crossspectrum(self.pds, self.dt, self.N, self.mean, norm="none")
        assert np.isclose(pdsnorm.mean(), self.pds.mean(), rtol=0.01)

    def test_normalize_badnorm(self):
        with pytest.raises(ValueError):
            pdsnorm = normalize_crossspectrum(
                self.pds, self.var, self.N, self.mean, norm="asdfjlasdjf"
            )
