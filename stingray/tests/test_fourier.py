import pytest
from stingray.fourier import *

def test_norm():
    mean = var = 100000
    N = 1000000
    dt = 0.2
    meanrate = mean / dt
    lc = np.random.poisson(mean, N)
    pds = np.abs(fft(lc))**2

    pdsabs = normalize_abs(pds, dt, lc.size)
    pdsfrac = normalize_frac(pds, dt, lc.size, mean)

    assert np.isclose(pdsabs[1:N//2].mean(), poisson_level(meanrate=meanrate, norm="abs"), rtol=0.01)
    assert np.isclose(pdsfrac[1:N//2].mean(), poisson_level(meanrate=meanrate, norm="frac"), rtol=0.01)


class TestFourier(object):
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
        leahyvar =  normalize_leahy_from_variance(self.pds_bksub, np.var(self.lc_bksub), self.N)
        leahy = 2 * self.pds / np.sum(self.lc)
        ratio = np.mean(leahyvar / leahy)
        assert np.isclose(ratio, 1, rtol=0.01)

    def test_abs_bksub(self):
        """Test that the abs rms normalization does not change with background-subtracted lcs"""
        ratio = normalize_abs(self.pds_bksub, self.dt, self.N) / normalize_abs(self.pds, self.dt, self.N)
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_frac_renorm_constant(self):
        """Test that the fractional rms normalization is equivalent when renormalized"""
        ratio = normalize_frac(self.pds_renorm, self.dt, self.N, 1) / normalize_frac(self.pds, self.dt, self.N, self.mean)
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_frac_to_abs_ctratesq(self):
        """Test that the fractional rms normalization times squared count rate is equivalent to abs when renormalized"""
        ratio = normalize_frac(self.pds, self.dt, self.N, self.mean) / normalize_abs(self.pds, self.dt, self.N) * self.meanrate**2
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_total_variance(self):
        """Test that the total variance of the unnormalized pds is the same as
        the variance from the light curve
        Attention: VdK defines the variance as \sum (x - x0)**2.
        The usual definition is divided by 'N'
        """
        vdk_total_variance = np.sum((self.lc - self.mean)**2)
        ratio = np.mean(self.pds) / vdk_total_variance
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level(self, norm):
        pdsnorm = normalize_crossspectrum(self.pds, self.dt, self.N, self.mean,
                                          norm=norm)

        assert np.isclose(pdsnorm.mean(),
                          poisson_level(meanrate=self.meanrate, norm=norm),
                          rtol=0.01)

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level_real(self, norm):
        pdsnorm = normalize_crossspectrum(self.pds, self.dt, self.N, self.mean,
                                          norm=norm, power_type="real")

        assert np.isclose(pdsnorm.mean(),
                          poisson_level(meanrate=self.meanrate, norm=norm),
                          rtol=0.01)

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level_absolute(self, norm):
        pdsnorm = normalize_crossspectrum(self.pds, self.dt, self.N, self.mean,
                                          norm=norm, power_type="abs")

        assert np.isclose(pdsnorm.mean(),
                          poisson_level(meanrate=self.meanrate, norm=norm),
                          rtol=0.01)

    def test_normalize_with_variance(self):
        pdsnorm = normalize_crossspectrum(self.pds, self.dt, self.N, self.mean,
                                          variance=self.var, norm="leahy")
        assert np.isclose(pdsnorm.mean(), 2, rtol=0.01)

    def test_normalize_none(self):
        pdsnorm = normalize_crossspectrum(self.pds, self.dt, self.N, self.mean,
                                          norm="none")
        assert np.isclose(pdsnorm.mean(), self.pds.mean(), rtol=0.01)

    def test_normalize_badnorm(self):
        with pytest.raises(ValueError):
            pdsnorm = normalize_crossspectrum(self.pds, self.var, self.N, self.mean,
                                              norm="asdfjlasdjf")
