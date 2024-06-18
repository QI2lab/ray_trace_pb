"""
Optical materials with wavelength dependent refractive indices
"""
import numpy as np

class Material:
    """
    Represent materials with wavelength-dependent index of refraction

    for information about various materials, see https://refractiveindex.info/ or https://www.schott.com
    abbe number vd = (nd - 1) / (nf - nc)
    vd > 50 = crown glass, otherwise flint glass
    """

    # helium d-line
    wd = 0.5876
    # hydrogen F-line
    wf = 0.4861
    # hydrogen c-line
    wc = 0.6563
    # abbe number
    vd = None

    def __init__(self, b_coeffs, c_coeffs):
        """
        Initialize Material using Sellmeier dispersion formula coefficients

        :param b_coeffs:
        :param c_coeffs:
        """
        # Sellmeier dispersion formula coefficients
        self.b1, self.b2, self.b3 = np.array(b_coeffs).squeeze()
        self.c1, self.c2, self.c3 = np.array(c_coeffs).squeeze()

        # abbe number (measure of dispersion)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.vd = (self.n(self.wd) - 1) / (self.n(self.wf) - self.n(self.wc))

    def n(self, wavelength: float):
        """
        compute index of refraction from Sellmeier dispersion formula. To use another method with a specific Material,
        override this function the derived class
        see https://www.schott.com/d/advanced_optics/02ffdb0d-00a6-408f-84a5-19de56652849/1.2/tie_29_refractive_index_and_dispersion_eng.pdf

        :param wavelength:
        :return refractive index:
        """
        val = self.b1 * wavelength ** 2 / (wavelength ** 2 - self.c1) + \
              self.b2 * wavelength ** 2 / (wavelength ** 2 - self.c2) + \
              self.b3 * wavelength ** 2 / (wavelength ** 2 - self.c3)
        return np.sqrt(val + 1)


class Vacuum(Material):
    def __init__(self):
        super(Vacuum, self).__init__([0., 0., 0.], [0., 0., 0.])


class Constant(Material):
    """
    Material with a Constant index of refraction versus wavelength
    """
    def __init__(self, n):
        self._n = float(n)
        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.c1 = None
        self.c2 = None
        self.c3 = None

    def n(self, wavelength: float):
        if isinstance(wavelength, float):
            ns = self._n
        else:
            wavelength = np.atleast_1d(np.array(wavelength))
            ns = np.ones(wavelength.shape) * self._n

        return ns


class FusedSilica(Material):
    def __init__(self):
        bs = [0.6961663, 0.4079426, 0.8974794]
        cs = [0.0684043**2, 0.1162414**2, 9.896161**2]
        super(FusedSilica, self).__init__(bs, cs)


# crown glasses (low dispersion, low refractive index)
class Bk7(Material):
    """
    BK7 is a common crown glass
    """
    def __init__(self):
        bs = [1.03961212, 0.231792344, 1.01046945]
        cs = [0.00600069867, 0.0200179144, 103.560653]
        super(Bk7, self).__init__(bs, cs)


class Nbak4(Material):
    """
    Nbak4 is a common crown glass
    """
    def __init__(self):
        bs = [1.28834642, 0.132817724, 0.945395373]
        cs = [0.00779980626, 0.0315631177, 105.965875]
        super(Nbak4, self).__init__(bs, cs)

class Nbaf10(Material):
    """
    Nbaf10 is a common crown glass
    """
    def __init__(self):
        bs = [1.5851495, 0.143559385, 1.08521269]
        cs = [0.00926681282, 0.0424489805, 105.613573]
        super(Nbaf10, self).__init__(bs, cs)


class Nlak22(Material):
    """
    https://www.schott.com/shop/advanced-optics/en/Optical-Glass/N-LAK22/c/glass-N-LAK22
    """
    def __init__(self):
        bs = [1.14229781, 0.535138441, 1.040883850]
        cs = [0.00585778594, 0.0198546147, 100.8340170]
        super(Nlak22, self).__init__(bs, cs)


# flint glasses (high dispersion, high refractive index)
class Sf10(Material):
    """
    Sf10 is a flint glass
    """
    def __init__(self):
        bs = [1.62153902, 0.256287842, 1.64447552]
        cs = [0.0122241457, 0.0595736775, 147.468793]
        super(Sf10, self).__init__(bs, cs)


class Nsf6(Material):
    """
    Nsf6 is a flint glass
    https://www.schott.com/shop/advanced-optics/en/Optical-Glass/N-SF6/c/glass-N-SF6
    """
    def __init__(self):
        bs = [1.77931763, 0.338149866, 2.087344740]
        cs = [0.01337141820, 0.0617533621, 174.0175900]
        super(Nsf6, self).__init__(bs, cs)


class Sf6(Material):
    """
    Sf6 is a flint glass
    https://www.schott.com/shop/advanced-optics/en/Optical-Glass/SF6/c/glass-SF6
    """
    def __init__(self):
        bs = [1.72448482, 0.390104889, 1.045728580]
        cs = [0.01348719470, 0.0569318095, 118.5571850]
        super(Sf6, self).__init__(bs, cs)


class Nsf6ht(Material):
    """
    flint glass
    https://www.schott.com/shop/advanced-optics/en/Optical-Glass/N-SF6HT/c/glass-N-SF6HT
    """
    def __init__(self):
        bs = [1.77931763, 0.338149866, 2.087344740]
        cs = [0.01337141820, 0.0617533621, 174.0175900]
        super(Nsf6ht, self).__init__(bs, cs)


class Sf2(Material):
    """
    flint glass
    """
    def __init__(self):
        bs = [1.40301821, 0.231767504, 0.939056586]
        cs = [0.0105795466, 0.0493226978, 112.405955]
        super(Sf2, self).__init__(bs, cs)
