"""
Models that have physical origins.
"""
import warnings
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from .core import Fittable1DModel
from .parameters import InputParameterError, Parameter
__all__ = ['BlackBody', 'Drude1D', 'Plummer1D', 'NFW']

class BlackBody(Fittable1DModel):
    """
    Blackbody model using the Planck function.

    Parameters
    ----------
    temperature : `~astropy.units.Quantity` ['temperature']
        Blackbody temperature.

    scale : float or `~astropy.units.Quantity` ['dimensionless']
        Scale factor.  If dimensionless, input units will assumed
        to be in Hz and output units in (erg / (cm ** 2 * s * Hz * sr).
        If not dimensionless, must be equivalent to either
        (erg / (cm ** 2 * s * Hz * sr) or erg / (cm ** 2 * s * AA * sr),
        in which case the result will be returned in the requested units and
        the scale will be stripped of units (with the float value applied).

    Notes
    -----
    Model formula:

        .. math:: B_{\\nu}(T) = A \\frac{2 h \\nu^{3} / c^{2}}{exp(h \\nu / k T) - 1}

    Examples
    --------
    >>> from astropy.modeling import models
    >>> from astropy import units as u
    >>> bb = models.BlackBody(temperature=5000*u.K)
    >>> bb(6000 * u.AA)  # doctest: +FLOAT_CMP
    <Quantity 1.53254685e-05 erg / (Hz s sr cm2)>

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import BlackBody
        from astropy import units as u
        from astropy.visualization import quantity_support

        bb = BlackBody(temperature=5778*u.K)
        wav = np.arange(1000, 110000) * u.AA
        flux = bb(wav)

        with quantity_support():
            plt.figure()
            plt.semilogx(wav, flux)
            plt.axvline(bb.nu_max.to(u.AA, equivalencies=u.spectral()).value, ls='--')
            plt.show()
    """
    temperature = Parameter(default=5000.0, min=0, unit=u.K, description='Blackbody temperature')
    scale = Parameter(default=1.0, min=0, description='Scale factor')
    _input_units_allow_dimensionless = True
    input_units_equivalencies = {'x': u.spectral()}
    _native_units = u.erg / (u.cm ** 2 * u.s * u.Hz * u.sr)
    _native_output_units = {'SNU': u.erg / (u.cm ** 2 * u.s * u.Hz * u.sr), 'SLAM': u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)}

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        scale = kwargs.get('scale', None)
        if hasattr(scale, 'unit') and (not scale.unit.is_equivalent(u.dimensionless_unscaled)):
            output_units = scale.unit
            if not output_units.is_equivalent(self._native_units, u.spectral_density(1 * u.AA)):
                raise ValueError(f'scale units not dimensionless or in surface brightness: {output_units}')
            kwargs['scale'] = scale.value
            self._output_units = output_units
        else:
            self._output_units = self._native_units
        return super().__init__(*args, **kwargs)

    def evaluate(self, x, temperature, scale):
        if False:
            i = 10
            return i + 15
        "Evaluate the model.\n\n        Parameters\n        ----------\n        x : float, `~numpy.ndarray`, or `~astropy.units.Quantity` ['frequency']\n            Frequency at which to compute the blackbody. If no units are given,\n            this defaults to Hz (or AA if `scale` was initialized with units\n            equivalent to erg / (cm ** 2 * s * AA * sr)).\n\n        temperature : float, `~numpy.ndarray`, or `~astropy.units.Quantity`\n            Temperature of the blackbody. If no units are given, this defaults\n            to Kelvin.\n\n        scale : float, `~numpy.ndarray`, or `~astropy.units.Quantity` ['dimensionless']\n            Desired scale for the blackbody.\n\n        Returns\n        -------\n        y : number or ndarray\n            Blackbody spectrum. The units are determined from the units of\n            ``scale``.\n\n        .. note::\n\n            Use `numpy.errstate` to suppress Numpy warnings, if desired.\n\n        .. warning::\n\n            Output values might contain ``nan`` and ``inf``.\n\n        Raises\n        ------\n        ValueError\n            Invalid temperature.\n\n        ZeroDivisionError\n            Wavelength is zero (when converting to frequency).\n        "
        if not isinstance(temperature, u.Quantity):
            in_temp = u.Quantity(temperature, u.K)
        else:
            in_temp = temperature
        if not isinstance(x, u.Quantity):
            in_x = u.Quantity(x, self.input_units['x'])
        else:
            in_x = x
        with u.add_enabled_equivalencies(u.spectral() + u.temperature()):
            freq = u.Quantity(in_x, u.Hz, dtype=np.float64)
            temp = u.Quantity(in_temp, u.K)
        if np.any(temp < 0):
            raise ValueError(f'Temperature should be positive: {temp}')
        if not np.all(np.isfinite(freq)) or np.any(freq <= 0):
            warnings.warn('Input contains invalid wavelength/frequency value(s)', AstropyUserWarning)
        log_boltz = const.h * freq / (const.k_B * temp)
        boltzm1 = np.expm1(log_boltz)
        bb_nu = 2.0 * const.h * freq ** 3 / (const.c ** 2 * boltzm1) / u.sr
        if self.scale.unit is not None:
            if not hasattr(scale, 'unit'):
                scale = scale * self.scale.unit
            scale = scale.to(u.dimensionless_unscaled).value
        y = scale * bb_nu.to(self._output_units, u.spectral_density(freq))
        if hasattr(temperature, 'unit'):
            return y
        return y.value

    @property
    def input_units(self):
        if False:
            while True:
                i = 10
        if self._output_units.is_equivalent(self._native_output_units['SNU']):
            return {self.inputs[0]: u.Hz}
        else:
            return {self.inputs[0]: u.AA}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        if False:
            for i in range(10):
                print('nop')
        return {'temperature': u.K}

    @property
    def bolometric_flux(self):
        if False:
            return 10
        'Bolometric flux.'
        if self.scale.unit is not None:
            scale = self.scale.quantity.to(u.dimensionless_unscaled)
        else:
            scale = self.scale.value
        native_bolflux = scale * const.sigma_sb * self.temperature ** 4 / np.pi
        return native_bolflux.to(u.erg / (u.cm ** 2 * u.s))

    @property
    def lambda_max(self):
        if False:
            print('Hello World!')
        'Peak wavelength when the curve is expressed as power density.'
        return const.b_wien / self.temperature

    @property
    def nu_max(self):
        if False:
            for i in range(10):
                print('nop')
        'Peak frequency when the curve is expressed as power density.'
        return 2.8214391 * const.k_B * self.temperature / const.h

class Drude1D(Fittable1DModel):
    """
    Drude model based one the behavior of electons in materials (esp. metals).

    Parameters
    ----------
    amplitude : float
        Peak value
    x_0 : float
        Position of the peak
    fwhm : float
        Full width at half maximum

    Model formula:

        .. math:: f(x) = A \\frac{(fwhm/x_0)^2}{((x/x_0 - x_0/x)^2 + (fwhm/x_0)^2}

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Drude1D

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(7.5 , 12.5 , 0.1)

        dmodel = Drude1D(amplitude=1.0, fwhm=1.0, x_0=10.0)
        ax.plot(x, dmodel(x))

        ax.set_xlabel('x')
        ax.set_ylabel('F(x)')

        plt.show()
    """
    amplitude = Parameter(default=1.0, description='Peak Value')
    x_0 = Parameter(default=1.0, description='Position of the peak')
    fwhm = Parameter(default=1.0, description='Full width at half maximum')

    @staticmethod
    def evaluate(x, amplitude, x_0, fwhm):
        if False:
            for i in range(10):
                print('nop')
        '\n        One dimensional Drude model function.\n        '
        return amplitude * (fwhm / x_0) ** 2 / ((x / x_0 - x_0 / x) ** 2 + (fwhm / x_0) ** 2)

    @staticmethod
    def fit_deriv(x, amplitude, x_0, fwhm):
        if False:
            i = 10
            return i + 15
        '\n        Drude1D model function derivatives.\n        '
        d_amplitude = (fwhm / x_0) ** 2 / ((x / x_0 - x_0 / x) ** 2 + (fwhm / x_0) ** 2)
        d_x_0 = -2 * amplitude * d_amplitude * (1 / x_0 + d_amplitude * (x_0 ** 2 / fwhm ** 2) * ((-x / x_0 - 1 / x) * (x / x_0 - x_0 / x) - 2 * fwhm ** 2 / x_0 ** 3))
        d_fwhm = 2 * amplitude * d_amplitude / fwhm * (1 - d_amplitude)
        return [d_amplitude, d_x_0, d_fwhm]

    @property
    def input_units(self):
        if False:
            for i in range(10):
                print('nop')
        if self.x_0.input_unit is None:
            return None
        return {self.inputs[0]: self.x_0.input_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        if False:
            i = 10
            return i + 15
        return {'x_0': inputs_unit[self.inputs[0]], 'fwhm': inputs_unit[self.inputs[0]], 'amplitude': outputs_unit[self.outputs[0]]}

    @property
    def return_units(self):
        if False:
            while True:
                i = 10
        if self.amplitude.unit is None:
            return None
        return {self.outputs[0]: self.amplitude.unit}

    def _x_0_validator(self, val):
        if False:
            return 10
        'Ensure `x_0` is not 0.'
        if np.any(val == 0):
            raise InputParameterError('0 is not an allowed value for x_0')
    x_0._validator = _x_0_validator

    def bounding_box(self, factor=50):
        if False:
            i = 10
            return i + 15
        'Tuple defining the default ``bounding_box`` limits,\n        ``(x_low, x_high)``.\n\n        Parameters\n        ----------\n        factor : float\n            The multiple of FWHM used to define the limits.\n        '
        x0 = self.x_0
        dx = factor * self.fwhm
        return (x0 - dx, x0 + dx)

class Plummer1D(Fittable1DModel):
    """One dimensional Plummer density profile model.

    Parameters
    ----------
    mass : float
        Total mass of cluster.
    r_plum : float
        Scale parameter which sets the size of the cluster core.

    Notes
    -----
    Model formula:

    .. math::

        \\rho(r)=\\frac{3M}{4\\pi a^3}(1+\\frac{r^2}{a^2})^{-5/2}

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/1911MNRAS..71..460P
    """
    mass = Parameter(default=1.0, description='Total mass of cluster')
    r_plum = Parameter(default=1.0, description='Scale parameter which sets the size of the cluster core')

    @staticmethod
    def evaluate(x, mass, r_plum):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate plummer density profile model.\n        '
        return 3 * mass / (4 * np.pi * r_plum ** 3) * (1 + (x / r_plum) ** 2) ** (-5 / 2)

    @staticmethod
    def fit_deriv(x, mass, r_plum):
        if False:
            while True:
                i = 10
        '\n        Plummer1D model derivatives.\n        '
        d_mass = 3 / (4 * np.pi * r_plum ** 3 * ((x / r_plum) ** 2 + 1) ** (5 / 2))
        d_r_plum = (6 * mass * x ** 2 - 9 * mass * r_plum ** 2) / (4 * np.pi * r_plum ** 6 * (1 + (x / r_plum) ** 2) ** (7 / 2))
        return [d_mass, d_r_plum]

    @property
    def input_units(self):
        if False:
            return 10
        mass_unit = self.mass.input_unit
        r_plum_unit = self.r_plum.input_unit
        if mass_unit is None and r_plum_unit is None:
            return None
        return {self.inputs[0]: r_plum_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        if False:
            i = 10
            return i + 15
        return {'mass': outputs_unit[self.outputs[0]] * inputs_unit[self.inputs[0]] ** 3, 'r_plum': inputs_unit[self.inputs[0]]}

class NFW(Fittable1DModel):
    """
    Navarro–Frenk–White (NFW) profile - model for radial distribution of dark matter.

    Parameters
    ----------
    mass : float or `~astropy.units.Quantity` ['mass']
        Mass of NFW peak within specified overdensity radius.
    concentration : float
        Concentration of the NFW profile.
    redshift : float
        Redshift of the NFW profile.
    massfactor : tuple or str
        Mass overdensity factor and type for provided profiles:
            Tuple version:
                ("virial",) : virial radius

                ("critical", N)  : radius where density is N times that of the critical density

                ("mean", N)  : radius where density is N times that of the mean density

            String version:
                "virial" : virial radius

                "Nc"  : radius where density is N times that of the critical density (e.g. "200c")

                "Nm"  : radius where density is N times that of the mean density (e.g. "500m")
    cosmo : :class:`~astropy.cosmology.Cosmology`
        Background cosmology for density calculation. If None, the default cosmology will be used.

    Notes
    -----
    Model formula:

    .. math:: \\rho(r)=\\frac{\\delta_c\\rho_{c}}{r/r_s(1+r/r_s)^2}

    References
    ----------
    .. [1] https://arxiv.org/pdf/astro-ph/9508025
    .. [2] https://en.wikipedia.org/wiki/Navarro%E2%80%93Frenk%E2%80%93White_profile
    .. [3] https://en.wikipedia.org/wiki/Virial_mass
    """
    mass = Parameter(default=1.0, min=1.0, unit=u.M_sun, description='Peak mass within specified overdensity radius')
    concentration = Parameter(default=1.0, min=1.0, description='Concentration')
    redshift = Parameter(default=0.0, min=0.0, description='Redshift')
    _input_units_allow_dimensionless = True

    def __init__(self, mass=u.Quantity(mass.default, mass.unit), concentration=concentration.default, redshift=redshift.default, massfactor=('critical', 200), cosmo=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if cosmo is None:
            from astropy.cosmology import default_cosmology
            cosmo = default_cosmology.get()
        self._density_delta(massfactor, cosmo, redshift)
        if not isinstance(mass, u.Quantity):
            in_mass = u.Quantity(mass, u.M_sun)
        else:
            in_mass = mass
        self._radius_s(mass, concentration)
        self._density_s(mass, concentration)
        super().__init__(mass=in_mass, concentration=concentration, redshift=redshift, **kwargs)

    def evaluate(self, r, mass, concentration, redshift):
        if False:
            i = 10
            return i + 15
        "\n        One dimensional NFW profile function.\n\n        Parameters\n        ----------\n        r : float or `~astropy.units.Quantity` ['length']\n            Radial position of density to be calculated for the NFW profile.\n        mass : float or `~astropy.units.Quantity` ['mass']\n            Mass of NFW peak within specified overdensity radius.\n        concentration : float\n            Concentration of the NFW profile.\n        redshift : float\n            Redshift of the NFW profile.\n\n        Returns\n        -------\n        density : float or `~astropy.units.Quantity` ['density']\n            NFW profile mass density at location ``r``. The density units are:\n            [``mass`` / ``r`` ^3]\n\n        Notes\n        -----\n        .. warning::\n\n            Output values might contain ``nan`` and ``inf``.\n        "
        if hasattr(r, 'unit'):
            in_r = r
        else:
            in_r = u.Quantity(r, u.kpc)
        radius_reduced = in_r / self._radius_s(mass, concentration).to(in_r.unit)
        density = self._density_s(mass, concentration) / (radius_reduced * (u.Quantity(1.0) + radius_reduced) ** 2)
        if hasattr(mass, 'unit'):
            return density
        else:
            return density.value

    def _density_delta(self, massfactor, cosmo, redshift):
        if False:
            print('Hello World!')
        '\n        Calculate density delta.\n        '
        if isinstance(massfactor, tuple):
            if massfactor[0].lower() == 'virial':
                delta = None
                masstype = massfactor[0].lower()
            elif massfactor[0].lower() == 'critical':
                delta = float(massfactor[1])
                masstype = 'c'
            elif massfactor[0].lower() == 'mean':
                delta = float(massfactor[1])
                masstype = 'm'
            else:
                raise ValueError(f"Massfactor '{massfactor[0]}' not one of 'critical', 'mean', or 'virial'")
        else:
            try:
                if massfactor.lower() == 'virial':
                    delta = None
                    masstype = massfactor.lower()
                elif massfactor[-1].lower() == 'c' or massfactor[-1].lower() == 'm':
                    delta = float(massfactor[0:-1])
                    masstype = massfactor[-1].lower()
                else:
                    raise ValueError(f"Massfactor {massfactor} string not of the form '#m', '#c', or 'virial'")
            except (AttributeError, TypeError):
                raise TypeError(f'Massfactor {massfactor} not a tuple or string')
        if masstype == 'virial':
            Om_c = cosmo.Om(redshift) - 1.0
            d_c = 18.0 * np.pi ** 2 + 82.0 * Om_c - 39.0 * Om_c ** 2
            self.density_delta = d_c * cosmo.critical_density(redshift)
        elif masstype == 'c':
            self.density_delta = delta * cosmo.critical_density(redshift)
        elif masstype == 'm':
            self.density_delta = delta * cosmo.critical_density(redshift) * cosmo.Om(redshift)
        return self.density_delta

    @staticmethod
    def A_NFW(y):
        if False:
            while True:
                i = 10
        '\n        Dimensionless volume integral of the NFW profile, used as an intermediate step in some\n        calculations for this model.\n\n        Notes\n        -----\n        Model formula:\n\n        .. math:: A_{NFW} = [\\ln(1+y) - \\frac{y}{1+y}]\n        '
        return np.log(1.0 + y) - y / (1.0 + y)

    def _density_s(self, mass, concentration):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculate scale density of the NFW profile.\n        '
        if not isinstance(mass, u.Quantity):
            in_mass = u.Quantity(mass, u.M_sun)
        else:
            in_mass = mass
        self.density_s = in_mass / (4.0 * np.pi * self._radius_s(in_mass, concentration) ** 3 * self.A_NFW(concentration))
        return self.density_s

    @property
    def rho_scale(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Scale density of the NFW profile. Often written in the literature as :math:`\\rho_s`.\n        '
        return self.density_s

    def _radius_s(self, mass, concentration):
        if False:
            while True:
                i = 10
        '\n        Calculate scale radius of the NFW profile.\n        '
        if not isinstance(mass, u.Quantity):
            in_mass = u.Quantity(mass, u.M_sun)
        else:
            in_mass = mass
        self.radius_s = (3.0 * in_mass / (4.0 * np.pi * self.density_delta)) ** (1.0 / 3.0) / concentration
        return self.radius_s.to(u.kpc)

    @property
    def r_s(self):
        if False:
            i = 10
            return i + 15
        '\n        Scale radius of the NFW profile.\n        '
        return self.radius_s

    @property
    def r_virial(self):
        if False:
            print('Hello World!')
        '\n        Mass factor defined virial radius of the NFW profile (R200c for M200c, Rvir for Mvir, etc.).\n        '
        return self.r_s * self.concentration

    @property
    def r_max(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Radius of maximum circular velocity.\n        '
        return self.r_s * 2.16258

    @property
    def v_max(self):
        if False:
            while True:
                i = 10
        '\n        Maximum circular velocity.\n        '
        return self.circular_velocity(self.r_max)

    def circular_velocity(self, r):
        if False:
            i = 10
            return i + 15
        "\n        Circular velocities of the NFW profile.\n\n        Parameters\n        ----------\n        r : float or `~astropy.units.Quantity` ['length']\n            Radial position of velocity to be calculated for the NFW profile.\n\n        Returns\n        -------\n        velocity : float or `~astropy.units.Quantity` ['speed']\n            NFW profile circular velocity at location ``r``. The velocity units are:\n            [km / s]\n\n        Notes\n        -----\n        Model formula:\n\n        .. math:: v_{circ}(r)^2 = \\frac{1}{x}\\frac{\\ln(1+cx)-(cx)/(1+cx)}{\\ln(1+c)-c/(1+c)}\n\n        .. math:: x = r/r_s\n\n        .. warning::\n\n            Output values might contain ``nan`` and ``inf``.\n        "
        if hasattr(r, 'unit'):
            in_r = r
        else:
            in_r = u.Quantity(r, u.kpc)
        v_profile = np.sqrt(self.mass * const.G.to(in_r.unit ** 3 / (self.mass.unit * u.s ** 2)) / self.r_virial)
        reduced_radius = in_r / self.r_virial.to(in_r.unit)
        velocity = np.sqrt(v_profile ** 2 * self.A_NFW(self.concentration * reduced_radius) / (reduced_radius * self.A_NFW(self.concentration)))
        return velocity.to(u.km / u.s)

    @property
    def input_units(self):
        if False:
            return 10
        return {self.inputs[0]: u.kpc}

    @property
    def return_units(self):
        if False:
            for i in range(10):
                print('nop')
        if self.mass.unit is None:
            return {self.outputs[0]: u.M_sun / self.input_units[self.inputs[0]] ** 3}
        else:
            return {self.outputs[0]: self.mass.unit / self.input_units[self.inputs[0]] ** 3}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        if False:
            i = 10
            return i + 15
        return {'mass': u.M_sun, 'concentration': None, 'redshift': None}