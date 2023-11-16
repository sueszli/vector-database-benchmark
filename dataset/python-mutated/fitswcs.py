import warnings
import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import ICRS, Galactic, SpectralCoord
from astropy.coordinates.spectral_coordinate import attach_zero_velocities, update_differentials_to_match
from astropy.utils.exceptions import AstropyUserWarning
from .high_level_api import HighLevelWCSMixin
from .low_level_api import BaseLowLevelWCS
from .wrappers import SlicedLowLevelWCS
__all__ = ['custom_ctype_to_ucd_mapping', 'SlicedFITSWCS', 'FITSWCSAPIMixin']
C_SI = c.si.value
VELOCITY_FRAMES = {'GEOCENT': 'gcrs', 'BARYCENT': 'icrs', 'HELIOCENT': 'hcrs', 'LSRK': 'lsrk', 'LSRD': 'lsrd'}
VELOCITY_FRAMES['GALACTOC'] = Galactic(u=0 * u.km, v=0 * u.km, w=0 * u.km, U=0 * u.km / u.s, V=-220 * u.km / u.s, W=0 * u.km / u.s, representation_type='cartesian', differential_type='cartesian')
VELOCITY_FRAMES['LOCALGRP'] = Galactic(u=0 * u.km, v=0 * u.km, w=0 * u.km, U=0 * u.km / u.s, V=-300 * u.km / u.s, W=0 * u.km / u.s, representation_type='cartesian', differential_type='cartesian')
VELOCITY_FRAMES['CMBDIPOL'] = Galactic(l=263.85 * u.deg, b=48.25 * u.deg, distance=0 * u.km, radial_velocity=-(0.003346 / 2.725 * c).to(u.km / u.s))
CTYPE_TO_UCD1 = {'RA': 'pos.eq.ra', 'DEC': 'pos.eq.dec', 'GLON': 'pos.galactic.lon', 'GLAT': 'pos.galactic.lat', 'ELON': 'pos.ecliptic.lon', 'ELAT': 'pos.ecliptic.lat', 'TLON': 'pos.bodyrc.lon', 'TLAT': 'pos.bodyrc.lat', 'HPLT': 'custom:pos.helioprojective.lat', 'HPLN': 'custom:pos.helioprojective.lon', 'HPRZ': 'custom:pos.helioprojective.z', 'HGLN': 'custom:pos.heliographic.stonyhurst.lon', 'HGLT': 'custom:pos.heliographic.stonyhurst.lat', 'CRLN': 'custom:pos.heliographic.carrington.lon', 'CRLT': 'custom:pos.heliographic.carrington.lat', 'SOLX': 'custom:pos.heliocentric.x', 'SOLY': 'custom:pos.heliocentric.y', 'SOLZ': 'custom:pos.heliocentric.z', 'FREQ': 'em.freq', 'ENER': 'em.energy', 'WAVN': 'em.wavenumber', 'WAVE': 'em.wl', 'VRAD': 'spect.dopplerVeloc.radio', 'VOPT': 'spect.dopplerVeloc.opt', 'ZOPT': 'src.redshift', 'AWAV': 'em.wl', 'VELO': 'spect.dopplerVeloc', 'BETA': 'custom:spect.doplerVeloc.beta', 'STOKES': 'phys.polarization.stokes', 'TIME': 'time', 'TAI': 'time', 'TT': 'time', 'TDT': 'time', 'ET': 'time', 'IAT': 'time', 'UT1': 'time', 'UTC': 'time', 'GMT': 'time', 'GPS': 'time', 'TCG': 'time', 'TCB': 'time', 'TDB': 'time', 'LOCAL': 'time', 'DIST': 'pos.distance', 'DSUN': 'custom:pos.distance.sunToObserver'}
CTYPE_TO_UCD1_CUSTOM = []

class custom_ctype_to_ucd_mapping:
    """
    A context manager that makes it possible to temporarily add new CTYPE to
    UCD1+ mapping used by :attr:`FITSWCSAPIMixin.world_axis_physical_types`.

    Parameters
    ----------
    mapping : dict
        A dictionary mapping a CTYPE value to a UCD1+ value

    Examples
    --------
    Consider a WCS with the following CTYPE::

        >>> from astropy.wcs import WCS
        >>> wcs = WCS(naxis=1)
        >>> wcs.wcs.ctype = ['SPAM']

    By default, :attr:`FITSWCSAPIMixin.world_axis_physical_types` returns `None`,
    but this can be overridden::

        >>> wcs.world_axis_physical_types
        [None]
        >>> with custom_ctype_to_ucd_mapping({'SPAM': 'food.spam'}):
        ...     wcs.world_axis_physical_types
        ['food.spam']
    """

    def __init__(self, mapping):
        if False:
            i = 10
            return i + 15
        CTYPE_TO_UCD1_CUSTOM.insert(0, mapping)
        self.mapping = mapping

    def __enter__(self):
        if False:
            print('Hello World!')
        pass

    def __exit__(self, type, value, tb):
        if False:
            i = 10
            return i + 15
        CTYPE_TO_UCD1_CUSTOM.remove(self.mapping)

class SlicedFITSWCS(SlicedLowLevelWCS, HighLevelWCSMixin):
    pass

class FITSWCSAPIMixin(BaseLowLevelWCS, HighLevelWCSMixin):
    """
    A mix-in class that is intended to be inherited by the
    :class:`~astropy.wcs.WCS` class and provides the low- and high-level WCS API.
    """

    @property
    def pixel_n_dim(self):
        if False:
            while True:
                i = 10
        return self.naxis

    @property
    def world_n_dim(self):
        if False:
            return 10
        return len(self.wcs.ctype)

    @property
    def array_shape(self):
        if False:
            print('Hello World!')
        if self.pixel_shape is None:
            return None
        else:
            return self.pixel_shape[::-1]

    @array_shape.setter
    def array_shape(self, value):
        if False:
            while True:
                i = 10
        if value is None:
            self.pixel_shape = None
        else:
            self.pixel_shape = value[::-1]

    @property
    def pixel_shape(self):
        if False:
            i = 10
            return i + 15
        if self._naxis == [0, 0]:
            return None
        else:
            return tuple(self._naxis)

    @pixel_shape.setter
    def pixel_shape(self, value):
        if False:
            while True:
                i = 10
        if value is None:
            self._naxis = [0, 0]
        else:
            if len(value) != self.naxis:
                raise ValueError(f'The number of data axes, {self.naxis}, does not equal the shape {len(value)}.')
            self._naxis = list(value)

    @property
    def pixel_bounds(self):
        if False:
            return 10
        return self._pixel_bounds

    @pixel_bounds.setter
    def pixel_bounds(self, value):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            self._pixel_bounds = value
        else:
            if len(value) != self.naxis:
                raise ValueError(f'The number of data axes, {self.naxis}, does not equal the number of pixel bounds {len(value)}.')
            self._pixel_bounds = list(value)

    @property
    def world_axis_physical_types(self):
        if False:
            for i in range(10):
                print('nop')
        types = []
        for ctype in self.wcs.ctype:
            if ctype.upper().startswith(('UT(', 'TT(')):
                types.append('time')
            else:
                ctype_name = ctype.split('-')[0]
                for custom_mapping in CTYPE_TO_UCD1_CUSTOM:
                    if ctype_name in custom_mapping:
                        types.append(custom_mapping[ctype_name])
                        break
                else:
                    types.append(CTYPE_TO_UCD1.get(ctype_name.upper(), None))
        return types

    @property
    def world_axis_units(self):
        if False:
            i = 10
            return i + 15
        units = []
        for unit in self.wcs.cunit:
            if unit is None:
                unit = ''
            elif isinstance(unit, u.Unit):
                unit = unit.to_string(format='vounit')
            else:
                try:
                    unit = u.Unit(unit).to_string(format='vounit')
                except u.UnitsError:
                    unit = ''
            units.append(unit)
        return units

    @property
    def world_axis_names(self):
        if False:
            while True:
                i = 10
        return list(self.wcs.cname)

    @property
    def axis_correlation_matrix(self):
        if False:
            while True:
                i = 10
        if self.has_distortion:
            return np.ones((self.world_n_dim, self.pixel_n_dim), dtype=bool)
        matrix = self.wcs.get_pc() != 0
        celestial = self.wcs.axis_types // 1000 % 10 == 2
        celestial_indices = np.nonzero(celestial)[0]
        for world1 in celestial_indices:
            for world2 in celestial_indices:
                if world1 != world2:
                    matrix[world1] |= matrix[world2]
                    matrix[world2] |= matrix[world1]
        return matrix

    def pixel_to_world_values(self, *pixel_arrays):
        if False:
            print('Hello World!')
        world = self.all_pix2world(*pixel_arrays, 0)
        return world[0] if self.world_n_dim == 1 else tuple(world)

    def world_to_pixel_values(self, *world_arrays):
        if False:
            for i in range(10):
                print('nop')
        from astropy.wcs.wcs import NoConvergence
        try:
            pixel = self.all_world2pix(*world_arrays, 0)
        except NoConvergence as e:
            warnings.warn(str(e))
            pixel = self._array_converter(lambda *args: e.best_solution, 'input', *world_arrays, 0)
        return pixel[0] if self.pixel_n_dim == 1 else tuple(pixel)

    @property
    def world_axis_object_components(self):
        if False:
            i = 10
            return i + 15
        return self._get_components_and_classes()[0]

    @property
    def world_axis_object_classes(self):
        if False:
            i = 10
            return i + 15
        return self._get_components_and_classes()[1]

    @property
    def serialized_classes(self):
        if False:
            print('Hello World!')
        return False

    def _get_components_and_classes(self):
        if False:
            print('Hello World!')
        wcs_hash = (self.naxis, list(self.wcs.ctype), list(self.wcs.cunit), self.wcs.radesys, self.wcs.specsys, self.wcs.equinox, self.wcs.dateobs, self.wcs.lng, self.wcs.lat)
        if getattr(self, '_components_and_classes_cache', None) is not None:
            cache = self._components_and_classes_cache
            if cache[0] == wcs_hash:
                return cache[1]
            else:
                self._components_and_classes_cache = None
        from astropy.coordinates import EarthLocation, SkyCoord, StokesCoord
        from astropy.time import Time, TimeDelta
        from astropy.time.formats import FITS_DEPRECATED_SCALES
        from astropy.wcs.utils import wcs_to_celestial_frame
        components = [None] * self.naxis
        classes = {}
        if self.has_celestial:
            try:
                celestial_frame = wcs_to_celestial_frame(self)
            except ValueError:
                celestial_frame = None
            else:
                kwargs = {}
                kwargs['frame'] = celestial_frame
                kwargs['unit'] = (u.Unit(self.wcs.cunit[self.wcs.lng]), u.Unit(self.wcs.cunit[self.wcs.lat]))
                classes['celestial'] = (SkyCoord, (), kwargs)
                components[self.wcs.lng] = ('celestial', 0, 'spherical.lon.degree')
                components[self.wcs.lat] = ('celestial', 1, 'spherical.lat.degree')
        if self.has_spectral:
            ispec = self.wcs.spec
            ctype = self.wcs.ctype[ispec][:4]
            ctype = ctype.upper()
            kwargs = {}
            if np.isnan(self.wcs.obsgeo[0]):
                observer = None
            else:
                earth_location = EarthLocation(*self.wcs.obsgeo[:3], unit=u.m)
                tscale = self.wcs.timesys.lower() or 'utc'
                if np.isnan(self.wcs.mjdavg):
                    obstime = Time(self.wcs.mjdobs, format='mjd', scale=tscale, location=earth_location)
                else:
                    obstime = Time(self.wcs.mjdavg, format='mjd', scale=tscale, location=earth_location)
                observer_location = SkyCoord(earth_location.get_itrs(obstime=obstime))
                if self.wcs.specsys in VELOCITY_FRAMES:
                    frame = VELOCITY_FRAMES[self.wcs.specsys]
                    observer = observer_location.transform_to(frame)
                    if isinstance(frame, str):
                        observer = attach_zero_velocities(observer)
                    else:
                        observer = update_differentials_to_match(observer_location, VELOCITY_FRAMES[self.wcs.specsys], preserve_observer_frame=True)
                elif self.wcs.specsys == 'TOPOCENT':
                    observer = attach_zero_velocities(observer_location)
                else:
                    raise NotImplementedError(f'SPECSYS={self.wcs.specsys} not yet supported')
            if self.has_celestial and celestial_frame is not None:
                target = SkyCoord(self.wcs.crval[self.wcs.lng] * self.wcs.cunit[self.wcs.lng], self.wcs.crval[self.wcs.lat] * self.wcs.cunit[self.wcs.lat], frame=celestial_frame, distance=1000 * u.kpc)
                target = attach_zero_velocities(target)
            else:
                target = None
            if observer is not None:
                try:
                    observer.transform_to(ICRS())
                except Exception:
                    warnings.warn('observer cannot be converted to ICRS, so will not be set on SpectralCoord', AstropyUserWarning)
                    observer = None
            if target is not None:
                try:
                    target.transform_to(ICRS())
                except Exception:
                    warnings.warn('target cannot be converted to ICRS, so will not be set on SpectralCoord', AstropyUserWarning)
                    target = None
            if ctype == 'ZOPT':

                def spectralcoord_from_redshift(redshift):
                    if False:
                        while True:
                            i = 10
                    if isinstance(redshift, SpectralCoord):
                        return redshift
                    return SpectralCoord((redshift + 1) * self.wcs.restwav, unit=u.m, observer=observer, target=target)

                def redshift_from_spectralcoord(spectralcoord):
                    if False:
                        print('Hello World!')
                    if observer is None or spectralcoord.observer is None or spectralcoord.target is None:
                        if observer is None:
                            msg = 'No observer defined on WCS'
                        elif spectralcoord.observer is None:
                            msg = 'No observer defined on SpectralCoord'
                        else:
                            msg = 'No target defined on SpectralCoord'
                        warnings.warn(f'{msg}, SpectralCoord will be converted without any velocity frame change', AstropyUserWarning)
                        return spectralcoord.to_value(u.m) / self.wcs.restwav - 1.0
                    else:
                        return spectralcoord.with_observer_stationary_relative_to(observer).to_value(u.m) / self.wcs.restwav - 1.0
                classes['spectral'] = (u.Quantity, (), {}, spectralcoord_from_redshift)
                components[self.wcs.spec] = ('spectral', 0, redshift_from_spectralcoord)
            elif ctype == 'BETA':

                def spectralcoord_from_beta(beta):
                    if False:
                        for i in range(10):
                            print('nop')
                    if isinstance(beta, SpectralCoord):
                        return beta
                    return SpectralCoord(beta * C_SI, unit=u.m / u.s, doppler_convention='relativistic', doppler_rest=self.wcs.restwav * u.m, observer=observer, target=target)

                def beta_from_spectralcoord(spectralcoord):
                    if False:
                        while True:
                            i = 10
                    doppler_equiv = u.doppler_relativistic(self.wcs.restwav * u.m)
                    if observer is None or spectralcoord.observer is None or spectralcoord.target is None:
                        if observer is None:
                            msg = 'No observer defined on WCS'
                        elif spectralcoord.observer is None:
                            msg = 'No observer defined on SpectralCoord'
                        else:
                            msg = 'No target defined on SpectralCoord'
                        warnings.warn(f'{msg}, SpectralCoord will be converted without any velocity frame change', AstropyUserWarning)
                        return spectralcoord.to_value(u.m / u.s, doppler_equiv) / C_SI
                    else:
                        return spectralcoord.with_observer_stationary_relative_to(observer).to_value(u.m / u.s, doppler_equiv) / C_SI
                classes['spectral'] = (u.Quantity, (), {}, spectralcoord_from_beta)
                components[self.wcs.spec] = ('spectral', 0, beta_from_spectralcoord)
            else:
                kwargs['unit'] = self.wcs.cunit[ispec]
                if self.wcs.restfrq > 0:
                    if ctype == 'VELO':
                        kwargs['doppler_convention'] = 'relativistic'
                        kwargs['doppler_rest'] = self.wcs.restfrq * u.Hz
                    elif ctype == 'VRAD':
                        kwargs['doppler_convention'] = 'radio'
                        kwargs['doppler_rest'] = self.wcs.restfrq * u.Hz
                    elif ctype == 'VOPT':
                        kwargs['doppler_convention'] = 'optical'
                        kwargs['doppler_rest'] = self.wcs.restwav * u.m

                def spectralcoord_from_value(value):
                    if False:
                        while True:
                            i = 10
                    if isinstance(value, SpectralCoord):
                        return value
                    return SpectralCoord(value, observer=observer, target=target, **kwargs)

                def value_from_spectralcoord(spectralcoord):
                    if False:
                        return 10
                    if observer is None or spectralcoord.observer is None or spectralcoord.target is None:
                        if observer is None:
                            msg = 'No observer defined on WCS'
                        elif spectralcoord.observer is None:
                            msg = 'No observer defined on SpectralCoord'
                        else:
                            msg = 'No target defined on SpectralCoord'
                        warnings.warn(f'{msg}, SpectralCoord will be converted without any velocity frame change', AstropyUserWarning)
                        return spectralcoord.to_value(**kwargs)
                    else:
                        return spectralcoord.with_observer_stationary_relative_to(observer).to_value(**kwargs)
                classes['spectral'] = (u.Quantity, (), {}, spectralcoord_from_value)
                components[self.wcs.spec] = ('spectral', 0, value_from_spectralcoord)
        if 'time' in self.world_axis_physical_types:
            multiple_time = self.world_axis_physical_types.count('time') > 1
            for i in range(self.naxis):
                if self.world_axis_physical_types[i] == 'time':
                    if multiple_time:
                        name = f'time.{i}'
                    else:
                        name = 'time'
                    reference_time_delta = None
                    scale = self.wcs.ctype[i].split('-')[0].lower()
                    if scale == 'time':
                        if self.wcs.timesys:
                            scale = self.wcs.timesys.lower()
                        else:
                            scale = 'utc'
                    if '(' in scale:
                        pos = scale.index('(')
                        (scale, subscale) = (scale[:pos], scale[pos + 1:-1])
                        warnings.warn(f'Dropping unsupported sub-scale {subscale.upper()} from scale {scale.upper()}', UserWarning)
                    if scale == 'gps':
                        reference_time_delta = TimeDelta(19, format='sec')
                        scale = 'tai'
                    elif scale.upper() in FITS_DEPRECATED_SCALES:
                        scale = FITS_DEPRECATED_SCALES[scale.upper()]
                    elif scale not in Time.SCALES:
                        raise ValueError(f'Unrecognized time CTYPE={self.wcs.ctype[i]}')
                    trefpos = self.wcs.trefpos.lower()
                    if trefpos.startswith('topocent'):
                        if np.any(np.isnan(self.wcs.obsgeo[:3])):
                            warnings.warn('Missing or incomplete observer location information, setting location in Time to None', UserWarning)
                            location = None
                        else:
                            location = EarthLocation(*self.wcs.obsgeo[:3], unit=u.m)
                    elif trefpos == 'geocenter':
                        location = EarthLocation(0, 0, 0, unit=u.m)
                    elif trefpos == '':
                        location = None
                    else:
                        warnings.warn(f"Observation location '{trefpos}' is not supported, setting location in Time to None", UserWarning)
                        location = None
                    reference_time = Time(np.nan_to_num(self.wcs.mjdref[0]), np.nan_to_num(self.wcs.mjdref[1]), format='mjd', scale=scale, location=location)
                    if reference_time_delta is not None:
                        reference_time = reference_time + reference_time_delta

                    def time_from_reference_and_offset(offset):
                        if False:
                            for i in range(10):
                                print('nop')
                        if isinstance(offset, Time):
                            return offset
                        return reference_time + TimeDelta(offset, format='sec')

                    def offset_from_time_and_reference(time):
                        if False:
                            print('Hello World!')
                        return (time - reference_time).sec
                    classes[name] = (Time, (), {}, time_from_reference_and_offset)
                    components[i] = (name, 0, offset_from_time_and_reference)
        if 'phys.polarization.stokes' in self.world_axis_physical_types:
            for i in range(self.naxis):
                if self.world_axis_physical_types[i] == 'phys.polarization.stokes':
                    name = 'stokes'
                    classes[name] = (StokesCoord, (), {})
                    components[i] = (name, 0, 'value')
        for i in range(self.naxis):
            if components[i] is None:
                name = self.wcs.ctype[i].split('-')[0].lower()
                if name == '':
                    name = 'world'
                while name in classes:
                    name += '_'
                classes[name] = (u.Quantity, (), {'unit': self.wcs.cunit[i]})
                components[i] = (name, 0, 'value')
        self._components_and_classes_cache = (wcs_hash, (components, classes))
        return (components, classes)