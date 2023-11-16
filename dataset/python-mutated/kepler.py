import warnings
import numpy as np
from astropy.io import fits, registry
from astropy.table import MaskedColumn, Table
from astropy.time import Time, TimeDelta
from astropy.timeseries.sampled import TimeSeries
__all__ = ['kepler_fits_reader']

def kepler_fits_reader(filename, unit_parse_strict='warn'):
    if False:
        i = 10
        return i + 15
    '\n    This serves as the FITS reader for KEPLER or TESS files within\n    astropy-timeseries.\n\n    This function should generally not be called directly, and instead this\n    time series reader should be accessed with the\n    :meth:`~astropy.timeseries.TimeSeries.read` method::\n\n        >>> from astropy.timeseries import TimeSeries\n        >>> ts = TimeSeries.read(\'kplr33122.fits\', format=\'kepler.fits\')  # doctest: +SKIP\n\n    Parameters\n    ----------\n    filename : `str` or `pathlib.Path`\n        File to load.\n    unit_parse_strict : str, optional\n        Behaviour when encountering invalid column units in the FITS header.\n        Default is "warn", which will emit a ``UnitsWarning`` and create a\n        :class:`~astropy.units.core.UnrecognizedUnit`.\n        Values are the ones allowed by the ``parse_strict`` argument of\n        :class:`~astropy.units.core.Unit`: ``raise``, ``warn`` and ``silent``.\n\n    Returns\n    -------\n    ts : `~astropy.timeseries.TimeSeries`\n        Data converted into a TimeSeries.\n    '
    hdulist = fits.open(filename)
    telescope = hdulist[0].header['telescop'].lower()
    if telescope == 'tess':
        hdu = hdulist['LIGHTCURVE']
    elif telescope == 'kepler':
        hdu = hdulist[1]
    else:
        raise NotImplementedError(f"{hdulist[0].header['telescop']} is not implemented, only KEPLER or TESS are supported through this reader")
    if hdu.header['EXTVER'] > 1:
        raise NotImplementedError(f"Support for {hdu.header['TELESCOP']} v{hdu.header['EXTVER']} files not yet implemented")
    if hdu.header['TIMESYS'] != 'TDB':
        raise NotImplementedError(f"Support for {hdu.header['TIMESYS']} time scale not yet implemented in {hdu.header['TELESCOP']} reader")
    tab = Table.read(hdu, format='fits', unit_parse_strict=unit_parse_strict)
    if 'T' in tab.colnames:
        tab.rename_column('T', 'TIME')
    for colname in tab.colnames:
        unit = tab[colname].unit
        if unit and isinstance(tab[colname], MaskedColumn):
            tab[colname] = tab[colname].filled(np.nan)
        if unit == 'e-/s':
            tab[colname].unit = 'electron/s'
        if unit == 'pixels':
            tab[colname].unit = 'pixel'
        tab.rename_column(colname, colname.lower())
    nans = np.isnan(tab['time'].data)
    if np.any(nans):
        warnings.warn(f'Ignoring {np.sum(nans)} rows with NaN times')
    tab = tab[~nans]
    reference_date = Time(hdu.header['BJDREFI'], hdu.header['BJDREFF'], scale=hdu.header['TIMESYS'].lower(), format='jd')
    time = reference_date + TimeDelta(tab['time'].data, format='jd')
    time.format = 'isot'
    tab.remove_column('time')
    hdulist.close()
    return TimeSeries(time=time, data=tab)
registry.register_reader('kepler.fits', TimeSeries, kepler_fits_reader)
registry.register_reader('tess.fits', TimeSeries, kepler_fits_reader)