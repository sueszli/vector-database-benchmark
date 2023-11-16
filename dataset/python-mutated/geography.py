from typing import Optional
import geohash as geohash_lib
from flask_babel import gettext as _
from geopy.point import Point
from pandas import DataFrame
from superset.exceptions import InvalidPostProcessingError
from superset.utils.pandas_postprocessing.utils import _append_columns

def geohash_decode(df: DataFrame, geohash: str, longitude: str, latitude: str) -> DataFrame:
    if False:
        for i in range(10):
            print('nop')
    '\n    Decode a geohash column into longitude and latitude\n\n    :param df: DataFrame containing geohash data\n    :param geohash: Name of source column containing geohash location.\n    :param longitude: Name of new column to be created containing longitude.\n    :param latitude: Name of new column to be created containing latitude.\n    :return: DataFrame with decoded longitudes and latitudes\n    '
    try:
        lonlat_df = DataFrame()
        (lonlat_df['latitude'], lonlat_df['longitude']) = zip(*df[geohash].apply(geohash_lib.decode))
        return _append_columns(df, lonlat_df, {'latitude': latitude, 'longitude': longitude})
    except ValueError as ex:
        raise InvalidPostProcessingError(_('Invalid geohash string')) from ex

def geohash_encode(df: DataFrame, geohash: str, longitude: str, latitude: str) -> DataFrame:
    if False:
        for i in range(10):
            print('nop')
    '\n    Encode longitude and latitude into geohash\n\n    :param df: DataFrame containing longitude and latitude data\n    :param geohash: Name of new column to be created containing geohash location.\n    :param longitude: Name of source column containing longitude.\n    :param latitude: Name of source column containing latitude.\n    :return: DataFrame with decoded longitudes and latitudes\n    '
    try:
        encode_df = df[[latitude, longitude]]
        encode_df.columns = ['latitude', 'longitude']
        encode_df['geohash'] = encode_df.apply(lambda row: geohash_lib.encode(row['latitude'], row['longitude']), axis=1)
        return _append_columns(df, encode_df, {'geohash': geohash})
    except ValueError as ex:
        raise InvalidPostProcessingError(_('Invalid longitude/latitude')) from ex

def geodetic_parse(df: DataFrame, geodetic: str, longitude: str, latitude: str, altitude: Optional[str]=None) -> DataFrame:
    if False:
        return 10
    '\n    Parse a column containing a geodetic point string\n    [Geopy](https://geopy.readthedocs.io/en/stable/#geopy.point.Point).\n\n    :param df: DataFrame containing geodetic point data\n    :param geodetic: Name of source column containing geodetic point string.\n    :param longitude: Name of new column to be created containing longitude.\n    :param latitude: Name of new column to be created containing latitude.\n    :param altitude: Name of new column to be created containing altitude.\n    :return: DataFrame with decoded longitudes and latitudes\n    '

    def _parse_location(location: str) -> tuple[float, float, float]:
        if False:
            i = 10
            return i + 15
        '\n        Parse a string containing a geodetic point and return latitude, longitude\n        and altitude\n        '
        point = Point(location)
        return (point[0], point[1], point[2])
    try:
        geodetic_df = DataFrame()
        (geodetic_df['latitude'], geodetic_df['longitude'], geodetic_df['altitude']) = zip(*df[geodetic].apply(_parse_location))
        columns = {'latitude': latitude, 'longitude': longitude}
        if altitude:
            columns['altitude'] = altitude
        return _append_columns(df, geodetic_df, columns)
    except ValueError as ex:
        raise InvalidPostProcessingError(_('Invalid geodetic string')) from ex