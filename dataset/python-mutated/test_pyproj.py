from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['pyproj'])
def test_pyproj(selenium):
    if False:
        while True:
            i = 10
    from pyproj import CRS, Transformer
    latlon = CRS.from_epsg(4326)
    assert latlon.get_geod().a == 6378137
    lcc = CRS.from_proj4('+proj=lcc +lat_1=25 +lat_2=40 +lat_0=35 +lon_0=-90')
    t = Transformer.from_crs(latlon, lcc)
    (x, y) = t.transform(35, -90)
    assert int(x) == 0
    assert int(y) == 0