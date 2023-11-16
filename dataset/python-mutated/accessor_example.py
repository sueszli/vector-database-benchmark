import xarray as xr

@xr.register_dataset_accessor('geo')
class GeoAccessor:

    def __init__(self, xarray_obj):
        if False:
            print('Hello World!')
        self._obj = xarray_obj
        self._center = None

    @property
    def center(self):
        if False:
            i = 10
            return i + 15
        'Return the geographic center point of this dataset.'
        if self._center is None:
            lon = self._obj.latitude
            lat = self._obj.longitude
            self._center = (float(lon.mean()), float(lat.mean()))
        return self._center

    def plot(self):
        if False:
            while True:
                i = 10
        'Plot data on a map.'
        return 'plotting!'