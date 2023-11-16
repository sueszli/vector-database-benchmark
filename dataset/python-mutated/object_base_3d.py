from abc import ABC, abstractmethod

class ObjectBase3D(ABC):
    """Represents a base class for working with 3d data.

    Args:
        path (str): path to the compressed 3d object file
    """

    def __init__(self, path):
        if False:
            print('Hello World!')
        self.path = path
        self.data = self._parse_3d_data(path)
        (self.dimensions_names, self.dimensions_names_to_dtype) = self._parse_dimensions_names()
        self.headers = []
        self.meta_data = self._parse_meta_data()

    @abstractmethod
    def _parse_3d_data(self, path):
        if False:
            return 10
        raise NotImplementedError('PointCloudBase._parse_3d_data is not implemented')

    @abstractmethod
    def _parse_dimensions_names(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('PointCloudBase._parse_dimensions_names is not implemented')

    @abstractmethod
    def _parse_meta_data(self):
        if False:
            return 10
        raise NotImplementedError('PointCloudBase._parse_meta_data is not implemented')

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.data)