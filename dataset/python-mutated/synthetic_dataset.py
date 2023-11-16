import collections
import logging
import os
import shelve
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from opensfm import tracking, features as oft, types, pymap, pygeometry, io, geo
from opensfm.dataset import DataSet
logger: logging.Logger = logging.getLogger(__name__)

class SyntheticFeatures(collections.abc.MutableMapping):
    database: Union[Dict[str, oft.FeaturesData], shelve.Shelf]

    def __init__(self, on_disk_filename: Optional[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if on_disk_filename:
            self.database = shelve.open(on_disk_filename, flag='n')
        else:
            self.database = {}
        for m in ['keys', 'items', 'values', 'get']:
            setattr(self, m, getattr(self.database, m))

    def sync(self) -> None:
        if False:
            print('Hello World!')
        database = self.database
        if type(database) is dict:
            return
        else:
            database.sync()

    def __getitem__(self, key) -> oft.FeaturesData:
        if False:
            print('Hello World!')
        return self.database.__getitem__(key)

    def __setitem__(self, key, item) -> None:
        if False:
            print('Hello World!')
        return self.database.__setitem__(key, item)

    def __delitem__(self, key) -> None:
        if False:
            return 10
        return self.database.__delitem__(key)

    def __iter__(self) -> Iterator[str]:
        if False:
            return 10
        return self.database.__iter__()

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return self.database.__len__()

class SyntheticDataSet(DataSet):
    reconstruction: types.Reconstruction
    exifs: Dict[str, Any]
    features: Optional[SyntheticFeatures]
    reference: geo.TopocentricConverter
    gcps: Optional[Dict[str, pymap.GroundControlPoint]]

    def __init__(self, reconstruction: types.Reconstruction, exifs: Dict[str, Any], features: Optional[SyntheticFeatures]=None, tracks_manager: Optional[pymap.TracksManager]=None, gcps: Optional[Dict[str, pymap.GroundControlPoint]]=None, output_path: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        data_path = '' if not output_path else output_path
        if data_path:
            io.mkdir_p(data_path)
            io.mkdir_p(os.path.join(data_path, 'images'))
        super(SyntheticDataSet, self).__init__(data_path)
        self.reconstruction = reconstruction
        self.exifs = exifs
        self.gcps = gcps
        self.features = features
        self.tracks_manager = tracks_manager
        self.image_list = list(reconstruction.shots.keys())
        self.reference = reconstruction.reference
        self.matches = None
        self.config['use_altitude_tag'] = True
        self.config['align_method'] = 'naive'

    def images(self) -> List[str]:
        if False:
            while True:
                i = 10
        return self.image_list

    def _raise_if_absent_image(self, image: str):
        if False:
            for i in range(10):
                print('nop')
        if image not in self.image_list:
            raise RuntimeError("Image isn't present in the synthetic dataset")

    def load_camera_models(self) -> Dict[str, pygeometry.Camera]:
        if False:
            i = 10
            return i + 15
        return self.reconstruction.cameras

    def save_camera_models(self, camera_models: Dict[str, pygeometry.Camera]) -> None:
        if False:
            print('Hello World!')
        for camera in camera_models.values():
            self.reconstruction.add_camera(camera)

    def load_rig_cameras(self) -> Dict[str, pymap.RigCamera]:
        if False:
            for i in range(10):
                print('nop')
        return self.reconstruction.rig_cameras

    def load_rig_assignments(self) -> Dict[str, List[Tuple[str, str]]]:
        if False:
            print('Hello World!')
        rig_assignments = {}
        for instance in self.reconstruction.rig_instances.values():
            rig_assignments[instance.id] = [(k, v.id) for (k, v) in instance.rig_cameras.items()]
        return rig_assignments

    def load_exif(self, image: str) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        self._raise_if_absent_image(image)
        return self.exifs[image]

    def exif_exists(self, image: str) -> bool:
        if False:
            while True:
                i = 10
        return image in self.image_list

    def features_exist(self, image: str) -> bool:
        if False:
            i = 10
            return i + 15
        if image not in self.image_list:
            return False
        if self.features is None:
            return False
        feat = self.features
        if feat is None:
            return False
        return image in feat

    def load_words(self, image: str):
        if False:
            print('Hello World!')
        self._raise_if_absent_image(image)
        n_closest = 50
        return [image] * n_closest

    def load_features(self, image: str) -> Optional[oft.FeaturesData]:
        if False:
            i = 10
            return i + 15
        self._raise_if_absent_image(image)
        if not self.features:
            return None
        feat = self.features
        if feat is None:
            return None
        return feat[image]

    def save_features(self, image: str, features_data: oft.FeaturesData) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def matches_exists(self, image: str) -> bool:
        if False:
            return 10
        if image not in self.image_list:
            return False
        self._check_and_create_matches()
        if self.matches is None:
            return False
        return True

    def load_matches(self, image: str) -> Dict[str, np.ndarray]:
        if False:
            return 10
        self._raise_if_absent_image(image)
        self._check_and_create_matches()
        if self.matches is not None:
            return self.matches[image]
        else:
            return {}

    def load_image_list(self) -> None:
        if False:
            return 10
        pass

    def _check_and_create_matches(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.matches is None:
            self.matches = self._construct_matches()

    def _construct_matches(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        matches = {}
        tracks_manager = self.load_tracks_manager()
        for im1 in self.images():
            for im2 in self.images():
                if im1 == im2:
                    continue
                image_matches = matches.setdefault(im1, {})
                tracks = tracking.common_tracks(tracks_manager, im1, im2)[0]
                if len(tracks) > 10:
                    pair_matches = []
                    for t in tracks:
                        observations = tracks_manager.get_track_observations(t)
                        pair_matches.append(np.array([observations[im1].id, observations[im2].id]))
                    image_matches[im2] = np.array(pair_matches)
        return matches

    def load_tracks_manager(self, filename: Optional[str]=None) -> pymap.TracksManager:
        if False:
            while True:
                i = 10
        tracks_mgr = self.tracks_manager
        if not tracks_mgr:
            raise RuntimeError('No tracks manager for the synthetic dataset')
        return tracks_mgr

    def init_reference(self, images: Optional[List[str]]=None) -> None:
        if False:
            while True:
                i = 10
        pass

    def load_reference(self) -> geo.TopocentricConverter:
        if False:
            return 10
        return self.reference

    def reference_exists(self) -> bool:
        if False:
            return 10
        return True

    def load_ground_control_points(self) -> List[pymap.GroundControlPoint]:
        if False:
            return 10
        if self.gcps:
            return list(self.gcps.values())
        else:
            return []