import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, IO, Any
import numpy as np
from opensfm import features, geo, io, pygeometry, types, pymap
logger: logging.Logger = logging.getLogger(__name__)

class DataSetBase(ABC):
    """Base for dataset classes providing i/o access to persistent data.

    It is possible to store data remotely or in different formats
    by subclassing this class and overloading its methods.
    """

    @property
    @abstractmethod
    def io_handler(self) -> io.IoFilesystemBase:
        if False:
            while True:
                i = 10
        pass

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def images(self) -> List[str]:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def open_image_file(self, image: str) -> IO[Any]:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def load_image(self, image: str, unchanged: bool=False, anydepth: bool=False, grayscale: bool=False) -> np.ndarray:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def image_size(self, image: str) -> Tuple[int, int]:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def load_mask(self, image: str) -> Optional[np.ndarray]:
        if False:
            return 10
        pass

    @abstractmethod
    def load_instances(self, image: str) -> Optional[np.ndarray]:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def segmentation_labels(self) -> List[Any]:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def load_segmentation(self, image: str) -> Optional[np.ndarray]:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def segmentation_ignore_values(self, image: str) -> List[int]:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def undistorted_segmentation_ignore_values(self, image: str) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def load_exif(self, image: str) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def save_exif(self, image: str, data: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def exif_exists(self, image: str) -> bool:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def feature_type(self) -> str:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def features_exist(self, image: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def load_features(self, image: str) -> Optional[features.FeaturesData]:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def save_features(self, image: str, features_data: features.FeaturesData) -> None:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def words_exist(self, image: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def load_words(self, image: str) -> np.ndarray:
        if False:
            return 10
        pass

    @abstractmethod
    def save_words(self, image: str, words: np.ndarray) -> None:
        if False:
            return 10
        pass

    @abstractmethod
    def matches_exists(self, image: str) -> bool:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def load_matches(self, image: str) -> Dict[str, np.ndarray]:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def save_matches(self, image: str, matches: Dict[str, np.ndarray]) -> None:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def load_tracks_manager(self, filename: Optional[str]=None) -> pymap.TracksManager:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def save_tracks_manager(self, tracks_manager: pymap.TracksManager, filename: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def load_reconstruction(self, filename: Optional[str]=None) -> List[types.Reconstruction]:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def save_reconstruction(self, reconstruction: List[types.Reconstruction], filename: Optional[str]=None, minify=False) -> None:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def init_reference(self, images: Optional[List[str]]=None) -> None:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def reference_exists(self) -> bool:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def load_reference(self) -> geo.TopocentricConverter:
        if False:
            return 10
        pass

    @abstractmethod
    def save_reference(self, reference: geo.TopocentricConverter) -> None:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def load_camera_models(self) -> Dict[str, pygeometry.Camera]:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def save_camera_models(self, camera_models: Dict[str, pygeometry.Camera]) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def camera_models_overrides_exists(self) -> bool:
        if False:
            return 10
        pass

    @abstractmethod
    def load_camera_models_overrides(self) -> Dict[str, pygeometry.Camera]:
        if False:
            return 10
        pass

    @abstractmethod
    def save_camera_models_overrides(self, camera_models: Dict[str, pygeometry.Camera]) -> None:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def exif_overrides_exists(self) -> bool:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def load_exif_overrides(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def load_rig_cameras(self) -> Dict[str, pymap.RigCamera]:
        if False:
            return 10
        pass

    @abstractmethod
    def save_rig_cameras(self, rig_cameras: Dict[str, pymap.RigCamera]) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def load_rig_assignments(self) -> Dict[str, List[Tuple[str, str]]]:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def save_rig_assignments(self, rig_assignments: Dict[str, List[Tuple[str, str]]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def append_to_profile_log(self, content: str) -> None:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def load_report(self, path: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def save_report(self, report_str: str, path: str) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def load_ground_control_points(self) -> List[pymap.GroundControlPoint]:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def save_ground_control_points(self, points: List[pymap.GroundControlPoint]) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def clean_up(self) -> None:
        if False:
            return 10
        pass