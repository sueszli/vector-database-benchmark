import os
from opensfm import dataset
from opensfm import dense
from opensfm.dataset import DataSet

def run_dataset(data: DataSet, subfolder, interactive) -> None:
    if False:
        i = 10
        return i + 15
    "Compute depthmap on a dataset with has SfM ran already.\n\n    Args:\n        subfolder: dataset's subfolder where to store results\n        interactive : display plot of computed depthmaps\n\n    "
    udata_path = os.path.join(data.data_path, subfolder)
    udataset = dataset.UndistortedDataSet(data, udata_path, io_handler=data.io_handler)
    udataset.config['interactive'] = interactive
    reconstructions = udataset.load_undistorted_reconstruction()
    tracks_manager = udataset.load_undistorted_tracks_manager()
    dense.compute_depthmaps(udataset, tracks_manager, reconstructions[0])