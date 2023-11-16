from opensfm.large import metadataset
from opensfm.large import tools
from opensfm.dataset import DataSet

def run_dataset(data: DataSet) -> None:
    if False:
        for i in range(10):
            print('nop')
    ' Align submodel reconstructions for of MetaDataSet. '
    meta_data = metadataset.MetaDataSet(data.data_path)
    reconstruction_shots = tools.load_reconstruction_shots(meta_data)
    transformations = tools.align_reconstructions(reconstruction_shots, tools.partial_reconstruction_name, True)
    tools.apply_transformations(transformations)