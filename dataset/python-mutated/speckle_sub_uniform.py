import numpy as np
import matplotlib.pyplot as plt
from rich import print
import os
from PIL import Image
from speckleAnalysis.speckleFolderAnalysis import *
bin1_speck_path = '/mnt/c/Users/legen/OneDrive - USherbrooke/Été 2023/' + 'Stage T3/dcclab_t3stage_cervo_HiLo/' + 'data_analysis/week2/20230511/data/binning1_speckle/'
bin1_unif_path = '/mnt/c/Users/legen/OneDrive - USherbrooke/Été 2023/Stage T3/' + 'dcclab_t3stage_cervo_HiLo/data_analysis/' + 'week2/20230511/data/binning1_uniform/'
bin1_sub_directory = '/mnt/c/Users/legen/OneDrive - USherbrooke/Été 2023/' + 'Stage T3/dcclab_t3stage_cervo_HiLo/data_analysis/' + 'week2/20230511/data/binning1_substract'
bin2_speck_path = '/mnt/c/Users/legen/OneDrive - USherbrooke/Été 2023/' + 'Stage T3/dcclab_t3stage_cervo_HiLo/' + 'data_analysis/week2/20230511/data/binning2_speckle/'
bin2_unif_path = '/mnt/c/Users/legen/OneDrive - USherbrooke/Été 2023/Stage T3/' + 'dcclab_t3stage_cervo_HiLo/data_analysis/' + 'week2/20230511/data/binning2_uniform/'
bin2_sub_directory = '/mnt/c/Users/legen/OneDrive - USherbrooke/Été 2023/' + 'Stage T3/dcclab_t3stage_cervo_HiLo/data_analysis/' + 'week2/20230511/data/binning2_substract'
bin4_speck_path = '/mnt/c/Users/legen/OneDrive - USherbrooke/Été 2023/' + 'Stage T3/dcclab_t3stage_cervo_HiLo/' + 'data_analysis/week2/20230511/data/binning4_speckle/'
bin4_unif_path = '/mnt/c/Users/legen/OneDrive - USherbrooke/Été 2023/Stage T3/' + 'dcclab_t3stage_cervo_HiLo/data_analysis/' + 'week2/20230511/data/binning4_uniform/'
bin4_sub_directory = '/mnt/c/Users/legen/OneDrive - USherbrooke/Été 2023/' + 'Stage T3/dcclab_t3stage_cervo_HiLo/data_analysis/' + 'week2/20230511/data/binning4_substract'

def tiff_to_im(filepath: str) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'DOCS\n    '
    filepath_dir = os.listdir(filepath)
    im_to_pix = [np.array(Image.open(filepath + i)).astype('int16') for i in sorted(filepath_dir, key=str.__len__)]
    return np.array(im_to_pix)

def substracted_im_to_tiff(speckle_list: np.array, uniform_list: np.array) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'DOCS\n    '
    neg_value_pix = np.array([np.subtract(i, j) for (i, j) in zip(speckle_list, uniform_list)])
    pos_value_pix = np.array([i + np.abs(np.min(i)) for i in neg_value_pix])
    return pos_value_pix

def build_tiff_folder(list_of_pix: np.ndarray, folderpath: str) -> None:
    if False:
        while True:
            i = 10
    'DOCS\n    '
    if os.path.isdir(folderpath) == True:
        print('The directory already exists')
    else:
        os.mkdir(folderpath)
    for (idx, i) in enumerate(list_of_pix):
        temp_im = Image.fromarray(i, mode='I;16')
        temp_im.save(folderpath + f'/substracted_im_{idx}.tiff', 'TIFF')
    return

def get_speckle_diam(speckel_path: str, uniform_path: str, substraction_folder: str) -> None:
    if False:
        return 10
    'DOCS\n    '
    speck_im = tiff_to_im(speckel_path)
    unif_im = tiff_to_im(uniform_path)
    substracted_im = substracted_im_to_tiff(speck_im, unif_im)
    build_tiff_folder(substracted_im, substraction_folder)
    speckleInfo = SpeckleFolderCaracterizations(substraction_folder, cropAroundCenter=(300, 300), gaussianFilterNormalizationStdDev=75, medianFilterSize=0)
    speckleInfo.allDataToCSV(substraction_folder + '_speckles_info.csv', averageRange=0.3)
    return
if __name__ == '__main__':
    pass