import numpy as np
import pandas as pd


def load_data(data_folder_path):

    urm = pd.read_csv(data_folder_path + "/data_train.csv",
                      dtype={"row": np.int32,
                             "col": np.int32,
                             "data": np.float32})

    icm_channel = pd.read_csv(data_folder_path + "/data_ICM_channel.csv",
                              dtype={"row": np.int32,
                                     "col": np.int32,
                                     "data": np.float32})

    icm_event = pd.read_csv(data_folder_path + "/data_ICM_event.csv",
                            dtype={"row": np.int32,
                                   "col": np.int32,
                                   "data": np.float32})

    icm_genre = pd.read_csv(data_folder_path + "/data_ICM_genre.csv",
                            dtype={"row": np.int32,
                                   "col": np.int32,
                                   "data": np.float32})

    icm_subgenre = pd.read_csv(data_folder_path + "/data_ICM_subgenre.csv",
                               dtype={"row": np.int32,
                                      "col": np.int32,
                                      "data": np.float32})

    return urm, icm_channel, icm_event, icm_genre, icm_subgenre
