from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class ForestCoverLoader(DatasetLoader):

    def __init__(self, config: DatasetConfig, cache_dir: Optional[str]=None, use_tabnet_split=True):
        if False:
            return 10
        super().__init__(config, cache_dir=cache_dir)
        self.use_tabnet_split = use_tabnet_split

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        df = super().transform_dataframe(dataframe)
        st_cols = ['Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39', 'Soil_Type_40']
        st_vals = []
        for (_, row) in df[st_cols].iterrows():
            st_vals.append(row.to_numpy().nonzero()[0].item(0))
        df = df.drop(columns=st_cols)
        df['Soil_Type'] = st_vals
        wa_cols = ['Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4']
        wa_vals = []
        for (_, row) in df[wa_cols].iterrows():
            wa_vals.append(row.to_numpy().nonzero()[0].item(0))
        df = df.drop(columns=wa_cols)
        df['Wilderness_Area'] = wa_vals
        if not self.use_tabnet_split:
            df['split'] = [0] * 11340 + [1] * 3780 + [2] * 565892
        else:
            (train_val_indices, test_indices) = train_test_split(range(len(df)), test_size=0.2, random_state=0)
            (train_indices, val_indices) = train_test_split(train_val_indices, test_size=0.2 / 0.6, random_state=0)
            df['split'] = 0
            df.loc[val_indices, 'split'] = 1
            df.loc[test_indices, 'split'] = 2
        return df