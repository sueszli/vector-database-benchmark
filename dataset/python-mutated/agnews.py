import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class AGNewsLoader(DatasetLoader):

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            print('Hello World!')
        processed_df = super().transform_dataframe(dataframe)
        class_names = ['', 'world', 'sports', 'business', 'sci_tech']
        processed_df['class'] = processed_df.class_index.apply(lambda i: class_names[i])
        val_set_n = int(len(processed_df) * 0.05 // len(class_names))
        for ci in range(1, 5):
            train_rows = processed_df[(processed_df.split == 0) & (processed_df.class_index == ci)].index
            processed_df.loc[train_rows[:val_set_n], 'split'] = 1
        return processed_df