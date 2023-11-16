import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class GoEmotionsLoader(DatasetLoader):

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            print('Hello World!')
        processed_df = super().transform_dataframe(dataframe)
        processed_df['emotion_ids'] = processed_df['emotion_ids'].apply(lambda e_id: ' '.join(e_id.split(',')))
        return processed_df