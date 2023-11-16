from bigdl.chronos.autots.deprecated.feature.utils import save_config
from bigdl.chronos.autots.deprecated.feature.abstract import BaseFeatureTransformer

class IdentityTransformer(BaseFeatureTransformer):
    """
    echo transformer
    """

    def __init__(self, feature_cols=None, target_col=None):
        if False:
            return 10
        self.feature_cols = feature_cols
        self.target_col = target_col

    def fit_transform(self, input_df, **config):
        if False:
            print('Hello World!')
        train_x = input_df[self.feature_cols]
        train_y = input_df[[self.target_col]]
        return (train_x, train_y)

    def transform(self, input_df, is_train=True):
        if False:
            return 10
        train_x = input_df[self.feature_cols]
        train_y = input_df[[self.target_col]]
        return (train_x, train_y)

    def save(self, file_path, replace=False):
        if False:
            print('Hello World!')
        data_to_save = {'feature_cols': self.feature_cols, 'target_col': self.target_col}
        save_config(file_path, data_to_save, replace=replace)

    def restore(self, **config):
        if False:
            while True:
                i = 10
        self.feature_cols = config['feature_cols']
        self.target_col = config['target_col']

    def _get_required_parameters(self):
        if False:
            print('Hello World!')
        return set()

    def _get_optional_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        return set()

    def post_processing(self, input_df, y_pred, is_train):
        if False:
            while True:
                i = 10
        if is_train:
            return (input_df[[self.target_col]], y_pred)
        else:
            return y_pred