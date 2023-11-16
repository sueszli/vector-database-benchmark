"""Custom formatting functions for Alpha158 dataset.

Defines dataset specific column definitions and data transformations.
"""
import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing
GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

class Alpha158Formatter(GenericDataFormatter):
    """Defines and formats data for the Alpha158 dataset.

    Attributes:
      column_definition: Defines input and data type of column used in the
        experiment.
      identifiers: Entity identifiers used in experiments.
    """
    _column_definition = [('instrument', DataTypes.CATEGORICAL, InputTypes.ID), ('LABEL0', DataTypes.REAL_VALUED, InputTypes.TARGET), ('date', DataTypes.DATE, InputTypes.TIME), ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT), ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT), ('RESI5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('WVMA5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('RSQR5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('KLEN', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('RSQR10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('CORR5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('CORD5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('CORR10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('ROC60', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('RESI10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('VSTD5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('RSQR60', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('CORR60', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('WVMA60', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('STD5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('RSQR20', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('CORD60', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('CORD10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('CORR20', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('KLOW', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), ('const', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)]

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialises formatter.'
        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def split_data(self, df, valid_boundary=2016, test_boundary=2018):
        if False:
            return 10
        'Splits data frame into training-validation-test data frames.\n\n        This also calibrates scaling object, and transforms data for each split.\n\n        Args:\n          df: Source data frame to split.\n          valid_boundary: Starting year for validation data\n          test_boundary: Starting year for test data\n\n        Returns:\n          Tuple of transformed (train, valid, test) data.\n        '
        print('Formatting train-valid-test splits.')
        index = df['year']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
        test = df.loc[index >= test_boundary]
        self.set_scalers(train)
        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        if False:
            i = 10
            return i + 15
        'Calibrates scalers using the data supplied.\n\n        Args:\n          df: Data to use to calibrate scalers.\n        '
        print('Setting scalers with training data...')
        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)
        self.identifiers = list(df[id_column].unique())
        real_inputs = utils.extract_cols_from_data_type(DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME})
        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(df[[target_column]].values)
        categorical_inputs = utils.extract_cols_from_data_type(DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME})
        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        if False:
            print('Hello World!')
        'Performs feature transformations.\n\n        This includes both feature engineering, preprocessing and normalisation.\n\n        Args:\n          df: Data frame to transform.\n\n        Returns:\n          Transformed data frame.\n\n        '
        output = df.copy()
        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')
        column_definitions = self.get_column_definition()
        real_inputs = utils.extract_cols_from_data_type(DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME})
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)
        return output

    def format_predictions(self, predictions):
        if False:
            i = 10
            return i + 15
        'Reverts any normalisation to give predictions in original scale.\n\n        Args:\n          predictions: Dataframe of model predictions.\n\n        Returns:\n          Data frame of unnormalised predictions.\n        '
        output = predictions.copy()
        column_names = predictions.columns
        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(predictions[[col]])
        return output

    def get_fixed_params(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns fixed model parameters for experiments.'
        fixed_params = {'total_time_steps': 6 + 6, 'num_encoder_steps': 6, 'num_epochs': 100, 'early_stopping_patience': 10, 'multiprocessing_workers': 5}
        return fixed_params

    def get_default_model_params(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns default optimised model parameters.'
        model_params = {'dropout_rate': 0.4, 'hidden_layer_size': 160, 'learning_rate': 0.0001, 'minibatch_size': 128, 'max_gradient_norm': 0.0135, 'num_heads': 1, 'stack_size': 1}
        return model_params