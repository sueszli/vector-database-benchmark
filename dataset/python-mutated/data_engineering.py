from typing import Any, Dict, List
import pandas as pd
from prefect import Flow, Parameter, task
from prefect.engine.results import LocalResult
from sklearn.model_selection import train_test_split

@task
def load_data(path: str) -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    return pd.read_csv(path)

@task(target='{date:%a_%b_%d_%Y_%H-%M-%S}/{task_name}_output', result=LocalResult(dir='data/processed'))
def get_classes(data: pd.DataFrame, target_col: str) -> List[str]:
    if False:
        print('Hello World!')
    'Task for getting the classes from the Iris data set.'
    return sorted(data[target_col].unique())

@task
def encode_categorical_columns(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if False:
        while True:
            i = 10
    'Task for encoding the categorical columns in the Iris data set.'
    return pd.get_dummies(data, columns=[target_col], prefix='', prefix_sep='')

@task(log_stdout=True, target='{date:%a_%b_%d_%Y_%H-%M-%S}/{task_name}_output', result=LocalResult(dir='data/processed'))
def split_data(data: pd.DataFrame, test_data_ratio: float, classes: list) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    'Task for splitting the classical Iris data set into training and test\n    sets, each split into features and labels.\n    '
    print(f'Splitting data into training and test sets with ratio {test_data_ratio}')
    (X, y) = (data.drop(columns=classes), data[classes])
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_data_ratio)
    return dict(train_x=X_train, train_y=y_train, test_x=X_test, test_y=y_test)
with Flow('data-engineer') as flow:
    target_col = 'species'
    test_data_ratio = Parameter('test_data_ratio', default=0.2)
    data = load_data(path='data/raw/iris.csv')
    classes = get_classes(data=data, target_col=target_col)
    categorical_columns = encode_categorical_columns(data=data, target_col=target_col)
    train_test_dict = split_data(data=categorical_columns, test_data_ratio=test_data_ratio, classes=classes)
flow.run()