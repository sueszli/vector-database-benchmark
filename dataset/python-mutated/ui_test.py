import vaex
import vaex.ml
import vaex.datasets
import vaex.ml.ui

def test_widgetize():
    if False:
        while True:
            i = 10
    ds = vaex.datasets.iris()
    transformer_list = [vaex.ml.StandardScaler(), vaex.ml.MinMaxScaler(), vaex.ml.LabelEncoder(), vaex.ml.OneHotEncoder(), vaex.ml.MaxAbsScaler(), vaex.ml.RobustScaler()]
    for (i, v) in enumerate(transformer_list):
        vaex.ml.ui.Widgetize(v, ds)