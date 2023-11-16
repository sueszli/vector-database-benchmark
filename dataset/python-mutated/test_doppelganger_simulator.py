import tempfile
import numpy as np
import pytest
from bigdl.chronos.utils import LazyImport
DPGANSimulator = LazyImport('bigdl.chronos.simulator.DPGANSimulator')
Output = LazyImport('bigdl.chronos.simulator.doppelganger.output.Output')
OutputType = LazyImport('bigdl.chronos.simulator.doppelganger.output.OutputType')
Normalization = LazyImport('bigdl.chronos.simulator.doppelganger.output.Normalization')
from unittest import TestCase
from .. import op_torch

def get_train_data():
    if False:
        i = 10
        return i + 15
    import os
    import io
    import shutil
    import urllib.request as req
    dfp = f"{os.getenv('FTP_URI')}/analytics-zoo-data/apps/doppelGANger_data/data_train_small.npz"
    fi = io.BytesIO()
    with req.urlopen(dfp) as dp:
        shutil.copyfileobj(dp, fi)
        fi.seek(0)
        df = np.load(fi)
    return df

@op_torch
class TestDoppelganer(TestCase):

    def setup_method(self, method):
        if False:
            print('Hello World!')
        pass

    def teardown_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_output_value(self):
        if False:
            print('Hello World!')
        attribute_outputs = [Output(type_=OutputType.DISCRETE, dim=2), Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE)]
        assert set([val.type_.value for val in attribute_outputs]) == set([val.type_.name for val in attribute_outputs])
        with pytest.raises(Exception):
            [Output(type_=OutputType.CONTINUOUS, dim=2, normalization=None)]

    def test_init_doppelganer(self):
        if False:
            print('Hello World!')
        with get_train_data() as df:
            feature_outputs = [Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE)]
            attribute_outputs = [Output(type_=OutputType.DISCRETE, dim=9), Output(type_=OutputType.DISCRETE, dim=3), Output(type_=OutputType.DISCRETE, dim=2)]
            doppelganger = DPGANSimulator(L_max=550, sample_len=10, feature_dim=1, num_real_attribute=3, num_threads=1)
            doppelganger.fit(data_feature=df['data_feature'], data_attribute=df['data_attribute'], data_gen_flag=df['data_gen_flag'], feature_outputs=feature_outputs, attribute_outputs=attribute_outputs, epoch=2, batch_size=32)
            (feature, attribute, gen_flags, lengths) = doppelganger.generate()
            assert feature.shape == (1, doppelganger.L_max, 1)
            assert attribute.shape == (1, df['data_attribute'].shape[-1])
            assert gen_flags.shape == (1, doppelganger.L_max) and (gen_flags[0, :] == 1).all()
            assert lengths[0] == doppelganger.L_max
            with tempfile.TemporaryDirectory() as tf:
                doppelganger.save(tf)
                doppelganger.load(tf)
        with get_train_data() as df:
            feature_outputs = [Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE)]
            attribute_outputs = [Output(type_=OutputType.DISCRETE, dim=9), Output(type_=OutputType.DISCRETE, dim=3), Output(type_=OutputType.DISCRETE, dim=2)]
            doppelganger = DPGANSimulator(L_max=551, sample_len=10, feature_dim=1, num_real_attribute=3, num_threads=1)
            with pytest.raises(RuntimeError):
                doppelganger.fit(data_feature=df['data_feature'], data_attribute=df['data_attribute'], data_gen_flag=df['data_gen_flag'], feature_outputs=feature_outputs, attribute_outputs=attribute_outputs)