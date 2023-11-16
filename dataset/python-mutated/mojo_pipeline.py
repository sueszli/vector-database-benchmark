"""
H2O MOJO Pipeline.

:copyright: (c) 2018 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import h2o
from h2o.expr import ExprNode
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type
__all__ = ('H2OMojoPipeline',)

class H2OMojoPipeline(object):
    """
    Representation of a MOJO Pipeline. This is currently an experimental feature.
    """

    def __init__(self, mojo_path=None):
        if False:
            while True:
                i = 10
        '\n        Create a new H2OMojoPipeline object.\n\n        :param mojo_path path to a MOJO file.\n        '
        assert_is_type(mojo_path, str)
        self.pipeline_id = h2o.lazy_import(mojo_path)

    def transform(self, data, allow_timestamps=False):
        if False:
            while True:
                i = 10
        '\n        Transform H2OFrame using a MOJO Pipeline.\n\n        :param data: Frame to be transformed.\n        :param allow_timestamps: Allows datetime columns to be used directly with MOJO pipelines. It is recommended\n        to parse your datetime columns as Strings when using pipelines because pipelines can interpret certain datetime\n        formats in a different way. If your H2OFrame is parsed from a binary file format (eg. Parquet) instead of CSV\n        it is safe to turn this option on and use datetime columns directly.\n\n        :returns: A new H2OFrame.\n        '
        assert_is_type(data, H2OFrame)
        assert_is_type(allow_timestamps, bool)
        return H2OFrame._expr(ExprNode('mojo.pipeline.transform', self.pipeline_id[0], data, allow_timestamps))

    @staticmethod
    def available():
        if False:
            while True:
                i = 10
        '\n        Returns True if a MOJO Pipelines can be used, or False otherwise.\n        '
        if 'MojoPipeline' not in h2o.cluster().list_core_extensions():
            print('Cannot use MOJO Pipelines - runtime was not found.')
            return False
        else:
            return True