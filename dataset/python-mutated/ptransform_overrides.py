"""Ptransform overrides for DataflowRunner."""
from apache_beam.pipeline import PTransformOverride

class NativeReadPTransformOverride(PTransformOverride):
    """A ``PTransformOverride`` for ``Read`` using native sources.

  The DataflowRunner expects that the Read PTransform using native sources act
  as a primitive. So this override replaces the Read with a primitive.
  """

    def matches(self, applied_ptransform):
        if False:
            i = 10
            return i + 15
        from apache_beam.io import Read
        return isinstance(applied_ptransform.transform, Read) and (not getattr(applied_ptransform.transform, 'override', False)) and hasattr(applied_ptransform.transform.source, 'format')

    def get_replacement_transform(self, ptransform):
        if False:
            print('Hello World!')
        from apache_beam import pvalue
        from apache_beam.io import iobase

        class Read(iobase.Read):
            override = True

            def expand(self, pbegin):
                if False:
                    i = 10
                    return i + 15
                return pvalue.PCollection.from_(pbegin)
        return Read(ptransform.source).with_output_types(ptransform.source.coder.to_type_hint())