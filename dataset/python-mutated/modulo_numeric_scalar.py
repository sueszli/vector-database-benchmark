from woodwork.column_schema import ColumnSchema
from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library

class ModuloNumericScalar(TransformPrimitive):
    """Computes the modulo of each element in the list by a given scalar.

    Description:
        Given a list of numeric values and a scalar, return
        the modulo, or remainder of each value after being
        divided by the scalar.

    Examples:
        >>> modulo_numeric_scalar = ModuloNumericScalar(value=2)
        >>> modulo_numeric_scalar([3, 1, 2]).tolist()
        [1, 1, 0]
    """
    name = 'modulo_numeric_scalar'
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, value=1):
        if False:
            i = 10
            return i + 15
        self.value = value
        self.description_template = 'the remainder after dividing {{}} by {}'.format(self.value)

    def get_function(self):
        if False:
            print('Hello World!')

        def modulo_scalar(vals):
            if False:
                for i in range(10):
                    print('nop')
            return vals % self.value
        return modulo_scalar

    def generate_name(self, base_feature_names):
        if False:
            print('Hello World!')
        return '%s %% %s' % (base_feature_names[0], str(self.value))