"""Tests for tensorflow.python.ops.op_def_library."""
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_library
from tensorflow.python.framework import op_def_library_pybind
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.util import compat

@test_util.add_graph_building_optimization_tests
class OpDefLibraryTest(test_util.TensorFlowTestCase):

    def Tensor(self, t, name='in'):
        if False:
            print('Hello World!')
        return op_def_library.apply_op('OutT', T=t, name=name)

    def testNoRegisteredOpFails(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(RuntimeError) as cm:
            op_def_library.apply_op('unknown')
        self.assertEqual(str(cm.exception), 'Unrecognized Op name unknown')

    def testSimple(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            out = op_def_library.apply_op('Simple', a=3)
            self.assertEqual(dtypes.float32, out.dtype)
            self.assertProtoEquals("\n        name: 'Simple' op: 'Simple' input: 'Simple/a'\n        ", out.op.node_def)
            out = op_def_library.apply_op('Simple', a=4)
            self.assertProtoEquals("\n        name: 'Simple_1' op: 'Simple' input: 'Simple_1/a'\n        ", out.op.node_def)
            out = op_def_library.apply_op('Simple', a=5, name='named')
            self.assertProtoEquals("\n        name: 'named' op: 'Simple' input: 'named/a'\n        ", out.op.node_def)
            out = op_def_library.apply_op('Simple', a=[[1, 2, 3], [4, 5, 6]], name='two_d')
            self.assertProtoEquals("\n        name: 'two_d' op: 'Simple' input: 'two_d/a'\n        ", out.op.node_def)

    def testSimpleFailures(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Simple', a='Bad string')
            self.assertIn("Expected int32 passed to parameter 'a' of op 'Simple', got 'Bad string' of type 'str' instead.", str(cm.exception))
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Simple', a=self.Tensor(dtypes.string))
            self.assertIn("Input 'a' of 'Simple' Op has type string that does not match expected type of int32.", str(cm.exception))
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Simple', a=6, extra='bogus')
            self.assertIn('Simple got unexpected keyword arguments: extra', str(cm.exception))
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Simple', a=6, extra1='bogus', extra2='also_bogus')
            self.assertIn('Simple got unexpected keyword arguments: extra1, extra2', str(cm.exception))
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Simple')
            self.assertIn('No argument for input a', str(cm.exception))
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Simple', wrong=7)
            self.assertIn('No argument for input a', str(cm.exception))
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Simple', a={'label': 1})
            self.assertIn("Expected int32 passed to parameter 'a' of op 'Simple', got {'label': 1} of type 'dict' instead.", str(cm.exception))

    def testReservedInput(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            op = op_def_library.apply_op('ReservedInput', input_=7, name='x')
            self.assertProtoEquals("\n        name: 'x' op: 'ReservedInput' input: 'x/input'\n        ", op.node_def)

    def testPolymorphic(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            out = op_def_library.apply_op('Polymorphic', a=7, name='p')
            self.assertEqual(dtypes.int32, out.dtype)
            self.assertProtoEquals("\n        name: 'p' op: 'Polymorphic' input: 'p/a'\n        attr { key: 'T' value { type: DT_INT32 } }\n        ", out.op.node_def)
            out = op_def_library.apply_op('Polymorphic', a='s', name='q')
            self.assertEqual(dtypes.string, out.dtype)
            self.assertProtoEquals("\n        name: 'q' op: 'Polymorphic' input: 'q/a'\n        attr { key: 'T' value { type: DT_STRING } }\n        ", out.op.node_def)
            out = op_def_library.apply_op('Polymorphic', a=['s', 't', 'u'], name='r')
            self.assertEqual(dtypes.string, out.dtype)
            self.assertProtoEquals("\n        name: 'r' op: 'Polymorphic' input: 'r/a'\n        attr { key: 'T' value { type: DT_STRING } }\n        ", out.op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Polymorphic', a='s', T=dtypes.string)
            self.assertEqual(str(cm.exception), "Should not specify value for inferred attr 'T' for Polymorphic.")

    def testPolymorphicOut(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            out = op_def_library.apply_op('PolymorphicOut', T=dtypes.int32, name='p')
            self.assertEqual(dtypes.int32, out.dtype)
            self.assertProtoEquals("\n        name: 'p' op: 'PolymorphicOut'\n        attr { key: 'T' value { type: DT_INT32 } }\n        ", out.op.node_def)
            out = op_def_library.apply_op('PolymorphicOut', T=dtypes.bool, name='q')
            self.assertEqual(dtypes.bool, out.dtype)
            self.assertProtoEquals("\n        name: 'q' op: 'PolymorphicOut'\n        attr { key: 'T' value { type: DT_BOOL } }\n        ", out.op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('PolymorphicOut')
            self.assertEqual(str(cm.exception), 'No argument found for attr T for PolymorphicOut')
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('PolymorphicOut', T=None)
            self.assertEqual(str(cm.exception), "Expected DataType for argument 'T' not None.")

    def testPolymorphicDefaultOut(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            out = op_def_library.apply_op('PolymorphicDefaultOut', T=None, name='p')
            self.assertEqual(dtypes.string, out.dtype)
            self.assertProtoEquals("\n        name: 'p' op: 'PolymorphicDefaultOut'\n        attr { key: 'T' value { type: DT_STRING } }\n        ", out.op.node_def)
            out = op_def_library.apply_op('PolymorphicDefaultOut', T=dtypes.bool, name='q')
            self.assertEqual(dtypes.bool, out.dtype)
            self.assertProtoEquals("\n        name: 'q' op: 'PolymorphicDefaultOut'\n        attr { key: 'T' value { type: DT_BOOL } }\n        ", out.op.node_def)

    def testBinary(self):
        if False:
            return 10
        with ops.Graph().as_default():
            out = op_def_library.apply_op('Binary', a=8, b=9, name='b')
            self.assertEqual(dtypes.int32, out.dtype)
            self.assertProtoEquals("\n        name: 'b' op: 'Binary' input: 'b/a' input: 'b/b'\n        attr { key: 'T' value { type: DT_INT32 } }\n        ", out.op.node_def)
            out = op_def_library.apply_op('Binary', a='left', b='right', name='c')
            self.assertEqual(dtypes.string, out.dtype)
            self.assertProtoEquals("\n        name: 'c' op: 'Binary' input: 'c/a' input: 'c/b'\n        attr { key: 'T' value { type: DT_STRING } }\n        ", out.op.node_def)
            with self.assertRaises(TypeError):
                op_def_library.apply_op('Binary', a='left', b=12)
            with self.assertRaises(TypeError):
                op_def_library.apply_op('Binary', a=self.Tensor(dtypes.string), b=self.Tensor(dtypes.int32))

    def testRestrict(self):
        if False:
            return 10
        with ops.Graph().as_default():
            out = op_def_library.apply_op('Restrict', a='foo', name='g')
            self.assertEqual(dtypes.string, out.dtype)
            self.assertProtoEquals("\n        name: 'g' op: 'Restrict' input: 'g/a'\n        attr { key: 'T' value { type: DT_STRING } }\n        ", out.op.node_def)
            out = op_def_library.apply_op('Restrict', a=True, name='h')
            self.assertEqual(dtypes.bool, out.dtype)
            self.assertProtoEquals("\n        name: 'h' op: 'Restrict' input: 'h/a'\n        attr { key: 'T' value { type: DT_BOOL } }\n        ", out.op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Restrict', a=17)
            self.assertEqual(str(cm.exception), "Value passed to parameter 'a' has DataType int32 not in list of allowed values: string, bool")

    def testTypeList(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            op = op_def_library.apply_op('TypeList', a=['foo'], name='z')
            self.assertProtoEquals("\n        name: 'z' op: 'TypeList' input: 'z/a_0'\n        attr { key: 'T' value { list { type: DT_STRING } } }\n        ", op.node_def)
            op = op_def_library.apply_op('TypeList', a=[True, 12], name='y')
            self.assertProtoEquals("\n        name: 'y' op: 'TypeList' input: 'y/a_0' input: 'y/a_1'\n        attr { key: 'T' value { list { type: DT_BOOL type: DT_INT32 } } }\n        ", op.node_def)
            op = op_def_library.apply_op('TypeList', a=[], name='empty')
            self.assertProtoEquals("\n        name: 'empty' op: 'TypeList' attr { key: 'T' value { list { } } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('TypeList', a=17)
            self.assertStartsWith(str(cm.exception), "Expected list for 'a' argument to 'TypeList' Op, not ")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('TypeList', a=[self.Tensor(dtypes.int32), None])
            self.assertStartsWith(str(cm.exception), "Tensors in list passed to 'a' of 'TypeList' Op have types [int32, <NOT CONVERTIBLE TO TENSOR>]")

    def testTypeListTwice(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            op = op_def_library.apply_op('TypeListTwice', a=['foo', True], b=['bar', False], name='z')
            self.assertProtoEquals("\n        name: 'z' op: 'TypeListTwice'\n        input: 'z/a_0' input: 'z/a_1' input: 'z/b_0' input: 'z/b_1'\n        attr { key: 'T' value { list { type: DT_STRING type: DT_BOOL } } }\n        ", op.node_def)
            op = op_def_library.apply_op('TypeListTwice', a=[], b=[], name='empty')
            self.assertProtoEquals("\n        name: 'empty' op: 'TypeListTwice' attr { key: 'T' value { list { } } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('TypeListTwice', a=['foo', True], b=['bar', 6])
            self.assertEqual(str(cm.exception), "Input 'b' of 'TypeListTwice' Op has type list of string, int32 that does not match type list string, bool of argument 'a'.")

    def testOutTypeList(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            (out,) = op_def_library.apply_op('OutTypeList', T=[dtypes.float32], name='x')
            self.assertEqual(dtypes.float32, out.dtype)
            self.assertProtoEquals("\n        name: 'x' op: 'OutTypeList'\n        attr { key: 'T' value { list { type: DT_FLOAT } } }\n        ", out.op.node_def)
            (out1, out2) = op_def_library.apply_op('OutTypeList', T=[dtypes.int32, dtypes.bool], name='w')
            self.assertEqual(dtypes.int32, out1.dtype)
            self.assertEqual(dtypes.bool, out2.dtype)
            self.assertProtoEquals("\n        name: 'w' op: 'OutTypeList'\n        attr { key: 'T' value { list { type: DT_INT32 type: DT_BOOL } } }\n        ", out1.op.node_def)
            out = op_def_library.apply_op('OutTypeList', T=[], name='empty')
            self.assertEqual([], out)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('OutTypeList', T=dtypes.int32)
            self.assertEqual(str(cm.exception), 'Expected list for attr T, obtained DType instead.')

    def testTypeListRestrict(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            op = op_def_library.apply_op('TypeListRestrict', a=['foo', False], name='v')
            self.assertProtoEquals("\n        name: 'v' op: 'TypeListRestrict' input: 'v/a_0' input: 'v/a_1'\n        attr { key: 'T' value { list { type: DT_STRING type: DT_BOOL } } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('TypeListRestrict', a=[True, 12])
            self.assertEqual(str(cm.exception), "Value passed to parameter 'a' has DataType int32 not in list of allowed values: string, bool")

    def testOutTypeListRestrict(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            (out1, out2) = op_def_library.apply_op('OutTypeListRestrict', t=[dtypes.bool, dtypes.string], name='u')
            self.assertEqual(dtypes.bool, out1.dtype)
            self.assertEqual(dtypes.string, out2.dtype)
            self.assertProtoEquals("\n        name: 'u' op: 'OutTypeListRestrict'\n        attr { key: 't' value { list { type: DT_BOOL type: DT_STRING } } }\n        ", out1.op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('OutTypeListRestrict', t=[dtypes.string, dtypes.int32])
            self.assertEqual(str(cm.exception), "Value passed to parameter 't' has DataType int32 not in list of allowed values: string, bool")

    def testAttr(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('Attr', a=12, name='t')
            self.assertProtoEquals("\n        name: 't' op: 'Attr' attr { key: 'a' value { i: 12 } }\n        ", op.node_def)
            op = op_def_library.apply_op('Attr', a=tensor_shape.Dimension(13), name='u')
            self.assertProtoEquals("\n        name: 'u' op: 'Attr' attr { key: 'a' value { i: 13 } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Attr', a='bad')
            self.assertEqual(str(cm.exception), "Expected int for argument 'a' not 'bad'.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Attr', a=[12])
            self.assertEqual(str(cm.exception), "Expected int for argument 'a' not [12].")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Attr', a=None)
            self.assertEqual(str(cm.exception), "Expected int for argument 'a' not None.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('Attr')
            self.assertEqual(str(cm.exception), 'No argument found for attr a for Attr')

    def testAttrFloat(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrFloat', a=1.2, name='t')
            self.assertProtoEquals("\n        name: 't' op: 'AttrFloat' attr { key: 'a' value { f: 1.2 } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrFloat', a=12, name='u')
            self.assertProtoEquals("\n        name: 'u' op: 'AttrFloat' attr { key: 'a' value { f: 12 } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('AttrFloat', a='bad')
            self.assertEqual(str(cm.exception), "Expected float for argument 'a' not 'bad'.")

    def testAttrFunc(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():

            @function.Defun(dtypes.float32, func_name='MyFn')
            def fn(x):
                if False:
                    i = 10
                    return i + 15
                return 2 + x
            op = op_def_library.apply_op('FuncAttr', f=fn, name='t')
            self.assertProtoEquals("\n        name: 't' op: 'FuncAttr' attr { key: 'f'\n                                        value { func { name: 'MyFn' } } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('FuncAttr', f=3)
            self.assertEqual(str(cm.exception), "Don't know how to convert 3 to a func for argument f")

    def testAttrFuncWithFuncWithAttrs(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():

            @def_function.function(input_signature=(tensor.TensorSpec(None, dtypes.float32),), autograph=False, experimental_attributes={'_implements': 15})
            def fn(x):
                if False:
                    print('Hello World!')
                return 2 + x
            concrete_fn = fn.get_concrete_function()
            op = op_def_library.apply_op('FuncAttr', f=concrete_fn, name='t')
            self.assertEqual(15, op.node_def.attr['f'].func.attr['_implements'].i)
            self.assertEqual(compat.as_str(concrete_fn.name), op.node_def.attr['f'].func.name)

    def testAttrFuncList(self):
        if False:
            return 10
        with ops.Graph().as_default():

            @function.Defun(dtypes.float32, func_name='MyFn')
            def fn1(x):
                if False:
                    print('Hello World!')
                return 2 + x

            @function.Defun(dtypes.int32, dtypes.float32, func_name='MyFn2')
            def fn2(x, y):
                if False:
                    while True:
                        i = 10
                return (2 + x, y * 3)

            @function.Defun(dtypes.int32, func_name='MyFn3')
            def fn3(y):
                if False:
                    while True:
                        i = 10
                return 2 + y
            op = op_def_library.apply_op('FuncListAttr', f=[fn1, fn2, fn3], name='t')
            self.assertProtoEquals("\n        name: 't' op: 'FuncListAttr'\n        attr { key: 'f' value { list { func { name: 'MyFn' }\n                                       func { name: 'MyFn2' }\n                                       func { name: 'MyFn3' } } } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('FuncListAttr', f=[fn1, 3, fn2])
            self.assertEqual(str(cm.exception), "Don't know how to convert 3 to a func for argument f")

    def testAttrBool(self):
        if False:
            return 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrBool', a=True, name='t')
            self.assertProtoEquals("\n        name: 't' op: 'AttrBool' attr { key: 'a' value { b: true } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrBool', a=False, name='u')
            self.assertProtoEquals("\n        name: 'u' op: 'AttrBool' attr { key: 'a' value { b: false } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('AttrBool', a=0)
            self.assertEqual(str(cm.exception), "Expected bool for argument 'a' not 0.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('AttrBool', a=1)
            self.assertEqual(str(cm.exception), "Expected bool for argument 'a' not 1.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('AttrBool', a=[])
            self.assertEqual(str(cm.exception), "Expected bool for argument 'a' not [].")

    def testAttrBoolList(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrBoolList', a=[True, False, True], name='t')
            self.assertProtoEquals("\n        name: 't' op: 'AttrBoolList'\n        attr { key: 'a' value { list { b: true b: false b:true } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrBoolList', a=[], name='u')
            self.assertProtoEquals("\n        name: 'u' op: 'AttrBoolList' attr { key: 'a' value { list { } } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('AttrBoolList', a=[0])
            self.assertEqual(str(cm.exception), "Expected bool for argument 'a' not 0.")

    def testAttrMin(self):
        if False:
            return 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrMin', a=12, name='s')
            self.assertProtoEquals("\n        name: 's' op: 'AttrMin' attr { key: 'a' value { i: 12 } }\n        ", op.node_def)
            with self.assertRaises(ValueError) as cm:
                op_def_library.apply_op('AttrMin', a=2)
            self.assertEqual(str(cm.exception), "Attr 'a' of 'AttrMin' Op passed 2 less than minimum 5.")

    def testAttrListMin(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrListMin', a=[1, 2], name='r')
            self.assertProtoEquals("\n        name: 'r' op: 'AttrListMin'\n        attr { key: 'a' value { list { i: 1 i: 2 } } }\n        ", op.node_def)
            with self.assertRaises(ValueError) as cm:
                op_def_library.apply_op('AttrListMin', a=[17])
            self.assertEqual(str(cm.exception), "Attr 'a' of 'AttrListMin' Op passed list of length 1 less than minimum 2.")

    def testAttrEnum(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrEnum', a='oranges', name='e')
            self.assertProtoEquals("\n        name: 'e' op: 'AttrEnum' attr { key: 'a' value { s: 'oranges' } }\n        ", op.node_def)
            with self.assertRaises(ValueError) as cm:
                op_def_library.apply_op('AttrEnum', a='invalid')
            self.assertEqual(str(cm.exception), 'Attr \'a\' of \'AttrEnum\' Op passed string \'invalid\' not in: "apples", "oranges".')

    def testAttrEnumList(self):
        if False:
            return 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrEnumList', a=['oranges', 'apples'], name='f')
            self.assertProtoEquals("\n        name: 'f' op: 'AttrEnumList'\n        attr { key: 'a' value { list { s: 'oranges' s: 'apples' } } }\n        ", op.node_def)
            with self.assertRaises(ValueError) as cm:
                op_def_library.apply_op('AttrEnumList', a=['apples', 'invalid', 'oranges'])
            self.assertEqual(str(cm.exception), 'Attr \'a\' of \'AttrEnumList\' Op passed string \'invalid\' not in: "apples", "oranges".')

    def testAttrShape(self):
        if False:
            return 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrShape', a=[5], name='s1')
            self.assertProtoEquals("\n        name: 's1' op: 'AttrShape'\n        attr { key: 'a' value { shape { dim { size: 5 } } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrShape', a=(4, 3, 2), name='s2')
            self.assertProtoEquals("\n        name: 's2' op: 'AttrShape'\n        attr { key: 'a' value {\n          shape { dim { size: 4 } dim { size: 3 } dim { size: 2 } } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrShape', a=tensor_shape.TensorShape([3, 2]), name='s3')
            self.assertProtoEquals("\n        name: 's3' op: 'AttrShape'\n        attr { key: 'a' value {\n          shape { dim { size: 3 } dim { size: 2 } } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrShape', a=[], name='s4')
            self.assertProtoEquals("\n        name: 's4' op: 'AttrShape' attr { key: 'a' value { shape { } } }\n        ", op.node_def)
            shape = tensor_shape_pb2.TensorShapeProto()
            shape.dim.add().size = 6
            shape.dim.add().size = 3
            op = op_def_library.apply_op('AttrShape', a=shape, name='s5')
            self.assertProtoEquals("\n        name: 's5' op: 'AttrShape'\n        attr { key: 'a' value { shape { dim { size: 6 } dim { size: 3 } } } }\n        ", op.node_def)
            with self.assertRaises(TypeError):
                op_def_library.apply_op('AttrShape', a='ABC')

    def testAttrShapeList(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrShapeList', a=[[3, 2], [6, 5, 4]], name='sl')
            self.assertProtoEquals("\n        name: 'sl' op: 'AttrShapeList'\n        attr { key: 'a' value { list {\n          shape { dim { size: 3 } dim { size: 2 } }\n          shape { dim { size: 6 } dim { size: 5 } dim { size: 4 } } } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrShapeList', a=[], name='esl')
            self.assertProtoEquals("\n        name: 'esl' op: 'AttrShapeList' attr { key: 'a' value { list { } } }\n        ", op.node_def)

    def testAttrPartialShape(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrPartialShape', a=[5], name='s1')
            self.assertProtoEquals("\n        name: 's1' op: 'AttrPartialShape'\n        attr { key: 'a' value { shape { dim { size: 5 } } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrPartialShape', a=(4, None, 2), name='s2')
            self.assertProtoEquals("\n        name: 's2' op: 'AttrPartialShape'\n        attr { key: 'a' value {\n          shape { dim { size: 4 } dim { size: -1 } dim { size: 2 } } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrPartialShape', a=tensor_shape.TensorShape([3, None]), name='s3')
            self.assertProtoEquals("\n        name: 's3' op: 'AttrPartialShape'\n        attr { key: 'a' value {\n          shape { dim { size: 3 } dim { size: -1 } } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrPartialShape', a=[], name='s4')
            self.assertProtoEquals("\n        name: 's4' op: 'AttrPartialShape'\n        attr { key: 'a' value { shape { } } }\n        ", op.node_def)
            shape = tensor_shape_pb2.TensorShapeProto()
            shape.dim.add().size = -1
            shape.dim.add().size = 3
            op = op_def_library.apply_op('AttrPartialShape', a=shape, name='s5')
            self.assertProtoEquals("\n        name: 's5' op: 'AttrPartialShape'\n        attr { key: 'a' value {\n          shape { dim { size: -1 } dim { size: 3 } } } }\n        ", op.node_def)
            with self.assertRaises(TypeError):
                op_def_library.apply_op('AttrPartialShape', a='ABC')

    def testAttrPartialShapeList(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrPartialShapeList', a=[[3, 2], [6, None, 4]], name='sl')
            self.assertProtoEquals("\n        name: 'sl' op: 'AttrPartialShapeList'\n        attr { key: 'a' value { list {\n          shape { dim { size: 3 } dim { size: 2 } }\n          shape { dim { size: 6 } dim { size: -1 } dim { size: 4 } } } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrPartialShapeList', a=[], name='esl')
            self.assertProtoEquals("\n        name: 'esl' op: 'AttrPartialShapeList' attr {\n          key: 'a' value { list { } } }\n        ", op.node_def)

    def testAttrDefault(self):
        if False:
            return 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrDefault', a=None, name='d')
            self.assertProtoEquals("\n        name: 'd' op: 'AttrDefault' attr { key: 'a' value { s: 'banana' } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrDefault', a='kiwi', name='c')
            self.assertProtoEquals("\n        name: 'c' op: 'AttrDefault' attr { key: 'a' value { s: 'kiwi' } }\n        ", op.node_def)

    def testAttrListDefault(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrListDefault', a=None, name='b')
            self.assertProtoEquals("\n        name: 'b' op: 'AttrListDefault'\n        attr { key: 'a' value { list { i: 5 i: 15 } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrListDefault', a=[3], name='a')
            self.assertProtoEquals("\n        name: 'a' op: 'AttrListDefault'\n        attr { key: 'a' value { list { i: 3 } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrListDefault', a=[], name='empty')
            self.assertProtoEquals("\n        name: 'empty' op: 'AttrListDefault'\n        attr { key: 'a' value { list { } } }\n        ", op.node_def)

    def testAttrEmptyListDefault(self):
        if False:
            return 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrEmptyListDefault', a=None, name='b')
            self.assertProtoEquals("\n        name: 'b' op: 'AttrEmptyListDefault'\n        attr { key: 'a' value { list { } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrEmptyListDefault', a=[3], name='a')
            self.assertProtoEquals("\n        name: 'a' op: 'AttrEmptyListDefault'\n        attr { key: 'a' value { list { f: 3 } } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrEmptyListDefault', a=[], name='empty')
            self.assertProtoEquals("\n        name: 'empty' op: 'AttrEmptyListDefault'\n        attr { key: 'a' value { list { } } }\n        ", op.node_def)

    def testReservedAttr(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            op = op_def_library.apply_op('ReservedAttr', range_=7, name='x')
            self.assertProtoEquals("\n        name: 'x' op: 'ReservedAttr' attr { key: 'range' value { i: 7 } }\n        ", op.node_def)

    def testDefaultAttrType(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrTypeDefault', a=[], name='n')
            self.assertProtoEquals("\n        name: 'n' op: 'AttrTypeDefault' input: 'n/a'\n        attr { key: 'T' value { type: DT_INT32 } }\n        ", op.node_def)
            op = op_def_library.apply_op('AttrTypeDefault', a=[1.0], name='f')
            self.assertProtoEquals("\n        name: 'f' op: 'AttrTypeDefault' input: 'f/a'\n        attr { key: 'T' value { type: DT_FLOAT } }\n        ", op.node_def)

    def testDefaultListAttrType(self):
        if False:
            return 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('AttrListTypeDefault', a=[1.0], b=[2.0], name='n')
            self.assertProtoEquals("\n        name: 'n' op: 'AttrListTypeDefault' input: 'n/a_0' input: 'n/b_0'\n        attr { key: 'T' value { type: DT_FLOAT } }\n        attr { key: 'N' value { i: 1 } }\n        ", op.node_def)

    def testNIntsIn(self):
        if False:
            return 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('NIntsIn', a=[1, 2], name='n')
            self.assertProtoEquals("\n        name: 'n' op: 'NIntsIn' input: 'n/a_0' input: 'n/a_1'\n        attr { key: 'N' value { i: 2 } }\n        ", op.node_def)
            op = op_def_library.apply_op('NIntsIn', a=[5, 4, 3, 2, 1], name='o')
            self.assertProtoEquals("\n        name: 'o' op: 'NIntsIn'\n        input: 'o/a_0' input: 'o/a_1' input: 'o/a_2' input: 'o/a_3' input: 'o/a_4'\n        attr { key: 'N' value { i: 5 } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NIntsIn', a=['foo', 'bar'])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'a' of 'NIntsIn' Op have types [string, string] that do not match expected type int32.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NIntsIn', a=[self.Tensor(dtypes.string), self.Tensor(dtypes.string)])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'a' of 'NIntsIn' Op have types [string, string] that do not match expected type int32.")
            with self.assertRaises(ValueError) as cm:
                op_def_library.apply_op('NIntsIn', a=[99])
            self.assertEqual(str(cm.exception), "List argument 'a' to 'NIntsIn' Op with length 1 shorter than minimum length 2.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NIntsIn', a=[38, 'bar'])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'a' of 'NIntsIn' Op have types [int32, string] that do not match expected type int32.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NIntsIn', a=[self.Tensor(dtypes.int32), self.Tensor(dtypes.string)])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'a' of 'NIntsIn' Op have types [int32, string] that do not match expected type int32.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NIntsIn', a=17)
            self.assertStartsWith(str(cm.exception), "Expected list for 'a' argument to 'NIntsIn' Op, not ")

    def testNPolymorphicIn(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            op = op_def_library.apply_op('NPolymorphicIn', a=[1, 2], name='n')
            self.assertProtoEquals("\n        name: 'n' op: 'NPolymorphicIn' input: 'n/a_0' input: 'n/a_1'\n        attr { key: 'T' value { type: DT_INT32 } }\n        attr { key: 'N' value { i: 2 } }\n        ", op.node_def)
            op = op_def_library.apply_op('NPolymorphicIn', a=[5, 4, 3, 2, 1], name='o')
            self.assertProtoEquals("\n        name: 'o' op: 'NPolymorphicIn'\n        input: 'o/a_0' input: 'o/a_1' input: 'o/a_2' input: 'o/a_3' input: 'o/a_4'\n        attr { key: 'T' value { type: DT_INT32 } }\n        attr { key: 'N' value { i: 5 } }\n        ", op.node_def)
            op = op_def_library.apply_op('NPolymorphicIn', a=['foo', 'bar'], name='p')
            self.assertProtoEquals("\n        name: 'p' op: 'NPolymorphicIn' input: 'p/a_0' input: 'p/a_1'\n        attr { key: 'T' value { type: DT_STRING } }\n        attr { key: 'N' value { i: 2 } }\n        ", op.node_def)
            op = op_def_library.apply_op('NPolymorphicIn', a=[1, self.Tensor(dtypes.float32, name='x')], name='q')
            self.assertProtoEquals("\n        name: 'q' op: 'NPolymorphicIn' input: 'q/a_0' input: 'x'\n        attr { key: 'T' value { type: DT_FLOAT } }\n        attr { key: 'N' value { i: 2 } }\n        ", op.node_def)
            op = op_def_library.apply_op('NPolymorphicIn', a=[self.Tensor(dtypes.float32, name='y'), self.Tensor(dtypes.float32_ref, name='z')], name='r')
            self.assertProtoEquals("\n        name: 'r' op: 'NPolymorphicIn' input: 'y' input: 'z'\n        attr { key: 'T' value { type: DT_FLOAT } }\n        attr { key: 'N' value { i: 2 } }\n        ", op.node_def)
            with self.assertRaises(ValueError) as cm:
                op_def_library.apply_op('NPolymorphicIn', a=[99])
            self.assertEqual(str(cm.exception), "List argument 'a' to 'NPolymorphicIn' Op with length 1 shorter than minimum length 2.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NPolymorphicIn', a=[38, 'bar'])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'a' of 'NPolymorphicIn' Op have types [int32, string] that don't all match.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NPolymorphicIn', a=[38, self.Tensor(dtypes.string)])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'a' of 'NPolymorphicIn' Op have types [int32, string] that don't all match.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NPolymorphicIn', a=[38, None])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'a' of 'NPolymorphicIn' Op have types [int32, <NOT CONVERTIBLE TO TENSOR>] that don't all match.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NPolymorphicIn', a=['abcd', self.Tensor(dtypes.int32)])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'a' of 'NPolymorphicIn' Op have types [string, int32] that don't all match.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NPolymorphicIn', a=17)
            self.assertStartsWith(str(cm.exception), "Expected list for 'a' argument to 'NPolymorphicIn' Op, not ")

    def testNPolymorphicRestrictIn(self):
        if False:
            return 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('NPolymorphicRestrictIn', a=['foo', 'bar'], name='p')
            self.assertProtoEquals("\n        name: 'p' op: 'NPolymorphicRestrictIn' input: 'p/a_0' input: 'p/a_1'\n        attr { key: 'T' value { type: DT_STRING } }\n        attr { key: 'N' value { i: 2 } }\n        ", op.node_def)
            op = op_def_library.apply_op('NPolymorphicRestrictIn', a=[False, True, False], name='b')
            self.assertProtoEquals("\n        name: 'b' op: 'NPolymorphicRestrictIn'\n        input: 'b/a_0' input: 'b/a_1' input: 'b/a_2'\n        attr { key: 'T' value { type: DT_BOOL } }\n        attr { key: 'N' value { i: 3 } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NPolymorphicRestrictIn', a=[1, 2])
            self.assertEqual(str(cm.exception), "Value passed to parameter 'a' has DataType int32 not in list of allowed values: string, bool")

    def testNInTwice(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            op = op_def_library.apply_op('NInTwice', a=[1, 2], b=['one', 'two'], name='n')
            self.assertProtoEquals("\n        name: 'n' op: 'NInTwice'\n        input: 'n/a_0' input: 'n/a_1' input: 'n/b_0' input: 'n/b_1'\n        attr { key: 'N' value { i: 2 } }\n        ", op.node_def)
            op = op_def_library.apply_op('NInTwice', a=[], b=[], name='o')
            self.assertProtoEquals("\n        name: 'o' op: 'NInTwice' attr { key: 'N' value { i: 0 } }\n        ", op.node_def)
            with self.assertRaises(ValueError) as cm:
                op_def_library.apply_op('NInTwice', a=[1, 2, 3], b=['too short'])
            self.assertEqual(str(cm.exception), "List argument 'b' to 'NInTwice' Op with length 1 must match length 3 of argument 'a'.")

    def testNInPolymorphicTwice(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            op = op_def_library.apply_op('NInPolymorphicTwice', a=[1, 2], b=[3, 4], name='n')
            self.assertProtoEquals("\n        name: 'n' op: 'NInPolymorphicTwice'\n        input: 'n/a_0' input: 'n/a_1' input: 'n/b_0' input: 'n/b_1'\n        attr { key: 'T' value { type: DT_INT32 } }\n        attr { key: 'N' value { i: 2 } }\n        ", op.node_def)
            with self.assertRaises(ValueError) as cm:
                op_def_library.apply_op('NInPolymorphicTwice', a=[1, 2, 3], b=[5])
            self.assertEqual(str(cm.exception), "List argument 'b' to 'NInPolymorphicTwice' Op with length 1 must match length 3 of argument 'a'.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NInPolymorphicTwice', a=[1, 2], b=['one', 'two'])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'b' of 'NInPolymorphicTwice' Op have types [string, string] that do not match type int32 inferred from earlier arguments.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NInPolymorphicTwice', a=[self.Tensor(dtypes.int32)], b=[self.Tensor(dtypes.string)])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'b' of 'NInPolymorphicTwice' Op have types [string] that do not match type int32 inferred from earlier arguments.")

    def testNInTwoTypeVariables(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            op = op_def_library.apply_op('NInTwoTypeVariables', a=[1, 2], b=[True, False], name='n')
            self.assertProtoEquals("\n        name: 'n' op: 'NInTwoTypeVariables'\n        input: 'n/a_0' input: 'n/a_1' input: 'n/b_0' input: 'n/b_1'\n        attr { key: 'S' value { type: DT_INT32 } }\n        attr { key: 'T' value { type: DT_BOOL } }\n        attr { key: 'N' value { i: 2 } }\n        ", op.node_def)
            op = op_def_library.apply_op('NInTwoTypeVariables', a=[1, 2], b=[3, 4], name='o')
            self.assertProtoEquals("\n        name: 'o' op: 'NInTwoTypeVariables'\n        input: 'o/a_0' input: 'o/a_1' input: 'o/b_0' input: 'o/b_1'\n        attr { key: 'S' value { type: DT_INT32 } }\n        attr { key: 'T' value { type: DT_INT32 } }\n        attr { key: 'N' value { i: 2 } }\n        ", op.node_def)
            op = op_def_library.apply_op('NInTwoTypeVariables', a=[self.Tensor(dtypes.int32, name='q')], b=[self.Tensor(dtypes.string, name='r')], name='p')
            self.assertProtoEquals("\n        name: 'p' op: 'NInTwoTypeVariables' input: 'q' input: 'r'\n        attr { key: 'S' value { type: DT_INT32 } }\n        attr { key: 'T' value { type: DT_STRING } }\n        attr { key: 'N' value { i: 1 } }\n        ", op.node_def)
            with self.assertRaises(ValueError) as cm:
                op_def_library.apply_op('NInTwoTypeVariables', a=[1, 2, 3], b=['5'])
            self.assertEqual(str(cm.exception), "List argument 'b' to 'NInTwoTypeVariables' Op with length 1 must match length 3 of argument 'a'.")

    def testInPolymorphicTwice(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            op = op_def_library.apply_op('InPolymorphicTwice', a=[8], b=[3, 4, 5], name='n')
            self.assertProtoEquals("\n        name: 'n' op: 'InPolymorphicTwice'\n        input: 'n/a_0' input: 'n/b_0' input: 'n/b_1' input: 'n/b_2'\n        attr { key: 'T' value { type: DT_INT32 } }\n        attr { key: 'N' value { i: 1 } }\n        attr { key: 'M' value { i: 3 } }\n        ", op.node_def)
            op = op_def_library.apply_op('InPolymorphicTwice', a=[8], b=[], name='o')
            self.assertProtoEquals("\n        name: 'o' op: 'InPolymorphicTwice' input: 'o/a_0'\n        attr { key: 'T' value { type: DT_INT32 } }\n        attr { key: 'N' value { i: 1 } }\n        attr { key: 'M' value { i: 0 } }\n        ", op.node_def)
            op = op_def_library.apply_op('InPolymorphicTwice', a=[], b=[3, 4], name='p')
            self.assertProtoEquals("\n        name: 'p' op: 'InPolymorphicTwice' input: 'p/b_0' input: 'p/b_1'\n        attr { key: 'T' value { type: DT_INT32 } }\n        attr { key: 'N' value { i: 0 } }\n        attr { key: 'M' value { i: 2 } }\n        ", op.node_def)
            op = op_def_library.apply_op('InPolymorphicTwice', a=[], b=[3.0, 4.0], name='q')
            self.assertProtoEquals("\n        name: 'q' op: 'InPolymorphicTwice' input: 'q/b_0' input: 'q/b_1'\n        attr { key: 'T' value { type: DT_FLOAT } }\n        attr { key: 'N' value { i: 0 } }\n        attr { key: 'M' value { i: 2 } }\n        ", op.node_def)
            op = op_def_library.apply_op('InPolymorphicTwice', a=[], b=[], name='r')
            self.assertProtoEquals("\n        name: 'r' op: 'InPolymorphicTwice'\n        attr { key: 'T' value { type: DT_INT32 } }\n        attr { key: 'N' value { i: 0 } }\n        attr { key: 'M' value { i: 0 } }\n        ", op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('InPolymorphicTwice', a=[1, 2], b=['one', 'two'])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'b' of 'InPolymorphicTwice' Op have types [string, string] that do not match type int32 inferred from earlier arguments.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('InPolymorphicTwice', a=[self.Tensor(dtypes.int32)], b=[self.Tensor(dtypes.string)])
            self.assertEqual(str(cm.exception), "Tensors in list passed to 'b' of 'InPolymorphicTwice' Op have types [string] that do not match type int32 inferred from earlier arguments.")

    def testNIntsOut(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            (out1, out2) = op_def_library.apply_op('NIntsOut', N=2, name='n')
            self.assertEqual(dtypes.int32, out1.dtype)
            self.assertEqual(dtypes.int32, out2.dtype)
            self.assertProtoEquals("\n        name: 'n' op: 'NIntsOut' attr { key: 'N' value { i: 2 } }\n        ", out1.op.node_def)
            (out1, out2, out3, out4, out5) = op_def_library.apply_op('NIntsOut', N=5, name='o')
            self.assertEqual(dtypes.int32, out1.dtype)
            self.assertEqual(dtypes.int32, out2.dtype)
            self.assertEqual(dtypes.int32, out3.dtype)
            self.assertEqual(dtypes.int32, out4.dtype)
            self.assertEqual(dtypes.int32, out5.dtype)
            self.assertProtoEquals("\n        name: 'o' op: 'NIntsOut' attr { key: 'N' value { i: 5 } }\n        ", out5.op.node_def)
            with self.assertRaises(ValueError) as cm:
                op_def_library.apply_op('NIntsOut', N=1)
            self.assertEqual(str(cm.exception), "Attr 'N' of 'NIntsOut' Op passed 1 less than minimum 2.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NIntsOut', N=[3])
            self.assertEqual(str(cm.exception), "Expected int for argument 'N' not [3].")

    def testNIntsOutDefault(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            (out1, out2, out3) = op_def_library.apply_op('NIntsOutDefault', N=None, name='z')
            self.assertEqual(dtypes.int32, out1.dtype)
            self.assertEqual(dtypes.int32, out2.dtype)
            self.assertEqual(dtypes.int32, out3.dtype)
            self.assertProtoEquals("\n        name: 'z' op: 'NIntsOutDefault' attr { key: 'N' value { i: 3 } }\n        ", out1.op.node_def)
            (out1, out2) = op_def_library.apply_op('NIntsOutDefault', N=2, name='y')
            self.assertEqual(dtypes.int32, out1.dtype)
            self.assertEqual(dtypes.int32, out2.dtype)
            self.assertProtoEquals("\n        name: 'y' op: 'NIntsOutDefault' attr { key: 'N' value { i: 2 } }\n        ", out2.op.node_def)

    def testNPolymorphicOut(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            (out1, out2) = op_def_library.apply_op('NPolymorphicOut', N=2, T=dtypes.int32, name='n')
            self.assertEqual(dtypes.int32, out1.dtype)
            self.assertEqual(dtypes.int32, out2.dtype)
            self.assertProtoEquals("\n        name: 'n' op: 'NPolymorphicOut'\n        attr { key: 'T' value { type: DT_INT32 } }\n        attr { key: 'N' value { i: 2 } }\n        ", out1.op.node_def)
            (out1, out2, out3) = op_def_library.apply_op('NPolymorphicOut', T=dtypes.string, N=3, name='o')
            self.assertEqual(dtypes.string, out1.dtype)
            self.assertEqual(dtypes.string, out2.dtype)
            self.assertEqual(dtypes.string, out3.dtype)
            self.assertProtoEquals("\n        name: 'o' op: 'NPolymorphicOut'\n        attr { key: 'T' value { type: DT_STRING } }\n        attr { key: 'N' value { i: 3 } }\n        ", out3.op.node_def)
            with self.assertRaises(ValueError) as cm:
                op_def_library.apply_op('NPolymorphicOut', N=1, T=dtypes.string)
            self.assertEqual(str(cm.exception), "Attr 'N' of 'NPolymorphicOut' Op passed 1 less than minimum 2.")
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NPolymorphicOut', N=3, T=[dtypes.string])
            self.assertEqual(str(cm.exception), "Expected DataType for argument 'T' not [tf.string].")

    def testNPolymorphicOutDefault(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            (out1, out2) = op_def_library.apply_op('NPolymorphicOutDefault', N=None, T=None, name='r')
            self.assertEqual(dtypes.bool, out1.dtype)
            self.assertEqual(dtypes.bool, out2.dtype)
            self.assertProtoEquals("\n        name: 'r' op: 'NPolymorphicOutDefault'\n        attr { key: 'T' value { type: DT_BOOL } }\n        attr { key: 'N' value { i: 2 } }\n        ", out1.op.node_def)
            (out1, out2, out3) = op_def_library.apply_op('NPolymorphicOutDefault', N=3, T=None, name='s')
            self.assertEqual(dtypes.bool, out1.dtype)
            self.assertEqual(dtypes.bool, out2.dtype)
            self.assertEqual(dtypes.bool, out3.dtype)
            self.assertProtoEquals("\n        name: 's' op: 'NPolymorphicOutDefault'\n        attr { key: 'T' value { type: DT_BOOL } }\n        attr { key: 'N' value { i: 3 } }\n        ", out1.op.node_def)
            (out1, out2) = op_def_library.apply_op('NPolymorphicOutDefault', N=None, T=dtypes.int32, name='t')
            self.assertEqual(dtypes.int32, out1.dtype)
            self.assertEqual(dtypes.int32, out2.dtype)
            self.assertProtoEquals("\n        name: 't' op: 'NPolymorphicOutDefault'\n        attr { key: 'T' value { type: DT_INT32 } }\n        attr { key: 'N' value { i: 2 } }\n        ", out1.op.node_def)
            (out1, out2, out3) = op_def_library.apply_op('NPolymorphicOutDefault', N=3, T=dtypes.int32, name='u')
            self.assertEqual(dtypes.int32, out1.dtype)
            self.assertEqual(dtypes.int32, out2.dtype)
            self.assertEqual(dtypes.int32, out3.dtype)
            self.assertProtoEquals("\n        name: 'u' op: 'NPolymorphicOutDefault'\n        attr { key: 'T' value { type: DT_INT32 } }\n        attr { key: 'N' value { i: 3 } }\n        ", out1.op.node_def)

    def testNPolymorphicRestrictOut(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            (out1, out2, out3) = op_def_library.apply_op('NPolymorphicRestrictOut', N=3, T=dtypes.bool, name='u')
            self.assertEqual(dtypes.bool, out1.dtype)
            self.assertEqual(dtypes.bool, out2.dtype)
            self.assertEqual(dtypes.bool, out3.dtype)
            self.assertProtoEquals("\n        name: 'u' op: 'NPolymorphicRestrictOut'\n        attr { key: 'T' value { type: DT_BOOL } }\n        attr { key: 'N' value { i: 3 } }\n        ", out1.op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('NPolymorphicRestrictOut', N=2, T=dtypes.int32)
            self.assertEqual(str(cm.exception), "Value passed to parameter 'T' has DataType int32 not in list of allowed values: string, bool")

    def testRef(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            out = op_def_library.apply_op('RefOut', T=dtypes.bool, name='o')
            self.assertEqual(dtypes.bool_ref, out.dtype)
            self.assertProtoEquals("\n        name: 'o' op: 'RefOut'\n        attr { key: 'T' value { type: DT_BOOL } }\n        ", out.op.node_def)
            op = op_def_library.apply_op('RefIn', a=out, name='i')
            self.assertProtoEquals('\n        name: \'i\' op: \'RefIn\' input: \'o\'\n        attr { key: \'T\' value { type: DT_BOOL } }\n        attr { key: "_class" value { list { s: "loc:@o" } } }\n        ', op.node_def)
            out = op_def_library.apply_op('RefOut', T=dtypes.int32, name='r')
            out = op_def_library.apply_op('Simple', a=out, name='s')
            self.assertProtoEquals("\n        name: 's' op: 'Simple' input: 'r'\n        ", out.op.node_def)
            with self.assertRaises(TypeError) as cm:
                op_def_library.apply_op('RefIn', a=2)
            self.assertEqual(str(cm.exception), "'RefIn' Op requires that input 'a' be a mutable tensor " + '(e.g.: a tf.Variable)')
            input_a = op_def_library.apply_op('RefOut', T=dtypes.int32, name='t')
            input_b = op_def_library.apply_op('RefOut', T=dtypes.int32, name='u')
            op = op_def_library.apply_op('TwoRefsIn', a=input_a, b=input_b, name='v')
            self.assertProtoEquals('\n        name: \'v\' op: \'TwoRefsIn\' input: \'t\' input: \'u\'\n        attr { key: \'T\' value { type: DT_INT32 } }\n        attr { key: "_class" value { list { s: "loc:@t" s: "loc:@u" } } }\n        ', op.node_def)

    def testSpecifyDevice(self):
        if False:
            while True:
                i = 10
        graph = ops.Graph()
        with graph.as_default():
            with graph.device('/job:ADevice'):
                op_def_library.apply_op('Simple', a=3)
            graph_def = graph.as_graph_def()
            self.assertEqual(len(graph_def.node), 2)
            for node in graph_def.node:
                self.assertDeviceEqual(node.device, '/job:ADevice')

    def testStructuredOutputSingleList(self):
        if False:
            return 10
        with ops.Graph().as_default():
            for n_a in [0, 1, 3]:
                a = op_def_library.apply_op('SimpleStruct', n_a=n_a)
                self.assertIsInstance(a, list)
                self.assertEqual(n_a, len(a))

    def testStructuredOutputListAndSingle(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            for n_a in [0, 1, 3]:
                (a, b) = op_def_library.apply_op('MixedStruct', n_a=n_a)
                self.assertIsInstance(a, list)
                self.assertEqual(n_a, len(a))
                self.assertTrue(all((x.dtype == dtypes.int32 for x in a)))
                self.assertIsInstance(b, tensor.Tensor)
                self.assertEqual(dtypes.float32, b.dtype)

    def testStructuredOutputMultipleLists(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            for n_a in [0, 1, 3]:
                for n_b in [0, 1, 3]:
                    for t_c in [[], [dtypes.int32], [dtypes.int32, dtypes.float32]]:
                        (a, b, c) = op_def_library.apply_op('ComplexStruct', n_a=n_a, n_b=n_b, t_c=t_c)
                        self.assertEqual(n_a, len(a))
                        self.assertTrue(all((x.dtype == dtypes.int32 for x in a)))
                        self.assertEqual(n_b, len(b))
                        self.assertTrue(all((x.dtype == dtypes.int64 for x in b)))
                        self.assertEqual(t_c, [x.dtype for x in c])

@test_util.add_graph_building_optimization_tests
class OpDefLibraryGraphTest(test_util.TensorFlowTestCase):

    def testNoGraph(self):
        if False:
            for i in range(10):
                print('nop')
        out = op_def_library.apply_op('Simple', a=3)
        self.assertEqual(out.graph, ops.get_default_graph())

    def testDefaultGraph(self):
        if False:
            print('Hello World!')
        graph = ops.Graph()
        with graph.as_default():
            out = op_def_library.apply_op('Simple', a=3)
            self.assertEqual(out.graph, graph)

    def testDifferentGraphFails(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            a = op_def_library.apply_op('Simple', a=3)
        with ops.Graph().as_default():
            b = op_def_library.apply_op('Simple', a=4)
        with self.assertRaises(ValueError) as cm:
            op_def_library.apply_op('Binary', a=a, b=b)
        self.assertIn('must be from the same graph', str(cm.exception))

class OpDefLibraryPybindTest(test_util.TensorFlowTestCase):

    def testPybind(self):
        if False:
            return 10
        x = constant_op.constant(32, dtype=dtypes.float32)
        y = constant_op.constant(32, dtype=dtypes.float32)
        (attrs, inputs, input_types, output_structure) = op_def_library_pybind.process_inputs('AddV2', 1, {'x': x, 'y': y})
        proto = text_format.Parse('type: DT_FLOAT', attr_value_pb2.AttrValue())
        self.assertEqual(attrs, {'T': proto})
        self.assertEqual(inputs, [x, y])
        self.assertEqual(input_types, [dtypes.float32, dtypes.float32])
        self.assertEqual(output_structure, [None])
if __name__ == '__main__':
    googletest.main()