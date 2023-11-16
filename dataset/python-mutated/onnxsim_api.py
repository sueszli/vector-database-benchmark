from onnxsim import simplify
import onnx

def onnx_simplify(onnx_path: str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Simplify the ONNX model based on onnxsim.\n    If simplification is successful, will overwrite new ONNX model to onnx_path\n\n    :param onnx_path: File path of of onnx ModelProto object.\n    '
    model = onnx.load(onnx_path)
    (model_simp, check) = simplify(model)
    if check is True:
        onnx.save(model_simp, onnx_path)