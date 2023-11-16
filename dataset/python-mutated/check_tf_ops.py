import argparse
import json
import os
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
REPO_PATH = '.'
INTERNAL_OPS = ['Assert', 'AssignVariableOp', 'EmptyTensorList', 'MergeV2Checkpoints', 'ReadVariableOp', 'ResourceGather', 'RestoreV2', 'SaveV2', 'ShardedFilename', 'StatefulPartitionedCall', 'StaticRegexFullMatch', 'VarHandleOp']

def onnx_compliancy(saved_model_path, strict, opset):
    if False:
        return 10
    saved_model = SavedModel()
    onnx_ops = []
    with open(os.path.join(REPO_PATH, 'utils', 'tf_ops', 'onnx.json')) as f:
        onnx_opsets = json.load(f)['opsets']
    for i in range(1, opset + 1):
        onnx_ops.extend(onnx_opsets[str(i)])
    with open(saved_model_path, 'rb') as f:
        saved_model.ParseFromString(f.read())
    model_op_names = set()
    for meta_graph in saved_model.meta_graphs:
        model_op_names.update((node.op for node in meta_graph.graph_def.node))
        for func in meta_graph.graph_def.library.function:
            model_op_names.update((node.op for node in func.node_def))
    model_op_names = sorted(model_op_names)
    incompatible_ops = []
    for op in model_op_names:
        if op not in onnx_ops and op not in INTERNAL_OPS:
            incompatible_ops.append(op)
    if strict and len(incompatible_ops) > 0:
        raise Exception(f'Found the following incompatible ops for the opset {opset}:\n' + incompatible_ops)
    elif len(incompatible_ops) > 0:
        print(f'Found the following incompatible ops for the opset {opset}:')
        print(*incompatible_ops, sep='\n')
    else:
        print(f'The saved model {saved_model_path} can properly be converted with ONNX.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_path', help='Path of the saved model to check (the .pb file).')
    parser.add_argument('--opset', default=12, type=int, help='The ONNX opset against which the model has to be tested.')
    parser.add_argument('--framework', choices=['onnx'], default='onnx', help='Frameworks against which to test the saved model.')
    parser.add_argument('--strict', action='store_true', help='Whether make the checking strict (raise errors) or not (raise warnings)')
    args = parser.parse_args()
    if args.framework == 'onnx':
        onnx_compliancy(args.saved_model_path, args.strict, args.opset)