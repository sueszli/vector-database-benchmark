import argparse
import os
import libcst as cst
import pathlib
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

def partition(predicate: Callable[[Any], bool], iterator: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    if False:
        i = 10
        return i + 15
    'A stable, out-of-place partition.'
    results = ([], [])
    for i in iterator:
        results[int(predicate(i))].append(i)
    return (results[1], results[0])

class speechCallTransformer(cst.CSTTransformer):
    CTRL_PARAMS: Tuple[str] = ('retry', 'timeout', 'metadata')
    METHOD_TO_PARAMS: Dict[str, Tuple[str]] = {'batch_recognize': ('recognizer', 'config', 'config_mask', 'files', 'recognition_output_config', 'processing_strategy'), 'create_custom_class': ('custom_class', 'parent', 'validate_only', 'custom_class_id'), 'create_phrase_set': ('phrase_set', 'parent', 'validate_only', 'phrase_set_id'), 'create_recognizer': ('recognizer', 'parent', 'validate_only', 'recognizer_id'), 'delete_custom_class': ('name', 'validate_only', 'allow_missing', 'etag'), 'delete_phrase_set': ('name', 'validate_only', 'allow_missing', 'etag'), 'delete_recognizer': ('name', 'validate_only', 'allow_missing', 'etag'), 'get_config': ('name',), 'get_custom_class': ('name',), 'get_phrase_set': ('name',), 'get_recognizer': ('name',), 'list_custom_classes': ('parent', 'page_size', 'page_token', 'show_deleted'), 'list_phrase_sets': ('parent', 'page_size', 'page_token', 'show_deleted'), 'list_recognizers': ('parent', 'page_size', 'page_token', 'show_deleted'), 'recognize': ('recognizer', 'config', 'config_mask', 'content', 'uri'), 'streaming_recognize': ('recognizer', 'streaming_config', 'audio'), 'undelete_custom_class': ('name', 'validate_only', 'etag'), 'undelete_phrase_set': ('name', 'validate_only', 'etag'), 'undelete_recognizer': ('name', 'validate_only', 'etag'), 'update_config': ('config', 'update_mask'), 'update_custom_class': ('custom_class', 'update_mask', 'validate_only'), 'update_phrase_set': ('phrase_set', 'update_mask', 'validate_only'), 'update_recognizer': ('recognizer', 'update_mask', 'validate_only')}

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.CSTNode:
        if False:
            while True:
                i = 10
        try:
            key = original.func.attr.value
            kword_params = self.METHOD_TO_PARAMS[key]
        except (AttributeError, KeyError):
            return updated
        (args, kwargs) = partition(lambda a: not bool(a.keyword), updated.args)
        if any((k.keyword.value == 'request' for k in kwargs)):
            return updated
        (kwargs, ctrl_kwargs) = partition(lambda a: a.keyword.value not in self.CTRL_PARAMS, kwargs)
        (args, ctrl_args) = (args[:len(kword_params)], args[len(kword_params):])
        ctrl_kwargs.extend((cst.Arg(value=a.value, keyword=cst.Name(value=ctrl)) for (a, ctrl) in zip(ctrl_args, self.CTRL_PARAMS)))
        request_arg = cst.Arg(value=cst.Dict([cst.DictElement(cst.SimpleString("'{}'".format(name)), cst.Element(value=arg.value)) for (name, arg) in zip(kword_params, args + kwargs)]), keyword=cst.Name('request'))
        return updated.with_changes(args=[request_arg] + ctrl_kwargs)

def fix_files(in_dir: pathlib.Path, out_dir: pathlib.Path, *, transformer=speechCallTransformer()):
    if False:
        i = 10
        return i + 15
    'Duplicate the input dir to the output dir, fixing file method calls.\n\n    Preconditions:\n    * in_dir is a real directory\n    * out_dir is a real, empty directory\n    '
    pyfile_gen = (pathlib.Path(os.path.join(root, f)) for (root, _, files) in os.walk(in_dir) for f in files if os.path.splitext(f)[1] == '.py')
    for fpath in pyfile_gen:
        with open(fpath, 'r') as f:
            src = f.read()
        tree = cst.parse_module(src)
        updated = tree.visit(transformer)
        updated_path = out_dir.joinpath(fpath.relative_to(in_dir))
        updated_path.parent.mkdir(parents=True, exist_ok=True)
        with open(updated_path, 'w') as f:
            f.write(updated.code)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix up source that uses the speech client library.\n\nThe existing sources are NOT overwritten but are copied to output_dir with changes made.\n\nNote: This tool operates at a best-effort level at converting positional\n      parameters in client method calls to keyword based parameters.\n      Cases where it WILL FAIL include\n      A) * or ** expansion in a method call.\n      B) Calls via function or method alias (includes free function calls)\n      C) Indirect or dispatched calls (e.g. the method is looked up dynamically)\n\n      These all constitute false negatives. The tool will also detect false\n      positives when an API method shares a name with another method.\n')
    parser.add_argument('-d', '--input-directory', required=True, dest='input_dir', help='the input directory to walk for python files to fix up')
    parser.add_argument('-o', '--output-directory', required=True, dest='output_dir', help='the directory to output files fixed via un-flattening')
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    if not input_dir.is_dir():
        print(f"input directory '{input_dir}' does not exist or is not a directory", file=sys.stderr)
        sys.exit(-1)
    if not output_dir.is_dir():
        print(f"output directory '{output_dir}' does not exist or is not a directory", file=sys.stderr)
        sys.exit(-1)
    if os.listdir(output_dir):
        print(f"output directory '{output_dir}' is not empty", file=sys.stderr)
        sys.exit(-1)
    fix_files(input_dir, output_dir)