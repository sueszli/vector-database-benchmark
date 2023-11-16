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

class contentwarehouseCallTransformer(cst.CSTTransformer):
    CTRL_PARAMS: Tuple[str] = ('retry', 'timeout', 'metadata')
    METHOD_TO_PARAMS: Dict[str, Tuple[str]] = {'create_document': ('parent', 'document', 'request_metadata', 'policy', 'cloud_ai_document_option', 'create_mask'), 'create_document_link': ('parent', 'document_link', 'request_metadata'), 'create_document_schema': ('parent', 'document_schema'), 'create_rule_set': ('parent', 'rule_set'), 'create_synonym_set': ('parent', 'synonym_set'), 'delete_document': ('name', 'request_metadata'), 'delete_document_link': ('name', 'request_metadata'), 'delete_document_schema': ('name',), 'delete_rule_set': ('name',), 'delete_synonym_set': ('name',), 'fetch_acl': ('resource', 'request_metadata', 'project_owner'), 'get_document': ('name', 'request_metadata'), 'get_document_schema': ('name',), 'get_rule_set': ('name',), 'get_synonym_set': ('name',), 'list_document_schemas': ('parent', 'page_size', 'page_token'), 'list_linked_sources': ('parent', 'page_size', 'page_token', 'request_metadata'), 'list_linked_targets': ('parent', 'request_metadata'), 'list_rule_sets': ('parent', 'page_size', 'page_token'), 'list_synonym_sets': ('parent', 'page_size', 'page_token'), 'lock_document': ('name', 'collection_id', 'locking_user'), 'run_pipeline': ('name', 'gcs_ingest_pipeline', 'gcs_ingest_with_doc_ai_processors_pipeline', 'export_cdw_pipeline', 'process_with_doc_ai_pipeline', 'request_metadata'), 'search_documents': ('parent', 'request_metadata', 'document_query', 'offset', 'page_size', 'page_token', 'order_by', 'histogram_queries', 'require_total_size', 'total_result_size', 'qa_size_limit'), 'set_acl': ('resource', 'policy', 'request_metadata', 'project_owner'), 'update_document': ('name', 'document', 'request_metadata', 'cloud_ai_document_option', 'update_options'), 'update_document_schema': ('name', 'document_schema'), 'update_rule_set': ('name', 'rule_set'), 'update_synonym_set': ('name', 'synonym_set')}

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.CSTNode:
        if False:
            print('Hello World!')
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

def fix_files(in_dir: pathlib.Path, out_dir: pathlib.Path, *, transformer=contentwarehouseCallTransformer()):
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
    parser = argparse.ArgumentParser(description='Fix up source that uses the contentwarehouse client library.\n\nThe existing sources are NOT overwritten but are copied to output_dir with changes made.\n\nNote: This tool operates at a best-effort level at converting positional\n      parameters in client method calls to keyword based parameters.\n      Cases where it WILL FAIL include\n      A) * or ** expansion in a method call.\n      B) Calls via function or method alias (includes free function calls)\n      C) Indirect or dispatched calls (e.g. the method is looked up dynamically)\n\n      These all constitute false negatives. The tool will also detect false\n      positives when an API method shares a name with another method.\n')
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