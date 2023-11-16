import argparse
import os
import libcst as cst
import pathlib
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

def partition(predicate: Callable[[Any], bool], iterator: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    if False:
        return 10
    'A stable, out-of-place partition.'
    results = ([], [])
    for i in iterator:
        results[int(predicate(i))].append(i)
    return (results[1], results[0])

class contact_center_insightsCallTransformer(cst.CSTTransformer):
    CTRL_PARAMS: Tuple[str] = ('retry', 'timeout', 'metadata')
    METHOD_TO_PARAMS: Dict[str, Tuple[str]] = {'bulk_analyze_conversations': ('parent', 'filter', 'analysis_percentage', 'annotator_selector'), 'bulk_delete_conversations': ('parent', 'filter', 'max_delete_count', 'force'), 'calculate_issue_model_stats': ('issue_model',), 'calculate_stats': ('location', 'filter'), 'create_analysis': ('parent', 'analysis'), 'create_conversation': ('parent', 'conversation', 'conversation_id'), 'create_issue_model': ('parent', 'issue_model'), 'create_phrase_matcher': ('parent', 'phrase_matcher'), 'create_view': ('parent', 'view'), 'delete_analysis': ('name',), 'delete_conversation': ('name', 'force'), 'delete_issue': ('name',), 'delete_issue_model': ('name',), 'delete_phrase_matcher': ('name',), 'delete_view': ('name',), 'deploy_issue_model': ('name',), 'export_insights_data': ('parent', 'big_query_destination', 'filter', 'kms_key', 'write_disposition'), 'get_analysis': ('name',), 'get_conversation': ('name', 'view'), 'get_issue': ('name',), 'get_issue_model': ('name',), 'get_phrase_matcher': ('name',), 'get_settings': ('name',), 'get_view': ('name',), 'ingest_conversations': ('parent', 'gcs_source', 'transcript_object_config', 'conversation_config', 'redaction_config', 'speech_config'), 'list_analyses': ('parent', 'page_size', 'page_token', 'filter'), 'list_conversations': ('parent', 'page_size', 'page_token', 'filter', 'view'), 'list_issue_models': ('parent',), 'list_issues': ('parent',), 'list_phrase_matchers': ('parent', 'page_size', 'page_token', 'filter'), 'list_views': ('parent', 'page_size', 'page_token'), 'undeploy_issue_model': ('name',), 'update_conversation': ('conversation', 'update_mask'), 'update_issue': ('issue', 'update_mask'), 'update_issue_model': ('issue_model', 'update_mask'), 'update_phrase_matcher': ('phrase_matcher', 'update_mask'), 'update_settings': ('settings', 'update_mask'), 'update_view': ('view', 'update_mask'), 'upload_conversation': ('parent', 'conversation', 'conversation_id', 'redaction_config', 'speech_config')}

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.CSTNode:
        if False:
            i = 10
            return i + 15
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

def fix_files(in_dir: pathlib.Path, out_dir: pathlib.Path, *, transformer=contact_center_insightsCallTransformer()):
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
    parser = argparse.ArgumentParser(description='Fix up source that uses the contact_center_insights client library.\n\nThe existing sources are NOT overwritten but are copied to output_dir with changes made.\n\nNote: This tool operates at a best-effort level at converting positional\n      parameters in client method calls to keyword based parameters.\n      Cases where it WILL FAIL include\n      A) * or ** expansion in a method call.\n      B) Calls via function or method alias (includes free function calls)\n      C) Indirect or dispatched calls (e.g. the method is looked up dynamically)\n\n      These all constitute false negatives. The tool will also detect false\n      positives when an API method shares a name with another method.\n')
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