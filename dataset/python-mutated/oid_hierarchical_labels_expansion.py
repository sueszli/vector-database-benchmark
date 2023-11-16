"""An executable to expand image-level labels, boxes and segments.

The expansion is performed using class hierarchy, provided in JSON file.

The expected file formats are the following:
- for box and segment files: CSV file is expected to have LabelName field
- for image-level labels: CSV file is expected to have LabelName and Confidence
fields

Note, that LabelName is the only field used for expansion.

Example usage:
python models/research/object_detection/dataset_tools/\\
oid_hierarchical_labels_expansion.py \\
--json_hierarchy_file=<path to JSON hierarchy> \\
--input_annotations=<input csv file> \\
--output_annotations=<output csv file> \\
--annotation_type=<1 (for boxes and segments) or 2 (for image-level labels)>
"""
from __future__ import print_function
import copy
import json
from absl import app
from absl import flags
flags.DEFINE_string('json_hierarchy_file', None, 'Path to the file containing label hierarchy in JSON format.')
flags.DEFINE_string('input_annotations', None, 'Path to Open Images annotations file(either bounding boxes, segments or image-level labels).')
flags.DEFINE_string('output_annotations', None, 'Path to the output file.')
flags.DEFINE_integer('annotation_type', None, 'Type of the input annotations: 1 - boxes or segments,2 - image-level labels.')
FLAGS = flags.FLAGS

def _update_dict(initial_dict, update):
    if False:
        for i in range(10):
            print('nop')
    'Updates dictionary with update content.\n\n  Args:\n   initial_dict: initial dictionary.\n   update: updated dictionary.\n  '
    for (key, value_list) in update.items():
        if key in initial_dict:
            initial_dict[key].update(value_list)
        else:
            initial_dict[key] = set(value_list)

def _build_plain_hierarchy(hierarchy, skip_root=False):
    if False:
        i = 10
        return i + 15
    'Expands tree hierarchy representation to parent-child dictionary.\n\n  Args:\n   hierarchy: labels hierarchy as JSON file.\n   skip_root: if true skips root from the processing (done for the case when all\n     classes under hierarchy are collected under virtual node).\n\n  Returns:\n    keyed_parent - dictionary of parent - all its children nodes.\n    keyed_child  - dictionary of children - all its parent nodes\n    children - all children of the current node.\n  '
    all_children = set([])
    all_keyed_parent = {}
    all_keyed_child = {}
    if 'Subcategory' in hierarchy:
        for node in hierarchy['Subcategory']:
            (keyed_parent, keyed_child, children) = _build_plain_hierarchy(node)
            _update_dict(all_keyed_parent, keyed_parent)
            _update_dict(all_keyed_child, keyed_child)
            all_children.update(children)
    if not skip_root:
        all_keyed_parent[hierarchy['LabelName']] = copy.deepcopy(all_children)
        all_children.add(hierarchy['LabelName'])
        for (child, _) in all_keyed_child.items():
            all_keyed_child[child].add(hierarchy['LabelName'])
        all_keyed_child[hierarchy['LabelName']] = set([])
    return (all_keyed_parent, all_keyed_child, all_children)

class OIDHierarchicalLabelsExpansion(object):
    """ Main class to perform labels hierachical expansion."""

    def __init__(self, hierarchy):
        if False:
            return 10
        'Constructor.\n\n    Args:\n      hierarchy: labels hierarchy as JSON object.\n    '
        (self._hierarchy_keyed_parent, self._hierarchy_keyed_child, _) = _build_plain_hierarchy(hierarchy, skip_root=True)

    def expand_boxes_or_segments_from_csv(self, csv_row, labelname_column_index=1):
        if False:
            for i in range(10):
                print('nop')
        'Expands a row containing bounding boxes/segments from CSV file.\n\n    Args:\n      csv_row: a single row of Open Images released groundtruth file.\n      labelname_column_index: 0-based index of LabelName column in CSV file.\n\n    Returns:\n      a list of strings (including the initial row) corresponding to the ground\n      truth expanded to multiple annotation for evaluation with Open Images\n      Challenge 2018/2019 metrics.\n    '
        split_csv_row = csv_row.split(',')
        result = [csv_row]
        assert split_csv_row[labelname_column_index] in self._hierarchy_keyed_child
        parent_nodes = self._hierarchy_keyed_child[split_csv_row[labelname_column_index]]
        for parent_node in parent_nodes:
            split_csv_row[labelname_column_index] = parent_node
            result.append(','.join(split_csv_row))
        return result

    def expand_labels_from_csv(self, csv_row, labelname_column_index=1, confidence_column_index=2):
        if False:
            for i in range(10):
                print('nop')
        'Expands a row containing labels from CSV file.\n\n    Args:\n      csv_row: a single row of Open Images released groundtruth file.\n      labelname_column_index: 0-based index of LabelName column in CSV file.\n      confidence_column_index: 0-based index of Confidence column in CSV file.\n\n    Returns:\n      a list of strings (including the initial row) corresponding to the ground\n      truth expanded to multiple annotation for evaluation with Open Images\n      Challenge 2018/2019 metrics.\n    '
        split_csv_row = csv_row.split(',')
        result = [csv_row]
        if int(split_csv_row[confidence_column_index]) == 1:
            assert split_csv_row[labelname_column_index] in self._hierarchy_keyed_child
            parent_nodes = self._hierarchy_keyed_child[split_csv_row[labelname_column_index]]
            for parent_node in parent_nodes:
                split_csv_row[labelname_column_index] = parent_node
                result.append(','.join(split_csv_row))
        else:
            assert split_csv_row[labelname_column_index] in self._hierarchy_keyed_parent
            child_nodes = self._hierarchy_keyed_parent[split_csv_row[labelname_column_index]]
            for child_node in child_nodes:
                split_csv_row[labelname_column_index] = child_node
                result.append(','.join(split_csv_row))
        return result

def main(unused_args):
    if False:
        for i in range(10):
            print('nop')
    del unused_args
    with open(FLAGS.json_hierarchy_file) as f:
        hierarchy = json.load(f)
    expansion_generator = OIDHierarchicalLabelsExpansion(hierarchy)
    labels_file = False
    if FLAGS.annotation_type == 2:
        labels_file = True
    elif FLAGS.annotation_type != 1:
        print('--annotation_type expected value is 1 or 2.')
        return -1
    confidence_column_index = -1
    labelname_column_index = -1
    with open(FLAGS.input_annotations, 'r') as source:
        with open(FLAGS.output_annotations, 'w') as target:
            header = source.readline()
            target.writelines([header])
            column_names = header.strip().split(',')
            labelname_column_index = column_names.index('LabelName')
            if labels_file:
                confidence_column_index = column_names.index('Confidence')
            for line in source:
                if labels_file:
                    expanded_lines = expansion_generator.expand_labels_from_csv(line, labelname_column_index, confidence_column_index)
                else:
                    expanded_lines = expansion_generator.expand_boxes_or_segments_from_csv(line, labelname_column_index)
                target.writelines(expanded_lines)
if __name__ == '__main__':
    flags.mark_flag_as_required('json_hierarchy_file')
    flags.mark_flag_as_required('input_annotations')
    flags.mark_flag_as_required('output_annotations')
    flags.mark_flag_as_required('annotation_type')
    app.run(main)