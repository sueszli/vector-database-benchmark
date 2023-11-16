"""Computes metrics for Google Landmarks Recognition dataset predictions.

Metrics are written to stdout.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from tensorflow.python.platform import app
from delf.python.google_landmarks_dataset import dataset_file_io
from delf.python.google_landmarks_dataset import metrics
cmd_args = None

def main(argv):
    if False:
        for i in range(10):
            print('nop')
    if len(argv) > 1:
        raise RuntimeError('Too many command-line arguments.')
    print('Reading solution...')
    (public_solution, private_solution, ignored_ids) = dataset_file_io.ReadSolution(cmd_args.solution_path, dataset_file_io.RECOGNITION_TASK_ID)
    print('done!')
    print('Reading predictions...')
    (public_predictions, private_predictions) = dataset_file_io.ReadPredictions(cmd_args.predictions_path, set(public_solution.keys()), set(private_solution.keys()), set(ignored_ids), dataset_file_io.RECOGNITION_TASK_ID)
    print('done!')
    print('**********************************************')
    print('(Public)  Global Average Precision: %f' % metrics.GlobalAveragePrecision(public_predictions, public_solution))
    print('(Private) Global Average Precision: %f' % metrics.GlobalAveragePrecision(private_predictions, private_solution))
    print('**********************************************')
    print('(Public)  Global Average Precision ignoring non-landmark queries: %f' % metrics.GlobalAveragePrecision(public_predictions, public_solution, ignore_non_gt_test_images=True))
    print('(Private) Global Average Precision ignoring non-landmark queries: %f' % metrics.GlobalAveragePrecision(private_predictions, private_solution, ignore_non_gt_test_images=True))
    print('**********************************************')
    print('(Public)  Top-1 accuracy: %.2f' % (100.0 * metrics.Top1Accuracy(public_predictions, public_solution)))
    print('(Private) Top-1 accuracy: %.2f' % (100.0 * metrics.Top1Accuracy(private_predictions, private_solution)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--predictions_path', type=str, default='/tmp/predictions.csv', help="\n      Path to CSV predictions file, formatted with columns 'id,landmarks' (the\n      file should include a header).\n      ")
    parser.add_argument('--solution_path', type=str, default='/tmp/solution.csv', help="\n      Path to CSV solution file, formatted with columns 'id,landmarks,Usage'\n      (the file should include a header).\n      ")
    (cmd_args, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)