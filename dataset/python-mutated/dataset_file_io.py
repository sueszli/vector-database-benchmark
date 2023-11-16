"""IO module for files from Landmark recognition/retrieval challenges."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import tensorflow as tf
RECOGNITION_TASK_ID = 'recognition'
RETRIEVAL_TASK_ID = 'retrieval'

def ReadSolution(file_path, task):
    if False:
        i = 10
        return i + 15
    "Reads solution from file, for a given task.\n\n  Args:\n    file_path: Path to CSV file with solution. File contains a header.\n    task: Type of challenge task. Supported values: 'recognition', 'retrieval'.\n\n  Returns:\n    public_solution: Dict mapping test image ID to list of ground-truth IDs, for\n      the Public subset of test images. If `task` == 'recognition', the IDs are\n      integers corresponding to landmark IDs. If `task` == 'retrieval', the IDs\n      are strings corresponding to index image IDs.\n    private_solution: Same as `public_solution`, but for the private subset of\n      test images.\n    ignored_ids: List of test images that are ignored in scoring.\n\n  Raises:\n    ValueError: If Usage field is not Public, Private or Ignored; or if `task`\n      is not supported.\n  "
    public_solution = {}
    private_solution = {}
    ignored_ids = []
    with tf.gfile.GFile(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            test_id = row[0]
            if row[2] == 'Ignored':
                ignored_ids.append(test_id)
            else:
                ground_truth_ids = []
                if task == RECOGNITION_TASK_ID:
                    if row[1]:
                        for landmark_id in row[1].split(' '):
                            ground_truth_ids.append(int(landmark_id))
                elif task == RETRIEVAL_TASK_ID:
                    for image_id in row[1].split(' '):
                        ground_truth_ids.append(image_id)
                else:
                    raise ValueError('Unrecognized task: %s' % task)
                if row[2] == 'Public':
                    public_solution[test_id] = ground_truth_ids
                elif row[2] == 'Private':
                    private_solution[test_id] = ground_truth_ids
                else:
                    raise ValueError('Test image %s has unrecognized Usage tag %s' % (row[0], row[2]))
    return (public_solution, private_solution, ignored_ids)

def ReadPredictions(file_path, public_ids, private_ids, ignored_ids, task):
    if False:
        while True:
            i = 10
    "Reads predictions from file, for a given task.\n\n  Args:\n    file_path: Path to CSV file with predictions. File contains a header.\n    public_ids: Set (or list) of test image IDs in Public subset of test images.\n    private_ids: Same as `public_ids`, but for the private subset of test\n      images.\n    ignored_ids: Set (or list) of test image IDs that are ignored in scoring and\n      are associated to no ground-truth.\n    task: Type of challenge task. Supported values: 'recognition', 'retrieval'.\n\n  Returns:\n    public_predictions: Dict mapping test image ID to prediction, for the Public\n      subset of test images. If `task` == 'recognition', the prediction is a\n      dict with keys 'class' (integer) and 'score' (float). If `task` ==\n      'retrieval', the prediction is a list of strings corresponding to index\n      image IDs.\n    private_predictions: Same as `public_predictions`, but for the private\n      subset of test images.\n\n  Raises:\n    ValueError:\n      - If test image ID is unrecognized/repeated;\n      - If `task` is not supported;\n      - If prediction is malformed.\n  "
    public_predictions = {}
    private_predictions = {}
    with tf.gfile.GFile(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            test_id = row[0]
            if test_id in public_predictions:
                raise ValueError('Test image %s is repeated.' % test_id)
            if test_id in private_predictions:
                raise ValueError('Test image %s is repeated' % test_id)
            if test_id in ignored_ids:
                continue
            if row[1]:
                prediction_split = row[1].split(' ')
                if not prediction_split[-1]:
                    prediction_split = prediction_split[:-1]
                if task == RECOGNITION_TASK_ID:
                    if len(prediction_split) != 2:
                        raise ValueError('Prediction is malformed: there should only be 2 elements in second column, but found %d for test image %s' % (len(prediction_split), test_id))
                    landmark_id = int(prediction_split[0])
                    score = float(prediction_split[1])
                    prediction_entry = {'class': landmark_id, 'score': score}
                elif task == RETRIEVAL_TASK_ID:
                    prediction_entry = prediction_split
                else:
                    raise ValueError('Unrecognized task: %s' % task)
                if test_id in public_ids:
                    public_predictions[test_id] = prediction_entry
                elif test_id in private_ids:
                    private_predictions[test_id] = prediction_entry
                else:
                    raise ValueError('test_id %s is unrecognized' % test_id)
    return (public_predictions, private_predictions)