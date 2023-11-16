"""Common IO utils used in offline metric computation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv

def write_csv(fid, metrics):
    if False:
        return 10
    'Writes metrics key-value pairs to CSV file.\n\n  Args:\n    fid: File identifier of an opened file.\n    metrics: A dictionary with metrics to be written.\n  '
    metrics_writer = csv.writer(fid, delimiter=',')
    for (metric_name, metric_value) in metrics.items():
        metrics_writer.writerow([metric_name, str(metric_value)])