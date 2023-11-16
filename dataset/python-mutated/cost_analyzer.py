"""Provides a proper python API for the symbols exported through swig."""
from tensorflow.python.grappler import _pywrap_cost_analyzer as tf_wrap
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import item as gitem

def GenerateCostReport(metagraph, per_node_report=False, verbose=False, cluster=None):
    if False:
        for i in range(10):
            print('nop')
    'Analyze the cost of each TensorFlow op and node in the provided metagraph.\n\n  Args:\n    metagraph: A TensorFlow MetaGraphDef.\n    per_node_report: by default the report contains stats aggregated on a per op\n      type basis, setting per_node_report to True adds results for each\n      individual node to the report.\n    verbose: Prints out the entire operation proto instead of a summary table.\n    cluster: Analyze the costs using the specified cluster, or the local machine\n      if no cluster was specified.\n\n  Returns:\n    A string of cost report.\n  '
    if cluster is None:
        cluster = gcluster.Cluster(disable_detailed_stats=False)
    return tf_wrap.GenerateCostReport(metagraph.SerializeToString(), per_node_report, verbose, cluster.tf_cluster)

def GenerateMemoryReport(metagraph, detailed_report=True, cluster=None):
    if False:
        return 10
    'Analyze the peak memory usage for the provided metagraph.\n\n  Args:\n    metagraph: A TensorFlow MetaGraphDef.\n    detailed_report: print the live tensors in addition to the peak memory\n      usage.\n    cluster: Analyze the memory using the specified cluster, or the local\n      machine if no cluster was specified.\n\n  Returns:\n    A string with the formatted memory usage.\n  '
    if cluster is None:
        cluster = gcluster.Cluster(disable_detailed_stats=True, disable_timeline=True)
    item = gitem.Item(metagraph)
    peak_usage = cluster.DeterminePeakMemoryUsage(item)
    report = ''
    for (device, snapshot) in peak_usage.items():
        peak_usage = snapshot[0]
        report += 'Peak usage for device ' + device + ': ' + str(peak_usage) + ' bytes\n'
        if detailed_report:
            live_tensors = snapshot[1]
            for tensor in live_tensors:
                op_name = tensor[0]
                output_id = tensor[1]
                mem_used = tensor[2]
                report += '  ' + str(op_name) + ':' + str(output_id) + ' uses ' + str(mem_used) + ' bytes\n'
    return report