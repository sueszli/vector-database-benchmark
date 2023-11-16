import re
import sys
from google.cloud import dataproc_v1 as dataproc
from google.cloud import storage

def submit_job(project_id, region, cluster_name):
    if False:
        print('Hello World!')
    job_client = dataproc.JobControllerClient(client_options={'api_endpoint': f'{region}-dataproc.googleapis.com:443'})
    job = {'placement': {'cluster_name': cluster_name}, 'spark_job': {'main_class': 'org.apache.spark.examples.SparkPi', 'jar_file_uris': ['file:///usr/lib/spark/examples/jars/spark-examples.jar'], 'args': ['1000']}}
    operation = job_client.submit_job_as_operation(request={'project_id': project_id, 'region': region, 'job': job})
    response = operation.result()
    matches = re.match('gs://(.*?)/(.*)', response.driver_output_resource_uri)
    output = storage.Client().get_bucket(matches.group(1)).blob(f'{matches.group(2)}.000000000').download_as_bytes().decode('utf-8')
    print(f'Job finished successfully: {output}')
if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('python submit_job.py project_id region cluster_name')
    project_id = sys.argv[1]
    region = sys.argv[2]
    cluster_name = sys.argv[3]
    submit_job(project_id, region, cluster_name)