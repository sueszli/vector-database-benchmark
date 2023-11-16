""" Integration tests for Dataproc samples.

Creates a Dataproc cluster, uploads a pyspark file to Google Cloud Storage,
submits a job to Dataproc that runs the pyspark file, then downloads
the output logs from Cloud Storage and verifies the expected output."""
import os
import submit_job_to_cluster
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
CLUSTER_NAME = 'testcluster3'
ZONE = 'us-central1-b'

def test_e2e():
    if False:
        i = 10
        return i + 15
    output = submit_job_to_cluster.main(PROJECT, ZONE, CLUSTER_NAME, BUCKET)
    assert b"['Hello,', 'dog', 'elephant', 'panther', 'world!']" in output