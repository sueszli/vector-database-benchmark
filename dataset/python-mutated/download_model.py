"""Loads the state_dict for an LLM model into Cloud Storage."""
from __future__ import annotations
import os
import torch
from transformers import AutoModelForSeq2SeqLM

def run_local(model_name: str, state_dict_path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Loads the state dict and saves it into the desired path.\n\n    If the `state_dict_path` is a Cloud Storage location starting\n    with "gs://", this assumes Cloud Storage is mounted with\n    Cloud Storage FUSE in `/gcs`. Vertex AI is set up like this.\n\n    Args:\n        model_name: HuggingFace model name compatible with AutoModelForSeq2SeqLM.\n        state_dict_path: File path to the model\'s state_dict, can be in Cloud Storage.\n    '
    print(f'Loading model: {model_name}')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    print(f'Model loaded, saving state dict to: {state_dict_path}')
    state_dict_path = state_dict_path.replace('gs://', '/gcs/')
    directory = os.path.dirname(state_dict_path)
    if directory and (not os.path.exists(directory)):
        os.makedirs(os.path.dirname(state_dict_path), exist_ok=True)
    torch.save(model.state_dict(), state_dict_path)
    print('State dict saved successfully!')

def run_vertex_job(model_name: str, state_dict_path: str, job_name: str, project: str, bucket: str, location: str='us-central1', machine_type: str='e2-highmem-2', disk_size_gb: int=100) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Launches a Vertex AI custom job to load the state dict.\n\n    If the model is too large to fit into memory or disk, we can launch\n    a Vertex AI custom job with a large enough VM for this to work.\n\n    Depending on the model\'s size, it might require a different VM\n    configuration. The model MUST fit into the VM\'s memory, and there\n    must be enough disk space to stage the entire model while it gets\n    copied to Cloud Storage.\n\n    Args:\n        model_name: HuggingFace model name compatible with AutoModelForSeq2SeqLM.\n        state_dict_path: File path to the model\'s state_dict, can be in Cloud Storage.\n        job_name: Job display name in the Vertex AI console.\n        project: Google Cloud Project ID.\n        bucket: Cloud Storage bucket name, without the "gs://" prefix.\n        location: Google Cloud regional location.\n        machine_type: Machine type for the VM to run the job.\n        disk_size_gb: Disk size in GB for the VM to run the job.\n    '
    from google.cloud import aiplatform
    aiplatform.init(project=project, staging_bucket=bucket, location=location)
    job = aiplatform.CustomJob.from_local_script(display_name=job_name, container_uri='us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest', script_path='download_model.py', args=['local', f'--model-name={model_name}', f'--state-dict-path={state_dict_path}'], machine_type=machine_type, boot_disk_size_gb=disk_size_gb, requirements=['transformers'])
    job.run()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    parser_local = subparsers.add_parser('local')
    parser_local.add_argument('--model-name', required=True, help='HuggingFace model name compatible with AutoModelForSeq2SeqLM')
    parser_local.add_argument('--state-dict-path', required=True, help="File path to the model's state_dict, can be in Cloud Storage")
    parser_local.set_defaults(run=run_local)
    parser_vertex = subparsers.add_parser('vertex')
    parser_vertex.add_argument('--model-name', required=True, help='HuggingFace model name compatible with AutoModelForSeq2SeqLM')
    parser_vertex.add_argument('--state-dict-path', required=True, help="File path to the model's state_dict, can be in Cloud Storage")
    parser_vertex.add_argument('--job-name', required=True, help='Job display name in the Vertex AI console')
    parser_vertex.add_argument('--project', required=True, help='Google Cloud Project ID')
    parser_vertex.add_argument('--bucket', required=True, help='Cloud Storage bucket name, without the "gs://" prefix')
    parser_vertex.add_argument('--location', default='us-central1', help='Google Cloud regional location')
    parser_vertex.add_argument('--machine-type', default='e2-highmem-2', help='Machine type for the VM to run the job')
    parser_vertex.add_argument('--disk-size-gb', type=int, default=100, help='Disk size in GB for the VM to run the job')
    parser_vertex.set_defaults(run=run_vertex_job)
    args = parser.parse_args()
    kwargs = args.__dict__.copy()
    kwargs.pop('run')
    args.run(**kwargs)