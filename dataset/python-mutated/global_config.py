import os
import yaml
from typing_extensions import TypedDict

class GlobalConfig(TypedDict):
    byod_ray_ecr: str
    byod_ray_cr_repo: str
    byod_ray_ml_cr_repo: str
    byod_ecr: str
    byod_aws_cr: str
    byod_gcp_cr: str
    state_machine_aws_bucket: str
    aws2gce_credentials: str
config = None

def init_global_config(config_file: str):
    if False:
        i = 10
        return i + 15
    '\n    Initiate the global configuration singleton.\n    '
    global config
    if not config:
        _init_global_config(config_file)

def get_global_config():
    if False:
        i = 10
        return i + 15
    '\n    Get the global configuration singleton. Need to be invoked after\n    init_global_config().\n    '
    global config
    return config

def _init_global_config(config_file: str):
    if False:
        return 10
    global config
    config_content = yaml.safe_load(open(config_file, 'rt'))
    config = GlobalConfig(byod_ray_ecr=config_content['byod']['ray_ecr'], byod_ray_cr_repo=config_content['byod']['ray_cr_repo'], byod_ray_ml_cr_repo=config_content['byod']['ray_ml_cr_repo'], byod_ecr=config_content['byod']['byod_ecr'], byod_aws_cr=config_content['byod'].get('aws_cr'), byod_gcp_cr=config_content['byod'].get('gcp_cr'), state_machine_aws_bucket=config_content['state_machine']['aws_bucket'], aws2gce_credentials=config_content.get('credentials', {}).get('aws2gce'))
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f"/workdir/{config['aws2gce_credentials']}"