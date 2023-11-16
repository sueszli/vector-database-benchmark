import os
import yaml
from azureml.core.conda_dependencies import CondaDependencies

def generate_yaml(directory: str, ref_filename: str, needed_libraries: list, conda_filename: str):
    if False:
        i = 10
        return i + 15
    '\n    Creates a deployment-specific yaml file as a subset of\n    the image classification environment.yml\n\n    Also adds extra libraries, if not present in environment.yml\n\n    Args:\n        directory (string): Directory name of reference yaml file\n        ref_filename (string): Name of reference yaml file\n        needed_libraries (list of strings): List of libraries needed\n        in the Docker container\n        conda_filename (string): Name of yaml file to be deployed\n        in the Docker container\n\n    Returns: Nothing\n\n    '
    with open(os.path.join(directory, ref_filename), 'r') as f:
        yaml_content = yaml.load(f, Loader=yaml.SafeLoader)
    extracted_libraries = [depend for depend in yaml_content['dependencies'] if any((lib in depend for lib in needed_libraries))]
    if any((isinstance(x, dict) for x in yaml_content['dependencies'])):
        ind = [yaml_content['dependencies'].index(depend) for depend in yaml_content['dependencies'] if isinstance(depend, dict)][0]
        extracted_libraries += [depend for depend in yaml_content['dependencies'][ind]['pip'] if any((lib in depend for lib in needed_libraries))]
    not_found = [lib for lib in needed_libraries if not any((lib in ext for ext in extracted_libraries))]
    conda_env = CondaDependencies()
    for ch in yaml_content['channels']:
        conda_env.add_channel(ch)
    for library in extracted_libraries + not_found:
        conda_env.add_conda_package(library)
    print(conda_env.serialize_to_string())
    conda_env.save_to_file(base_directory=os.getcwd(), conda_file_path=conda_filename)