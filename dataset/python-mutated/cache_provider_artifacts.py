import logging
import sys
import time
from apache_beam.version import __version__ as beam_version
from apache_beam.yaml import yaml_provider

def cache_provider_artifacts():
    if False:
        print('Hello World!')
    providers_by_id = {}
    for providers in yaml_provider.standard_providers().values():
        for provider in providers:
            providers_by_id[id(provider)] = provider
    for provider in providers_by_id.values():
        t = time.time()
        artifacts = provider.cache_artifacts()
        if artifacts:
            logging.info('Cached %s in %0.03f seconds.', ', '.join(artifacts), time.time() - t)
    if '.dev' not in beam_version:
        t = time.time()
        artifacts = yaml_provider.PypiExpansionService._create_venv_to_clone(sys.executable)
        logging.info('Cached %s in %0.03f seconds.', artifacts, time.time() - t)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    cache_provider_artifacts()