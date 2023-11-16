from typing import Any, Dict, Optional, TYPE_CHECKING, List, Tuple
import os
from pathlib import Path
from collections import defaultdict
import datetime
import logging
import uuid
import yaml
import posthog
from haystack.preview.telemetry._environment import collect_system_specs
if TYPE_CHECKING:
    from haystack.preview.pipeline import Pipeline
HAYSTACK_TELEMETRY_ENABLED = 'HAYSTACK_TELEMETRY_ENABLED'
CONFIG_PATH = Path('~/.haystack/config.yaml').expanduser()
MIN_SECONDS_BETWEEN_EVENTS = 60
logger = logging.getLogger(__name__)

class Telemetry:
    """
    Haystack reports anonymous usage statistics to support continuous software improvements for all its users.

    You can opt-out of sharing usage statistics by manually setting the environment
    variable `HAYSTACK_TELEMETRY_ENABLED` as described for different operating systems on the
    [documentation page](https://docs.haystack.deepset.ai/docs/telemetry#how-can-i-opt-out).

    Check out the documentation for more details: [Telemetry](https://docs.haystack.deepset.ai/docs/telemetry).
    """

    def __init__(self):
        if False:
            return 10
        '\n        Initializes the telemetry. Loads the user_id from the config file,\n        or creates a new id and saves it if the file is not found.\n\n        It also collects system information which cannot change across the lifecycle\n        of the process (for example `is_containerized()`).\n        '
        for module_name in ['posthog', 'backoff']:
            logging.getLogger(module_name).setLevel(logging.CRITICAL)
            logging.getLogger(module_name).addHandler(logging.NullHandler())
            logging.getLogger(module_name).propagate = False
        self.user_id = None
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
                    config = yaml.safe_load(config_file)
                    if 'user_id' in config:
                        self.user_id = config['user_id']
            except Exception as e:
                logger.debug('Telemetry could not read the config file %s', CONFIG_PATH, exc_info=e)
        else:
            logger.info('Haystack sends anonymous usage data to understand the actual usage and steer dev efforts towards features that are most meaningful to users. You can opt-out at anytime by manually setting the environment variable HAYSTACK_TELEMETRY_ENABLED as described for different operating systems in the [documentation page](https://docs.haystack.deepset.ai/docs/telemetry#how-can-i-opt-out). More information at [Telemetry](https://docs.haystack.deepset.ai/docs/telemetry).')
            CONFIG_PATH.parents[0].mkdir(parents=True, exist_ok=True)
            self.user_id = str(uuid.uuid4())
            try:
                with open(CONFIG_PATH, 'w') as outfile:
                    yaml.dump({'user_id': self.user_id}, outfile, default_flow_style=False)
            except Exception as e:
                logger.debug('Telemetry could not write config file to %s', CONFIG_PATH, exc_info=e)
        self.event_properties = collect_system_specs()

    def send_event(self, event_name: str, event_properties: Optional[Dict[str, Any]]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sends a telemetry event.\n\n        :param event_name: The name of the event to show in PostHog.\n        :param event_properties: Additional event metadata. These are merged with the\n            system metadata collected in __init__, so take care not to overwrite them.\n        '
        event_properties = event_properties or {}
        try:
            posthog.capture(distinct_id=self.user_id, event=event_name, properties={**self.event_properties, **event_properties})
        except Exception as e:
            logger.debug("Telemetry couldn't make a POST request to PostHog.", exc_info=e)

def send_telemetry(func):
    if False:
        while True:
            i = 10
    '\n    Decorator that sends the output of the wrapped function to PostHog.\n    The wrapped function is actually called only if telemetry is enabled.\n    '

    def send_telemetry_wrapper(*args, **kwargs):
        if False:
            while True:
                i = 10
        try:
            if telemetry:
                output = func(*args, **kwargs)
                if output:
                    telemetry.send_event(*output)
        except Exception as e:
            logger.debug('There was an issue sending a telemetry event', exc_info=e)
    return send_telemetry_wrapper

@send_telemetry
def pipeline_running(pipeline: 'Pipeline') -> Optional[Tuple[str, Dict[str, Any]]]:
    if False:
        print('Hello World!')
    '\n    Collects name, type and the content of the _telemetry_data attribute, if present, for each component in the\n    pipeline and sends such data to Posthog.\n\n    :param pipeline: the pipeline that is running.\n    '
    pipeline._telemetry_runs += 1
    if pipeline._last_telemetry_sent and (datetime.datetime.now() - pipeline._last_telemetry_sent).seconds < MIN_SECONDS_BETWEEN_EVENTS:
        return None
    pipeline._last_telemetry_sent = datetime.datetime.now()
    pipeline_description = pipeline.to_dict()
    components: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (component_name, component) in pipeline_description['components'].items():
        instance = pipeline.get_component(component_name)
        if hasattr(instance, '_get_telemetry_data'):
            telemetry_data = getattr(instance, '_get_telemetry_data')()
            try:
                components[component['type']].append({'name': component_name, **telemetry_data})
            except TypeError:
                components[component['type']].append({'name': component_name})
        else:
            components[component['type']].append({'name': component_name})
    return ('Pipeline run (2.x)', {'pipeline_id': str(id(pipeline)), 'runs': pipeline._telemetry_runs, 'components': components})

@send_telemetry
def tutorial_running(tutorial_id: str) -> Tuple[str, Dict[str, Any]]:
    if False:
        return 10
    '\n    Send a telemetry event for a tutorial, if telemetry is enabled.\n    :param tutorial_id: identifier of the tutorial\n    '
    return ('Tutorial', {'tutorial.id': tutorial_id})
telemetry = None
if os.getenv('HAYSTACK_TELEMETRY_ENABLED', 'true').lower() in ('true', '1'):
    telemetry = Telemetry()