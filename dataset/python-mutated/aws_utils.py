import re
import requests
from metaflow.exception import MetaflowException

def get_ec2_instance_metadata():
    if False:
        print('Hello World!')
    '\n    Fetches the EC2 instance metadata through AWS instance metadata service\n\n    Returns either an empty dictionary, or one with the keys\n        - ec2-instance-id\n        - ec2-instance-type\n        - ec2-region\n        - ec2-availability-zone\n    '
    meta = {}
    try:
        instance_meta = requests.get(url='http://169.254.169.254/latest/dynamic/instance-identity/document', timeout=(1, 10)).json()
        meta['ec2-instance-id'] = instance_meta.get('instanceId')
        meta['ec2-instance-type'] = instance_meta.get('instanceType')
        meta['ec2-region'] = instance_meta.get('region')
        meta['ec2-availability-zone'] = instance_meta.get('availabilityZone')
    except:
        pass
    return meta

def get_docker_registry(image_uri):
    if False:
        while True:
            i = 10
    '\n    Explanation:\n        (.+?(?:[:.].+?)\\/)? - [GROUP 0] REGISTRY\n            .+?                 - A registry must start with at least one character\n            (?:[:.].+?)\\/       - A registry must have ":" or "." and end with "/"\n            ?                   - Make a registry optional\n        (.*?)               - [GROUP 1] REPOSITORY\n            .*?                 - Get repository name until separator\n        (?:[@:])?           - SEPARATOR\n            ?:                  - Don\'t capture separator\n            [@:]                - The separator must be either "@" or ":"\n            ?                   - The separator is optional\n        ((?<=[@:]).*)?      - [GROUP 2] TAG / DIGEST\n            (?<=[@:])           - A tag / digest must be preceded by "@" or ":"\n            .*                  - Capture rest of tag / digest\n            ?                   - A tag / digest is optional\n    Examples:\n        image\n            - None\n            - image\n            - None\n        example/image\n            - None\n            - example/image\n            - None\n        example/image:tag\n            - None\n            - example/image\n            - tag\n        example.domain.com/example/image:tag\n            - example.domain.com/\n            - example/image\n            - tag\n        123.123.123.123:123/example/image:tag\n            - 123.123.123.123:123/\n            - example/image\n            - tag\n        example.domain.com/example/image@sha256:45b23dee0\n            - example.domain.com/\n            - example/image\n            - sha256:45b23dee0\n    '
    pattern = re.compile('^(.+?(?:[:.].+?)\\/)?(.*?)(?:[@:])?((?<=[@:]).*)?$')
    (registry, repository, tag) = pattern.match(image_uri).groups()
    if registry is not None:
        registry = registry.rstrip('/')
    return registry

def compute_resource_attributes(decos, compute_deco, resource_defaults):
    if False:
        while True:
            i = 10
    '\n    Compute resource values taking into account defaults, the values specified\n    in the compute decorator (like @batch or @kubernetes) directly, and\n    resources specified via @resources decorator.\n\n    Returns a dictionary of resource attr -> value (str).\n    '
    assert compute_deco is not None
    supported_keys = set([*resource_defaults.keys(), *compute_deco.attributes.keys()])
    result = {k: v for (k, v) in resource_defaults.items() if v is not None}
    for deco in decos:
        if deco.name == 'resources':
            for (k, v) in deco.attributes.items():
                my_val = compute_deco.attributes.get(k)
                if k not in supported_keys:
                    continue
                if my_val is None and v is None:
                    continue
                if my_val is not None and v is not None:
                    try:
                        result[k] = str(max(float(my_val or 0), float(v or 0)))
                    except ValueError:
                        if my_val != v:
                            raise MetaflowException("'resources' and compute decorator have conflicting values for '%s'. Please use consistent values or specify this resource constraint once" % k)
                elif my_val is not None:
                    result[k] = str(my_val or '0')
                else:
                    result[k] = str(v or '0')
            return result
    for k in resource_defaults:
        if compute_deco.attributes.get(k) is not None:
            result[k] = str(compute_deco.attributes[k] or '0')
    return result

def sanitize_batch_tag(key, value):
    if False:
        while True:
            i = 10
    '\n    Sanitize a key and value for use as a Batch tag.\n    '
    RE_NOT_PERMITTED = '[^A-Za-z0-9\\s\\+\\-\\=\\.\\_\\:\\/\\@]'
    _key = re.sub(RE_NOT_PERMITTED, '', key)[:128]
    _value = re.sub(RE_NOT_PERMITTED, '', value)[:256]
    return (_key, _value)