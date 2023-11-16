import collections
import collections.abc
import copy
import hashlib
import spack.util.spack_yaml as syaml

def sort_yaml_obj(obj):
    if False:
        while True:
            i = 10
    if isinstance(obj, collections.abc.Mapping):
        return syaml.syaml_dict(((k, sort_yaml_obj(v)) for (k, v) in sorted(obj.items(), key=lambda item: str(item[0]))))
    if isinstance(obj, collections.abc.Sequence) and (not isinstance(obj, str)):
        return syaml.syaml_list((sort_yaml_obj(x) for x in obj))
    return obj

def matches(obj, proto):
    if False:
        print('Hello World!')
    'Returns True if the test object "obj" matches the prototype object\n    "proto".\n\n    If obj and proto are mappings, obj matches proto if (key in obj) and\n    (obj[key] matches proto[key]) for every key in proto.\n\n    If obj and proto are sequences, obj matches proto if they are of the same\n    length and (a matches b) for every (a,b) in zip(obj, proto).\n\n    Otherwise, obj matches proto if obj == proto.\n\n    Precondition: proto must not have any reference cycles\n    '
    if isinstance(obj, collections.abc.Mapping):
        if not isinstance(proto, collections.abc.Mapping):
            return False
        return all((key in obj and matches(obj[key], val) for (key, val) in proto.items()))
    if isinstance(obj, collections.abc.Sequence) and (not isinstance(obj, str)):
        if not (isinstance(proto, collections.abc.Sequence) and (not isinstance(proto, str))):
            return False
        if len(obj) != len(proto):
            return False
        return all((matches(obj[index], val) for (index, val) in enumerate(proto)))
    return obj == proto

def subkeys(obj, proto):
    if False:
        return 10
    'Returns the test mapping "obj" after factoring out the items it has in\n    common with the prototype mapping "proto".\n\n    Consider a recursive merge operation, merge(a, b) on mappings a and b, that\n    returns a mapping, m, whose keys are the union of the keys of a and b, and\n    for every such key, "k", its corresponding value is:\n\n      - merge(a[key], b[key])  if a[key] and b[key] are mappings, or\n      - b[key]                 if (key in b) and not matches(a[key], b[key]),\n                               or\n      - a[key]                 otherwise\n\n\n    If obj and proto are mappings, the returned object is the smallest object,\n    "a", such that merge(a, proto) matches obj.\n\n    Otherwise, obj is returned.\n    '
    if not (isinstance(obj, collections.abc.Mapping) and isinstance(proto, collections.abc.Mapping)):
        return obj
    new_obj = {}
    for (key, value) in obj.items():
        if key not in proto:
            new_obj[key] = value
            continue
        if matches(value, proto[key]) and matches(proto[key], value):
            continue
        if isinstance(value, collections.abc.Mapping):
            new_obj[key] = subkeys(value, proto[key])
            continue
        new_obj[key] = value
    return new_obj

def add_extends(yaml, key):
    if False:
        while True:
            i = 10
    'Modifies the given object "yaml" so that it includes an "extends" key\n    whose value features "key".\n\n    If "extends" is not in yaml, then yaml is modified such that\n    yaml["extends"] == key.\n\n    If yaml["extends"] is a str, then yaml is modified such that\n    yaml["extends"] == [yaml["extends"], key]\n\n    If yaml["extends"] is a list that does not include key, then key is\n    appended to the list.\n\n    Otherwise, yaml is left unchanged.\n    '
    has_key = 'extends' in yaml
    extends = yaml.get('extends')
    if has_key and (not isinstance(extends, (str, collections.abc.Sequence))):
        return
    if extends is None:
        yaml['extends'] = key
        return
    if isinstance(extends, str):
        if extends != key:
            yaml['extends'] = [extends, key]
        return
    if key not in extends:
        extends.append(key)

def common_subobject(yaml, sub):
    if False:
        for i in range(10):
            print('nop')
    'Factor prototype object "sub" out of the values of mapping "yaml".\n\n    Consider a modified copy of yaml, "new", where for each key, "key" in yaml:\n\n      - If yaml[key] matches sub, then new[key] = subkeys(yaml[key], sub).\n      - Otherwise, new[key] = yaml[key].\n\n    If the above match criteria is not satisfied for any such key, then (yaml,\n    None) is returned. The yaml object is returned unchanged.\n\n    Otherwise, each matching value in new is modified as in\n    add_extends(new[key], common_key), and then new[common_key] is set to sub.\n    The common_key value is chosen such that it does not match any preexisting\n    key in new. In this case, (new, common_key) is returned.\n    '
    match_list = set((k for (k, v) in yaml.items() if matches(v, sub)))
    if not match_list:
        return (yaml, None)
    common_prefix = '.c'
    common_index = 0
    while True:
        common_key = ''.join((common_prefix, str(common_index)))
        if common_key not in yaml:
            break
        common_index += 1
    new_yaml = {}
    for (key, val) in yaml.items():
        new_yaml[key] = copy.deepcopy(val)
        if not matches(val, sub):
            continue
        new_yaml[key] = subkeys(new_yaml[key], sub)
        add_extends(new_yaml[key], common_key)
    new_yaml[common_key] = sub
    return (new_yaml, common_key)

def print_delta(name, old, new, applied=None):
    if False:
        print('Hello World!')
    delta = new - old
    reldelta = 1000 * delta // old
    reldelta = (reldelta // 10, reldelta % 10)
    if applied is None:
        applied = new <= old
    print('\n'.join(('{0} {1}:', '  before: {2: 10d}', '  after : {3: 10d}', '  delta : {4:+10d} ({5:=+3d}.{6}%)')).format(name, '+' if applied else 'x', old, new, delta, reldelta[0], reldelta[1]))

def try_optimization_pass(name, yaml, optimization_pass, *args, **kwargs):
    if False:
        print('Hello World!')
    'Try applying an optimization pass and return information about the\n    result\n\n    "name" is a string describing the nature of the pass. If it is a non-empty\n    string, summary statistics are also printed to stdout.\n\n    "yaml" is the object to apply the pass to.\n\n    "optimization_pass" is the function implementing the pass to be applied.\n\n    "args" and "kwargs" are the additional arguments to pass to optimization\n    pass. The pass is applied as\n\n    >>> (new_yaml, *other_results) = optimization_pass(yaml, *args, **kwargs)\n\n    The pass\'s results are greedily rejected if it does not modify the original\n    yaml document, or if it produces a yaml document that serializes to a\n    larger string.\n\n    Returns (new_yaml, yaml, applied, other_results) if applied, or\n    (yaml, new_yaml, applied, other_results) otherwise.\n    '
    result = optimization_pass(yaml, *args, **kwargs)
    (new_yaml, other_results) = (result[0], result[1:])
    if new_yaml is yaml:
        return (yaml, new_yaml, False, other_results)
    pre_size = len(syaml.dump_config(sort_yaml_obj(yaml), default_flow_style=True))
    post_size = len(syaml.dump_config(sort_yaml_obj(new_yaml), default_flow_style=True))
    applied = post_size <= pre_size
    if applied:
        (yaml, new_yaml) = (new_yaml, yaml)
    if name:
        print_delta(name, pre_size, post_size, applied)
    return (yaml, new_yaml, applied, other_results)

def build_histogram(iterator, key):
    if False:
        print('Hello World!')
    'Builds a histogram of values given an iterable of mappings and a key.\n\n    For each mapping "m" with key "key" in iterator, the value m[key] is\n    considered.\n\n    Returns a list of tuples (hash, count, proportion, value), where\n\n      - "hash" is a sha1sum hash of the value.\n      - "count" is the number of occurences of values that hash to "hash".\n      - "proportion" is the proportion of all values considered above that\n        hash to "hash".\n      - "value" is one of the values considered above that hash to "hash".\n        Which value is chosen when multiple values hash to the same "hash" is\n        undefined.\n\n    The list is sorted in descending order by count, yielding the most\n    frequently occuring hashes first.\n    '
    buckets = collections.defaultdict(int)
    values = {}
    num_objects = 0
    for obj in iterator:
        num_objects += 1
        try:
            val = obj[key]
        except (KeyError, TypeError):
            continue
        value_hash = hashlib.sha1()
        value_hash.update(syaml.dump_config(sort_yaml_obj(val)).encode())
        value_hash = value_hash.hexdigest()
        buckets[value_hash] += 1
        values[value_hash] = val
    return [(h, buckets[h], float(buckets[h]) / num_objects, values[h]) for h in sorted(buckets.keys(), key=lambda k: -buckets[k])]

def optimizer(yaml):
    if False:
        while True:
            i = 10
    original_size = len(syaml.dump_config(sort_yaml_obj(yaml), default_flow_style=True))
    common_job = {'variables': {'SPACK_COMPILER_ACTION': 'NONE'}, 'after_script': ['rm -rf "./spack"'], 'artifacts': {'paths': ['jobs_scratch_dir', 'cdash_report'], 'when': 'always'}}
    (_, count, proportion, tags) = next(iter(build_histogram(yaml.values(), 'tags')), (None,) * 4)
    if tags and count > 1 and (proportion >= 0.7):
        common_job['tags'] = tags
    (yaml, other, applied, rest) = try_optimization_pass('general common object factorization', yaml, common_subobject, common_job)
    (_, count, proportion, script) = next(iter(build_histogram(yaml.values(), 'script')), (None,) * 4)
    if script and count > 1 and (proportion >= 0.7):
        (yaml, other, applied, rest) = try_optimization_pass('script factorization', yaml, common_subobject, {'script': script})
    (_, count, proportion, script) = next(iter(build_histogram(yaml.values(), 'before_script')), (None,) * 4)
    if script and count > 1 and (proportion >= 0.7):
        (yaml, other, applied, rest) = try_optimization_pass('before_script factorization', yaml, common_subobject, {'before_script': script})
    h = build_histogram((getattr(val, 'get', lambda *args: {})('variables') for val in yaml.values()), 'SPACK_ROOT_SPEC')
    counter = 0
    for (_, count, proportion, spec) in h:
        if count <= 1:
            continue
        counter += 1
        (yaml, other, applied, rest) = try_optimization_pass('SPACK_ROOT_SPEC factorization ({count})'.format(count=counter), yaml, common_subobject, {'variables': {'SPACK_ROOT_SPEC': spec}})
    new_size = len(syaml.dump_config(sort_yaml_obj(yaml), default_flow_style=True))
    print('\n')
    print_delta('overall summary', original_size, new_size)
    print('\n')
    return yaml