"""Messaging mechanism to inspect the interactive environment.

A singleton instance is accessible from
interactive_environment.current_env().inspector.
"""
import apache_beam as beam
from apache_beam.runners.interactive.utils import as_json
from apache_beam.runners.interactive.utils import obfuscate

class InteractiveEnvironmentInspector(object):
    """Inspector that converts information of the current interactive environment
  including pipelines and pcollections into JSON data suitable for messaging
  with applications within/outside the Python kernel.

  The usage is always that the application side reads the inspectables or
  list_inspectables first then communicates back to the kernel and get_val for
  usage on the kernel side.
  """

    def __init__(self, ignore_synthetic=True):
        if False:
            return 10
        self._inspectables = {}
        self._anonymous = {}
        self._inspectable_pipelines = set()
        self._ignore_synthetic = ignore_synthetic
        self._clusters = {}

    @property
    def inspectables(self):
        if False:
            i = 10
            return i + 15
        'Lists pipelines and pcollections assigned to variables as inspectables.\n    '
        self._inspectables = inspect(self._ignore_synthetic)
        return self._inspectables

    @property
    def inspectable_pipelines(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a dictionary of all inspectable pipelines. The keys are\n    stringified id of pipeline instances.\n\n    This includes user defined pipeline assigned to variables and anonymous\n    pipelines with inspectable PCollections.\n    If a user defined pipeline is not within the returned dict, it can be\n    considered out of scope, and all resources and memory states related to it\n    should be released.\n    '
        _ = self.list_inspectables()
        return self._inspectable_pipelines

    @as_json
    def list_inspectables(self):
        if False:
            for i in range(10):
                print('nop')
        "Lists inspectables in JSON format.\n\n    When listing, pcollections are organized by the pipeline they belong to.\n    If a pipeline is no longer assigned to a variable but its pcollections\n    assigned to variables are still in scope, the pipeline will be given a name\n    as 'anonymous_pipeline[id:$inMemoryId]'.\n    The listing doesn't contain object values of the pipelines or pcollections.\n    The obfuscated identifier can be used to trace back to those values in the\n    kernel.\n    The listing includes anonymous pipelines that are not assigned to variables\n    but still containing inspectable PCollections.\n    "
        listing = {}
        pipelines = inspect_pipelines()
        for (pipeline, name) in pipelines.items():
            metadata = meta(name, pipeline)
            listing[obfuscate(metadata)] = {'metadata': metadata, 'pcolls': {}}
        for (identifier, inspectable) in self.inspectables.items():
            if inspectable['metadata']['type'] == 'pcollection':
                pipeline = inspectable['value'].pipeline
                if pipeline not in list(pipelines.keys()):
                    pipeline_name = synthesize_pipeline_name(pipeline)
                    pipelines[pipeline] = pipeline_name
                    pipeline_metadata = meta(pipeline_name, pipeline)
                    pipeline_identifier = obfuscate(pipeline_metadata)
                    self._anonymous[pipeline_identifier] = {'metadata': pipeline_metadata, 'value': pipeline}
                    listing[pipeline_identifier] = {'metadata': pipeline_metadata, 'pcolls': {identifier: inspectable['metadata']}}
                else:
                    pipeline_identifier = obfuscate(meta(pipelines[pipeline], pipeline))
                    listing[pipeline_identifier]['pcolls'][identifier] = inspectable['metadata']
        self._inspectable_pipelines = dict(((str(id(pipeline)), pipeline) for pipeline in pipelines))
        return listing

    def get_val(self, identifier):
        if False:
            return 10
        'Retrieves the in memory object itself by identifier.\n\n    The retrieved object could be a pipeline or a pcollection. If the\n    identifier is not recognized, return None.\n    The identifier can refer to an anonymous pipeline and the object will still\n    be retrieved.\n    '
        inspectable = self._inspectables.get(identifier, None)
        if inspectable:
            return inspectable['value']
        inspectable = self._anonymous.get(identifier, None)
        if inspectable:
            return inspectable['value']
        return None

    def get_pcoll_data(self, identifier, include_window_info=False):
        if False:
            while True:
                i = 10
        'Retrieves the json formatted PCollection data.\n\n    If no PCollection value can be retieved from the given identifier, an empty\n    json string will be returned.\n    '
        value = self.get_val(identifier)
        if isinstance(value, beam.pvalue.PCollection):
            from apache_beam.runners.interactive import interactive_beam as ib
            dataframe = ib.collect(value, include_window_info=include_window_info)
            return dataframe.to_json(orient='table')
        return {}

    @as_json
    def list_clusters(self):
        if False:
            return 10
        'Retrieves information for all clusters as a json.\n\n    The json object maps a unique obfuscated identifier of a cluster to\n    the corresponding cluster_name, project, region, master_url, dashboard,\n    and pipelines. Furthermore, copies the mapping to self._clusters.\n    '
        from apache_beam.runners.interactive import interactive_environment as ie
        clusters = ie.current_env().clusters
        all_cluster_data = {}
        for (meta, dcm) in clusters.dataproc_cluster_managers.items():
            all_cluster_data[obfuscate(meta)] = {'cluster_name': meta.cluster_name, 'project': meta.project_id, 'region': meta.region, 'master_url': meta.master_url, 'dashboard': meta.dashboard, 'pipelines': [str(id(p)) for p in dcm.pipelines]}
        self._clusters = all_cluster_data
        return all_cluster_data

    def get_cluster_master_url(self, identifier: str) -> str:
        if False:
            i = 10
            return i + 15
        'Returns the master_url corresponding to the obfuscated identifier.'
        return self._clusters[identifier]['master_url']

def inspect(ignore_synthetic=True):
    if False:
        print('Hello World!')
    'Inspects current interactive environment to track metadata and values of\n  pipelines and pcollections.\n\n  Each pipeline and pcollections tracked is given a unique identifier.\n  '
    from apache_beam.runners.interactive import interactive_environment as ie
    inspectables = {}
    for watching in ie.current_env().watching():
        for (name, value) in watching:
            if ignore_synthetic and name.startswith('synthetic_var_'):
                continue
            metadata = meta(name, value)
            identifier = obfuscate(metadata)
            if isinstance(value, (beam.pipeline.Pipeline, beam.pvalue.PCollection)):
                inspectables[identifier] = {'metadata': metadata, 'value': value}
    return inspectables

def inspect_pipelines():
    if False:
        print('Hello World!')
    'Inspects current interactive environment to track all pipelines assigned\n  to variables. The keys are pipeline objects and values are pipeline names.\n  '
    from apache_beam.runners.interactive import interactive_environment as ie
    pipelines = {}
    for watching in ie.current_env().watching():
        for (name, value) in watching:
            if isinstance(value, beam.pipeline.Pipeline):
                pipelines[value] = name
    return pipelines

def meta(name, val):
    if False:
        print('Hello World!')
    'Generates meta data for the given name and value.'
    return {'name': name, 'inMemoryId': id(val), 'type': type(val).__name__.lower()}

def synthesize_pipeline_name(val):
    if False:
        i = 10
        return i + 15
    'Synthesizes a pipeline name for the given pipeline object.'
    return 'anonymous_pipeline[id:{}]'.format(id(val))