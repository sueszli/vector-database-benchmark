from typing import Optional, Mapping, Any, Union

class ProjectRuntimeInfo:
    """A description of the Project's program runtime and associated metadata."""
    name: str
    options: Optional[Mapping[str, Any]]

    def __init__(self, name: str, options: Optional[Mapping[str, Any]]=None):
        if False:
            while True:
                i = 10
        self.name = name
        self.options = options

class ProjectTemplateConfigValue:
    """A placeholder config value for a project template."""
    description: Optional[str]
    default: Optional[str]
    secret: bool

    def __init__(self, description: Optional[str]=None, default: Optional[str]=None, secret: bool=False):
        if False:
            return 10
        self.description = description
        self.default = default
        self.secret = secret

class ProjectTemplate:
    """A template used to seed new stacks created from this project."""
    description: Optional[str]
    quickstart: Optional[str]
    config: Mapping[str, ProjectTemplateConfigValue]
    important: Optional[bool]

    def __init__(self, description: Optional[str]=None, quickstart: Optional[str]=None, config: Optional[Mapping[str, ProjectTemplateConfigValue]]=None, important: Optional[bool]=None):
        if False:
            return 10
        self.description = description
        self.quickstart = quickstart
        self.config = config or {}
        self.important = important

class ProjectBackend:
    """Configuration for the project's Pulumi state storage backend."""
    url: Optional[str]

    def __init__(self, url: Optional[str]=None):
        if False:
            return 10
        self.url = url

class ProjectSettings:
    """A Pulumi project manifest. It describes metadata applying to all sub-stacks created from the project."""
    name: str
    runtime: Union[str, ProjectRuntimeInfo]
    main: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    website: Optional[str] = None
    license: Optional[str] = None
    config: Optional[str] = None
    template: Optional[ProjectTemplate] = None
    backend: Optional[ProjectBackend] = None

    def __init__(self, name: str, runtime: Union[str, ProjectRuntimeInfo], main: Optional[str]=None, description: Optional[str]=None, author: Optional[str]=None, website: Optional[str]=None, license: Optional[str]=None, config: Optional[str]=None, template: Optional[ProjectTemplate]=None, backend: Optional[ProjectBackend]=None):
        if False:
            print('Hello World!')
        if isinstance(runtime, str) and runtime not in ['nodejs', 'python', 'go', 'dotnet']:
            raise ValueError(f"Invalid value {runtime!r} for runtime. Must be one of: 'nodejs', 'python', 'go', 'dotnet'.")
        self.name = name
        self.runtime = runtime
        self.main = main
        self.description = description
        self.author = author
        self.website = website
        self.license = license
        self.config = config
        self.template = template
        self.backend = backend