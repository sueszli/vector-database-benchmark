import copy
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from .resource import Resource, ProviderResource

class InvokeOptions:
    """
    InvokeOptions is a bag of options that control the behavior of a call to runtime.invoke.
    """
    parent: Optional['Resource']
    '\n    An optional parent to use for default options for this invoke (e.g. the default provider to use).\n    '
    provider: Optional['ProviderResource']
    "\n    An optional provider to use for this invocation. If no provider is supplied, the default provider for the\n    invoked function's package will be used.\n    "
    version: Optional[str]
    '\n    An optional version. If provided, the provider plugin with exactly this version will be used to service\n    the invocation.\n    '
    plugin_download_url: Optional[str]
    '\n    An optional URL. If provided, the provider plugin with exactly this download URL will be used to service\n    the invocation. This will override the URL sourced from the host package, and should be rarely used.\n    '

    def __init__(self, parent: Optional['Resource']=None, provider: Optional['ProviderResource']=None, version: Optional[str]='', plugin_download_url: Optional[str]=None) -> None:
        if False:
            return 10
        "\n        :param Optional[Resource] parent: An optional parent to use for default options for this invoke (e.g. the\n               default provider to use).\n        :param Optional[ProviderResource] provider: An optional provider to use for this invocation. If no provider is\n               supplied, the default provider for the invoked function's package will be used.\n        :param Optional[str] version: An optional version. If provided, the provider plugin with exactly this version\n               will be used to service the invocation.\n        :param Optional[str] plugin_download_url: An optional URL. If provided, the provider plugin with this download\n               URL will be used to service the invocation. This will override the URL sourced from the host package, and\n               should be rarely used.\n        "
        self.merge = self._merge_instance
        self.merge.__func__.__doc__ = InvokeOptions.merge.__doc__
        self.parent = parent
        self.provider = provider
        self.version = version
        self.plugin_download_url = plugin_download_url

    def _merge_instance(self, opts: 'InvokeOptions') -> 'InvokeOptions':
        if False:
            return 10
        return InvokeOptions.merge(self, opts)

    @staticmethod
    def merge(opts1: Optional['InvokeOptions'], opts2: Optional['InvokeOptions']) -> 'InvokeOptions':
        if False:
            i = 10
            return i + 15
        "\n        merge produces a new InvokeOptions object with the respective attributes of the `opts1`\n        instance in it with the attributes of `opts2` merged over them.\n\n        Both the `opts1` instance and the `opts2` instance will be unchanged.  Both of `opts1` and\n        `opts2` can be `None`, in which case its attributes are ignored.\n\n        Conceptually attributes merging follows these basic rules:\n\n        1. If the attributes is a collection, the final value will be a collection containing the\n           values from each options object. Both original collections in each options object will\n           be unchanged.\n\n        2. Simple scalar values from `opts2` (i.e. strings, numbers, bools) will replace the values\n           from `opts1`.\n\n        3. For the purposes of merging `depends_on` is always treated\n           as collections, even if only a single value was provided.\n\n        4. Attributes with value 'None' will not be copied over.\n\n        This method can be called either as static-method like `InvokeOptions.merge(opts1, opts2)`\n        or as an instance-method like `opts1.merge(opts2)`.  The former is useful for cases where\n        `opts1` may be `None` so the caller does not need to check for this case.\n        "
        opts1 = InvokeOptions() if opts1 is None else opts1
        opts2 = InvokeOptions() if opts2 is None else opts2
        if not isinstance(opts1, InvokeOptions):
            raise TypeError('Expected opts1 to be a InvokeOptions instance')
        if not isinstance(opts2, InvokeOptions):
            raise TypeError('Expected opts2 to be a InvokeOptions instance')
        dest = copy.copy(opts1)
        source = opts2
        dest.parent = dest.parent if source.parent is None else source.parent
        dest.provider = dest.provider if source.provider is None else source.provider
        dest.plugin_download_url = dest.plugin_download_url if source.plugin_download_url is None else source.plugin_download_url
        dest.version = dest.version if source.version is None else source.version
        return dest