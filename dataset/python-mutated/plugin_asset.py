"""TensorBoard Plugin asset abstract class.

TensorBoard plugins may need to provide arbitrary assets, such as
configuration information for specific outputs, or vocabulary files, or sprite
images, etc.

This module contains methods that allow plugin assets to be specified at graph
construction time. Plugin authors define a PluginAsset which is treated as a
singleton on a per-graph basis. The PluginAsset has an assets method which
returns a dictionary of asset contents. The tf.compat.v1.summary.FileWriter
(or any other Summary writer) will serialize these assets in such a way that
TensorBoard can retrieve them.
"""
import abc
from tensorflow.python.framework import ops
_PLUGIN_ASSET_PREFIX = '__tensorboard_plugin_asset__'

def get_plugin_asset(plugin_asset_cls, graph=None):
    if False:
        while True:
            i = 10
    'Acquire singleton PluginAsset instance from a graph.\n\n  PluginAssets are always singletons, and are stored in tf Graph collections.\n  This way, they can be defined anywhere the graph is being constructed, and\n  if the same plugin is configured at many different points, the user can always\n  modify the same instance.\n\n  Args:\n    plugin_asset_cls: The PluginAsset class\n    graph: (optional) The graph to retrieve the instance from. If not specified,\n      the default graph is used.\n\n  Returns:\n    An instance of the plugin_asset_class\n\n  Raises:\n    ValueError: If we have a plugin name collision, or if we unexpectedly find\n      the wrong number of items in a collection.\n  '
    if graph is None:
        graph = ops.get_default_graph()
    if not plugin_asset_cls.plugin_name:
        raise ValueError('Class %s has no plugin_name' % plugin_asset_cls.__name__)
    name = _PLUGIN_ASSET_PREFIX + plugin_asset_cls.plugin_name
    container = graph.get_collection(name)
    if container:
        if len(container) != 1:
            raise ValueError('Collection for %s had %d items, expected 1' % (name, len(container)))
        instance = container[0]
        if not isinstance(instance, plugin_asset_cls):
            raise ValueError('Plugin name collision between classes %s and %s' % (plugin_asset_cls.__name__, instance.__class__.__name__))
    else:
        instance = plugin_asset_cls()
        graph.add_to_collection(name, instance)
        graph.add_to_collection(_PLUGIN_ASSET_PREFIX, plugin_asset_cls.plugin_name)
    return instance

def get_all_plugin_assets(graph=None):
    if False:
        while True:
            i = 10
    'Retrieve all PluginAssets stored in the graph collection.\n\n  Args:\n    graph: Optionally, the graph to get assets from. If unspecified, the default\n      graph is used.\n\n  Returns:\n    A list with all PluginAsset instances in the graph.\n\n  Raises:\n    ValueError: if we unexpectedly find a collection with the wrong number of\n      PluginAssets.\n\n  '
    if graph is None:
        graph = ops.get_default_graph()
    out = []
    for name in graph.get_collection(_PLUGIN_ASSET_PREFIX):
        collection = graph.get_collection(_PLUGIN_ASSET_PREFIX + name)
        if len(collection) != 1:
            raise ValueError('Collection for %s had %d items, expected 1' % (name, len(collection)))
        out.append(collection[0])
    return out

class PluginAsset(metaclass=abc.ABCMeta):
    """This abstract base class allows TensorBoard to serialize assets to disk.

  Plugin authors are expected to extend the PluginAsset class, so that it:
  - has a unique plugin_name
  - provides an assets method that returns an {asset_name: asset_contents}
    dictionary. For now, asset_contents are strings, although we may add
    StringIO support later.

  LifeCycle of a PluginAsset instance:
  - It is constructed when get_plugin_asset is called on the class for
    the first time.
  - It is configured by code that follows the calls to get_plugin_asset
  - When the containing graph is serialized by the
    tf.compat.v1.summary.FileWriter, the writer calls assets and the
    PluginAsset instance provides its contents to be written to disk.
  """
    plugin_name = None

    @abc.abstractmethod
    def assets(self):
        if False:
            for i in range(10):
                print('nop')
        'Provide all of the assets contained by the PluginAsset instance.\n\n    The assets method should return a dictionary structured as\n    {asset_name: asset_contents}. asset_contents is a string.\n\n    This method will be called by the tf.compat.v1.summary.FileWriter when it\n    is time to write the assets out to disk.\n    '
        raise NotImplementedError()