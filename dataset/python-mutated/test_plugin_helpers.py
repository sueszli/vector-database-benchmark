import typing as t
from tmuxp.plugin import TmuxpPlugin
if t.TYPE_CHECKING:
    from typing_extensions import Unpack
    from tmuxp._types import PluginConfigSchema
    from ._types import PluginTestConfigSchema

class MyTestTmuxpPlugin(TmuxpPlugin):

    def __init__(self, **config: 'Unpack[PluginTestConfigSchema]') -> None:
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(config, dict)
        tmux_version = config.pop('tmux_version', None)
        libtmux_version = config.pop('libtmux_version', None)
        tmuxp_version = config.pop('tmuxp_version', None)
        t.cast('PluginConfigSchema', config)
        assert 'tmux_version' not in config
        super().__init__(**config)
        if tmux_version:
            self.version_constraints['tmux']['version'] = tmux_version
        if libtmux_version:
            self.version_constraints['libtmux']['version'] = libtmux_version
        if tmuxp_version:
            self.version_constraints['tmuxp']['version'] = tmuxp_version
        self._version_check()