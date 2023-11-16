from typing import Optional, Any, Dict, TYPE_CHECKING, List
from UM.Signal import Signal, signalemitter
from UM.i18n import i18nCatalog
from UM.Application import Application
from UM.Settings.ContainerFormatError import ContainerFormatError
from UM.Settings.ContainerStack import ContainerStack
from UM.Settings.InstanceContainer import InstanceContainer
from UM.Settings.DefinitionContainer import DefinitionContainer
from UM.Settings.ContainerRegistry import ContainerRegistry
import re
import json
import collections
i18n_catalog = i18nCatalog('cura')
if TYPE_CHECKING:
    from UM.Settings.Interfaces import DefinitionContainerInterface

@signalemitter
class Script:
    """Base class for scripts. All scripts should inherit the script class."""

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._stack = None
        self._definition = None
        self._instance = None

    def initialize(self) -> None:
        if False:
            i = 10
            return i + 15
        setting_data = self.getSettingData()
        self._stack = ContainerStack(stack_id=str(id(self)))
        self._stack.setDirty(False)
        if 'key' in setting_data:
            definitions = ContainerRegistry.getInstance().findDefinitionContainers(id=setting_data['key'])
            if definitions:
                self._definition = definitions[0]
            else:
                self._definition = DefinitionContainer(setting_data['key'])
                try:
                    self._definition.deserialize(json.dumps(setting_data))
                    ContainerRegistry.getInstance().addContainer(self._definition)
                except ContainerFormatError:
                    self._definition = None
                    return
        if self._definition is None:
            return
        self._stack.addContainer(self._definition)
        self._instance = InstanceContainer(container_id='ScriptInstanceContainer')
        self._instance.setDefinition(self._definition.getId())
        self._instance.setMetaDataEntry('setting_version', self._definition.getMetaDataEntry('setting_version', default=0))
        self._stack.addContainer(self._instance)
        self._stack.propertyChanged.connect(self._onPropertyChanged)
        ContainerRegistry.getInstance().addContainer(self._stack)
    settingsLoaded = Signal()
    valueChanged = Signal()

    def _onPropertyChanged(self, key: str, property_name: str) -> None:
        if False:
            i = 10
            return i + 15
        if property_name == 'value':
            self.valueChanged.emit()
            global_container_stack = Application.getInstance().getGlobalContainerStack()
            if global_container_stack is not None:
                global_container_stack.propertyChanged.emit(key, property_name)

    def getSettingData(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Needs to return a dict that can be used to construct a settingcategory file.\n\n        See the example script for an example.\n        It follows the same style / guides as the Uranium settings.\n        Scripts can either override getSettingData directly, or use getSettingDataString\n        to return a string that will be parsed as json. The latter has the benefit over\n        returning a dict in that the order of settings is maintained.\n        '
        setting_data_as_string = self.getSettingDataString()
        setting_data = json.loads(setting_data_as_string, object_pairs_hook=collections.OrderedDict)
        return setting_data

    def getSettingDataString(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def getDefinitionId(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        if self._stack:
            bottom = self._stack.getBottom()
            if bottom is not None:
                return bottom.getId()
        return None

    def getStackId(self) -> Optional[str]:
        if False:
            print('Hello World!')
        if self._stack:
            return self._stack.getId()
        return None

    def getSettingValueByKey(self, key: str) -> Any:
        if False:
            return 10
        'Convenience function that retrieves value of a setting from the stack.'
        if self._stack is not None:
            return self._stack.getProperty(key, 'value')
        return None

    def getValue(self, line: str, key: str, default=None) -> Any:
        if False:
            return 10
        'Convenience function that finds the value in a line of g-code.\n\n        When requesting key = x from line "G1 X100" the value 100 is returned.\n        '
        if not key in line or (';' in line and line.find(key) > line.find(';')):
            return default
        sub_part = line[line.find(key) + 1:]
        m = re.search('^-?[0-9]+\\.?[0-9]*', sub_part)
        if m is None:
            return default
        try:
            return int(m.group(0))
        except ValueError:
            try:
                return float(m.group(0))
            except ValueError:
                return default

    def putValue(self, line: str='', **kwargs) -> str:
        if False:
            i = 10
            return i + 15
        "Convenience function to produce a line of g-code.\n\n        You can put in an original g-code line and it'll re-use all the values\n        in that line.\n        All other keyword parameters are put in the result in g-code's format.\n        For instance, if you put ``G=1`` in the parameters, it will output\n        ``G1``. If you put ``G=1, X=100`` in the parameters, it will output\n        ``G1 X100``. The parameters will be added in order G M T S F X Y Z E.\n        Any other parameters will be added in arbitrary order.\n\n        :param line: The original g-code line that must be modified. If not\n            provided, an entirely new g-code line will be produced.\n        :return: A line of g-code with the desired parameters filled in.\n        "
        if ';' in line:
            comment = line[line.find(';'):]
            line = line[:line.find(';')]
        else:
            comment = ''
        for part in line.split(' '):
            if part == '':
                continue
            parameter = part[0]
            if parameter not in kwargs:
                value = part[1:]
                kwargs[parameter] = value
        line_parts = list()
        for parameter in ['G', 'M', 'T', 'S', 'F', 'X', 'Y', 'Z', 'E']:
            if parameter in kwargs:
                value = kwargs.pop(parameter)
                line_parts.append(parameter + str(value))
        for (parameter, value) in kwargs.items():
            line_parts.append(parameter + str(value))
        if comment != '':
            line_parts.append(comment)
        return ' '.join(line_parts)

    def execute(self, data: List[str]) -> List[str]:
        if False:
            i = 10
            return i + 15
        'This is called when the script is executed. \n\n        It gets a list of g-code strings and needs to return a (modified) list.\n        '
        raise NotImplementedError()