"""Domain object for changes made to domain objects of storage models."""
from __future__ import annotations
import copy
from core import feconf
from core import utils
from typing import Any, Dict, List, Mapping, Union, cast
MYPY = False
if MYPY:
    from core.domain import param_domain
    from core.domain import platform_parameter_domain
    from core.domain import question_domain
    from core.domain import skill_domain
    from core.domain import state_domain
    from core.domain import translation_domain
    AcceptableChangeDictTypes = Union[str, bool, float, int, None, List[str], List[int], Dict[str, Any], List[Dict[str, Any]], List[param_domain.ParamChangeDict], List[state_domain.AnswerGroupDict], List[state_domain.HintDict], List[skill_domain.WorkedExampleDict], translation_domain.WrittenTranslationsDict, List[platform_parameter_domain.PlatformParameterRuleDict], question_domain.QuestionDict, state_domain.AnswerGroupDict, state_domain.SubtitledHtmlDict, state_domain.SolutionDict, state_domain.StateDict, state_domain.OutcomeDict, state_domain.RecordedVoiceoversDict, feconf.TranslatedContentDict, question_domain.QuestionSuggestionChangeDict]

def validate_cmd(cmd_name: str, valid_cmd_attribute_specs: feconf.ValidCmdDict, actual_cmd_attributes: Mapping[str, AcceptableChangeDictTypes]) -> None:
    if False:
        print('Hello World!')
    'Validates that the attributes of a command contain all the required\n    attributes and some/all of optional attributes. It also checks that\n    the values of attributes belong to a set of allowed values if any.\n\n    Args:\n        cmd_name: str. The command for which validation process is being done.\n        valid_cmd_attribute_specs: dict. A dict containing the required and\n            optional attributes for a command along with allowed values\n            for attributes if any.\n        actual_cmd_attributes: dict. A dict containing the actual\n            attributes of a command with values for the attributes.\n\n    Raises:\n        ValidationError. Any required attribute is missing or an extra attribute\n            exists or the value of an attribute is not allowed.\n        DeprecatedCommandError. The value of any attribute is deprecated.\n    '
    required_attribute_names = valid_cmd_attribute_specs['required_attribute_names']
    optional_attribute_names = valid_cmd_attribute_specs['optional_attribute_names']
    actual_attribute_names = [key for key in actual_cmd_attributes.keys() if key != 'cmd']
    missing_attribute_names = [key for key in required_attribute_names if key not in actual_attribute_names]
    extra_attribute_names = [key for key in actual_attribute_names if key not in required_attribute_names + optional_attribute_names]
    error_msg_list = []
    if missing_attribute_names:
        error_msg_list.append('The following required attributes are missing: %s' % ', '.join(sorted(missing_attribute_names)))
    if extra_attribute_names:
        error_msg_list.append('The following extra attributes are present: %s' % ', '.join(sorted(extra_attribute_names)))
    if error_msg_list:
        raise utils.ValidationError(', '.join(error_msg_list))
    deprecated_values = valid_cmd_attribute_specs.get('deprecated_values', {})
    for (attribute_name, attribute_values) in deprecated_values.items():
        actual_value = actual_cmd_attributes.get(attribute_name)
        if actual_value in attribute_values:
            raise utils.DeprecatedCommandError('Value for %s in cmd %s: %s is deprecated' % (attribute_name, cmd_name, actual_value))
    allowed_values = valid_cmd_attribute_specs.get('allowed_values')
    if not allowed_values:
        return
    for (attribute_name, attribute_values) in allowed_values.items():
        actual_value = actual_cmd_attributes[attribute_name]
        if actual_value not in attribute_values:
            raise utils.ValidationError('Value for %s in cmd %s: %s is not allowed' % (attribute_name, cmd_name, actual_value))

class BaseChange:
    """Domain object for changes made to storage models' domain objects."""
    ALLOWED_COMMANDS: List[feconf.ValidCmdDict] = []
    DEPRECATED_COMMANDS: List[str] = []
    COMMON_ALLOWED_COMMANDS: List[feconf.ValidCmdDict] = [{'name': feconf.CMD_DELETE_COMMIT, 'required_attribute_names': [], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}]

    def __init__(self, change_dict: Mapping[str, AcceptableChangeDictTypes]) -> None:
        if False:
            i = 10
            return i + 15
        'Initializes a BaseChange object from a dict.\n\n        Args:\n            change_dict: dict. The dict containing cmd name and attributes.\n\n        Raises:\n            ValidationError. The given change_dict is not valid.\n        '
        self.validate_dict(change_dict)
        cmd_name = change_dict['cmd']
        self.cmd = cmd_name
        all_allowed_commands = self.ALLOWED_COMMANDS + self.COMMON_ALLOWED_COMMANDS
        cmd_attribute_names = []
        for cmd in all_allowed_commands:
            if cmd['name'] == cmd_name:
                cmd_attribute_names = cmd['required_attribute_names'] + cmd['optional_attribute_names']
                break
        for attribute_name in cmd_attribute_names:
            setattr(self, attribute_name, change_dict.get(attribute_name))

    def validate_dict(self, change_dict: Mapping[str, AcceptableChangeDictTypes]) -> None:
        if False:
            while True:
                i = 10
        'Checks that the command in change dict is valid for the domain\n        object.\n\n        Args:\n            change_dict: dict. A dict of changes with keys as a cmd and the\n                attributes of a command.\n\n        Raises:\n            ValidationError. The change dict does not contain the cmd key,\n                or the cmd name is not allowed for the Change domain object\n                or the command attributes are missing or extra.\n            DeprecatedCommandError. The change dict contains a deprecated\n                command or the value for the command attribute is deprecated.\n        '
        if 'cmd' not in change_dict:
            raise utils.ValidationError('Missing cmd key in change dict')
        cmd_name = change_dict['cmd']
        assert isinstance(cmd_name, str)
        valid_cmd_attribute_specs = None
        all_allowed_commands = self.ALLOWED_COMMANDS + self.COMMON_ALLOWED_COMMANDS
        for cmd in all_allowed_commands:
            if cmd['name'] == cmd_name:
                valid_cmd_attribute_specs = copy.deepcopy(cmd)
                break
        if cmd_name in self.DEPRECATED_COMMANDS:
            raise utils.DeprecatedCommandError('Command %s is deprecated' % cmd_name)
        if not valid_cmd_attribute_specs:
            raise utils.ValidationError('Command %s is not allowed' % cmd_name)
        actual_cmd_attributes = copy.deepcopy(change_dict)
        validate_cmd(cmd_name, valid_cmd_attribute_specs, actual_cmd_attributes)

    def to_dict(self) -> Dict[str, AcceptableChangeDictTypes]:
        if False:
            print('Hello World!')
        'Returns a dict representing the BaseChange domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of BaseChange instance.\n        '
        base_change_dict = {}
        base_change_dict['cmd'] = self.cmd
        all_allowed_commands = self.ALLOWED_COMMANDS + self.COMMON_ALLOWED_COMMANDS
        valid_cmd_attribute_names = []
        for cmd in all_allowed_commands:
            if cmd['name'] == self.cmd:
                valid_cmd_attribute_names = cmd['required_attribute_names'] + cmd['optional_attribute_names']
                break
        for attribute_name in valid_cmd_attribute_names:
            if hasattr(self, attribute_name):
                base_change_dict[attribute_name] = getattr(self, attribute_name)
        return base_change_dict

    @classmethod
    def from_dict(cls, base_change_dict: Mapping[str, AcceptableChangeDictTypes]) -> BaseChange:
        if False:
            print('Hello World!')
        'Returns a BaseChange domain object from a dict.\n\n        Args:\n            base_change_dict: dict. The dict representation of\n                BaseChange object.\n\n        Returns:\n            BaseChange. The corresponding BaseChange domain object.\n        '
        return cls(base_change_dict)

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates various properties of the BaseChange object.\n\n        Raises:\n            ValidationError. One or more attributes of the BaseChange are\n                invalid.\n        '
        self.validate_dict(self.to_dict())

    def __getattr__(self, name: str) -> str:
        if False:
            while True:
                i = 10
        try:
            return cast(str, self.__dict__[name])
        except KeyError as e:
            raise AttributeError(name) from e