import copy
import logging
import re
import warnings
from abc import ABCMeta
from typing import Iterator, List, Optional, Set, Type, Union, Sequence, Dict, Any
from slack_sdk.models import show_unknown_key_warning
from slack_sdk.models.basic_objects import JsonObject, JsonValidator, EnumValidator
from .basic_components import ButtonStyles, Workflow
from .basic_components import ConfirmObject
from .basic_components import DispatchActionConfig
from .basic_components import MarkdownTextObject
from .basic_components import Option
from .basic_components import OptionGroup
from .basic_components import PlainTextObject
from .basic_components import TextObject

class BlockElement(JsonObject, metaclass=ABCMeta):
    """Block Elements are things that exists inside of your Blocks.
    https://api.slack.com/reference/block-kit/block-elements
    """
    attributes = {'type'}
    logger = logging.getLogger(__name__)

    def _subtype_warning(self):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('subtype is deprecated since slackclient 2.6.0, use type instead', DeprecationWarning)

    @property
    def subtype(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self.type

    def __init__(self, *, type: Optional[str]=None, subtype: Optional[str]=None, **others: dict):
        if False:
            print('Hello World!')
        if subtype:
            self._subtype_warning()
        self.type = type if type else subtype
        show_unknown_key_warning(self, others)

    @classmethod
    def parse(cls, block_element: Union[dict, 'BlockElement']) -> Optional[Union['BlockElement', TextObject]]:
        if False:
            while True:
                i = 10
        if block_element is None:
            return None
        elif isinstance(block_element, dict):
            if 'type' in block_element:
                d = copy.copy(block_element)
                t = d.pop('type')
                for subclass in cls._get_sub_block_elements():
                    if t == subclass.type:
                        return subclass(**d)
                if t == PlainTextObject.type:
                    return PlainTextObject(**d)
                elif t == MarkdownTextObject.type:
                    return MarkdownTextObject(**d)
        elif isinstance(block_element, (TextObject, BlockElement)):
            return block_element
        cls.logger.warning(f'Unknown element detected and skipped ({block_element})')
        return None

    @classmethod
    def parse_all(cls, block_elements: Sequence[Union[dict, 'BlockElement', TextObject]]) -> List[Union['BlockElement', TextObject]]:
        if False:
            return 10
        return [cls.parse(e) for e in block_elements or []]

    @classmethod
    def _get_sub_block_elements(cls: Type['BlockElement']) -> Iterator[Type['BlockElement']]:
        if False:
            while True:
                i = 10
        for subclass in cls.__subclasses__():
            if hasattr(subclass, 'type'):
                yield subclass
            yield from subclass._get_sub_block_elements()

class InteractiveElement(BlockElement):
    action_id_max_length = 255

    @property
    def attributes(self) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        return super().attributes.union({'alt_text', 'action_id'})

    def __init__(self, *, action_id: Optional[str]=None, type: Optional[str]=None, subtype: Optional[str]=None, **others: dict):
        if False:
            i = 10
            return i + 15
        'An interactive block element.\n\n        We generally recommend using the concrete subclasses for better supports of available properties.\n        '
        if subtype:
            self._subtype_warning()
        super().__init__(type=type or subtype)
        self.action_id = action_id

    @JsonValidator(f'action_id attribute cannot exceed {action_id_max_length} characters')
    def _validate_action_id_length(self) -> bool:
        if False:
            print('Hello World!')
        return self.action_id is None or len(self.action_id) <= self.action_id_max_length

class InputInteractiveElement(InteractiveElement, metaclass=ABCMeta):
    placeholder_max_length = 150
    attributes = {'type', 'action_id', 'placeholder', 'confirm', 'focus_on_load'}

    @property
    def subtype(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self.type

    def __init__(self, *, action_id: Optional[str]=None, placeholder: Optional[Union[str, TextObject]]=None, type: Optional[str]=None, subtype: Optional[str]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            i = 10
            return i + 15
        'InteractiveElement that is usable in input blocks\n\n        We generally recommend using the concrete subclasses for better supports of available properties.\n        '
        if subtype:
            self._subtype_warning()
        super().__init__(action_id=action_id, type=type or subtype)
        self.placeholder = TextObject.parse(placeholder)
        self.confirm = ConfirmObject.parse(confirm)
        self.focus_on_load = focus_on_load

    @JsonValidator(f'placeholder attribute cannot exceed {placeholder_max_length} characters')
    def _validate_placeholder_length(self) -> bool:
        if False:
            print('Hello World!')
        return self.placeholder is None or self.placeholder.text is None or len(self.placeholder.text) <= self.placeholder_max_length

class ButtonElement(InteractiveElement):
    type = 'button'
    text_max_length = 75
    url_max_length = 3000
    value_max_length = 2000

    @property
    def attributes(self) -> Set[str]:
        if False:
            print('Hello World!')
        return super().attributes.union({'text', 'url', 'value', 'style', 'confirm', 'accessibility_label'})

    def __init__(self, *, text: Union[str, dict, TextObject], action_id: Optional[str]=None, url: Optional[str]=None, value: Optional[str]=None, style: Optional[str]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, accessibility_label: Optional[str]=None, **others: dict):
        if False:
            while True:
                i = 10
        'An interactive element that inserts a button. The button can be a trigger for\n        anything from opening a simple link to starting a complex workflow.\n        https://api.slack.com/reference/block-kit/block-elements#button\n\n        Args:\n            text (required): A text object that defines the button\'s text.\n                Can only be of type: plain_text.\n                Maximum length for the text in this field is 75 characters.\n            action_id (required): An identifier for this action.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            url: A URL to load in the user\'s browser when the button is clicked.\n                Maximum length for this field is 3000 characters.\n                If you\'re using url, you\'ll still receive an interaction payload\n                and will need to send an acknowledgement response.\n            value: The value to send along with the interaction payload.\n                Maximum length for this field is 2000 characters.\n            style: Decorates buttons with alternative visual color schemes. Use this option with restraint.\n                "primary" gives buttons a green outline and text, ideal for affirmation or confirmation actions.\n                "primary" should only be used for one button within a set.\n                "danger" gives buttons a red outline and text, and should be used when the action is destructive.\n                Use "danger" even more sparingly than "primary".\n                If you don\'t include this field, the default button style will be used.\n            confirm: A confirm object that defines an optional confirmation dialog after the button is clicked.\n            accessibility_label: A label for longer descriptive text about a button element.\n                This label will be read out by screen readers instead of the button text object.\n                Maximum length for this field is 75 characters.\n        '
        super().__init__(action_id=action_id, type=self.type)
        show_unknown_key_warning(self, others)
        self.text = TextObject.parse(text, default_type=PlainTextObject.type)
        self.url = url
        self.value = value
        self.style = style
        self.confirm = ConfirmObject.parse(confirm)
        self.accessibility_label = accessibility_label

    @JsonValidator(f'text attribute cannot exceed {text_max_length} characters')
    def _validate_text_length(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.text is None or self.text.text is None or len(self.text.text) <= self.text_max_length

    @JsonValidator(f'url attribute cannot exceed {url_max_length} characters')
    def _validate_url_length(self) -> bool:
        if False:
            while True:
                i = 10
        return self.url is None or len(self.url) <= self.url_max_length

    @JsonValidator(f'value attribute cannot exceed {value_max_length} characters')
    def _validate_value_length(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.value is None or len(self.value) <= self.value_max_length

    @EnumValidator('style', ButtonStyles)
    def _validate_style_valid(self):
        if False:
            while True:
                i = 10
        return self.style is None or self.style in ButtonStyles

    @JsonValidator(f'accessibility_label attribute cannot exceed {text_max_length} characters')
    def _validate_accessibility_label_length(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.accessibility_label is None or len(self.accessibility_label) <= self.text_max_length

class LinkButtonElement(ButtonElement):

    def __init__(self, *, text: Union[str, dict, PlainTextObject], url: str, action_id: Optional[str]=None, style: Optional[str]=None, **others: dict):
        if False:
            i = 10
            return i + 15
        'A simple button that simply opens a given URL. You will still receive an\n        interaction payload and will need to send an acknowledgement response.\n        This is a helper class that makes creating links simpler.\n        https://api.slack.com/reference/block-kit/block-elements#button\n\n        Args:\n            text (required): A text object that defines the button\'s text.\n                Can only be of type: plain_text.\n                Maximum length for the text in this field is 75 characters.\n            url (required): A URL to load in the user\'s browser when the button is clicked.\n                Maximum length for this field is 3000 characters.\n                If you\'re using url, you\'ll still receive an interaction payload\n                and will need to send an acknowledgement response.\n            action_id (required): An identifier for this action.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            style: Decorates buttons with alternative visual color schemes. Use this option with restraint.\n                "primary" gives buttons a green outline and text, ideal for affirmation or confirmation actions.\n                "primary" should only be used for one button within a set.\n                "danger" gives buttons a red outline and text, and should be used when the action is destructive.\n                Use "danger" even more sparingly than "primary".\n                If you don\'t include this field, the default button style will be used.\n        '
        super().__init__(text=text, url=url, action_id=action_id, value=None, style=style)
        show_unknown_key_warning(self, others)

class CheckboxesElement(InputInteractiveElement):
    type = 'checkboxes'

    @property
    def attributes(self) -> Set[str]:
        if False:
            while True:
                i = 10
        return super().attributes.union({'options', 'initial_options'})

    def __init__(self, *, action_id: Optional[str]=None, options: Optional[Sequence[Union[dict, Option]]]=None, initial_options: Optional[Sequence[Union[dict, Option]]]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            print('Hello World!')
        'A checkbox group that allows a user to choose multiple items from a list of possible options.\n        https://api.slack.com/reference/block-kit/block-elements#checkboxes\n\n        Args:\n            action_id (required): An identifier for the action triggered when the checkbox group is changed.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            options (required): An array of option objects. A maximum of 10 options are allowed.\n            initial_options: An array of option objects that exactly matches one or more of the options.\n                These options will be selected when the checkbox group initially loads.\n            confirm: A confirm object that defines an optional confirmation dialog that appears\n                after clicking one of the checkboxes in this element.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.options = Option.parse_all(options)
        self.initial_options = Option.parse_all(initial_options)

class DatePickerElement(InputInteractiveElement):
    type = 'datepicker'

    @property
    def attributes(self) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        return super().attributes.union({'initial_date'})

    def __init__(self, *, action_id: Optional[str]=None, placeholder: Optional[Union[str, dict, TextObject]]=None, initial_date: Optional[str]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            while True:
                i = 10
        '\n        An element which lets users easily select a date from a calendar style UI.\n        Date picker elements can be used inside of SectionBlocks and ActionsBlocks.\n        https://api.slack.com/reference/block-kit/block-elements#datepicker\n\n        Args:\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            placeholder: A plain_text only text object that defines the placeholder text shown on the datepicker.\n                Maximum length for the text in this field is 150 characters.\n            initial_date: The initial date that is selected when the element is loaded.\n                This should be in the format YYYY-MM-DD.\n            confirm: A confirm object that defines an optional confirmation dialog\n                that appears after a date is selected.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_date = initial_date

    @JsonValidator("initial_date attribute must be in format 'YYYY-MM-DD'")
    def _validate_initial_date_valid(self) -> bool:
        if False:
            print('Hello World!')
        return self.initial_date is None or re.match('\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])', self.initial_date) is not None

class TimePickerElement(InputInteractiveElement):
    type = 'timepicker'

    @property
    def attributes(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return super().attributes.union({'initial_time', 'timezone'})

    def __init__(self, *, action_id: Optional[str]=None, placeholder: Optional[Union[str, dict, TextObject]]=None, initial_time: Optional[str]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, focus_on_load: Optional[bool]=None, timezone: Optional[str]=None, **others: dict):
        if False:
            return 10
        '\n        An element which allows selection of a time of day.\n        On desktop clients, this time picker will take the form of a dropdown list\n        with free-text entry for precise choices.\n        On mobile clients, the time picker will use native time picker UIs.\n        https://api.slack.com/reference/block-kit/block-elements#timepicker\n\n        Args:\n            action_id (required): An identifier for the action triggered when a time is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            placeholder: A plain_text only text object that defines the placeholder text shown on the timepicker.\n                Maximum length for the text in this field is 150 characters.\n            initial_time: The initial time that is selected when the element is loaded.\n                This should be in the format HH:mm, where HH is the 24-hour format of an hour (00 to 23)\n                and mm is minutes with leading zeros (00 to 59), for example 22:25 for 10:25pm.\n            confirm: A confirm object that defines an optional confirmation dialog\n                that appears after a time is selected.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n            timezone: The timezone to consider for this input value.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_time = initial_time
        self.timezone = timezone

    @JsonValidator("initial_time attribute must be in format 'HH:mm'")
    def _validate_initial_time_valid(self) -> bool:
        if False:
            return 10
        return self.initial_time is None or re.match('([0-1][0-9]|2[0-3]):([0-5][0-9])', self.initial_time) is not None

class DateTimePickerElement(InputInteractiveElement):
    type = 'datetimepicker'

    @property
    def attributes(self) -> Set[str]:
        if False:
            print('Hello World!')
        return super().attributes.union({'initial_date_time'})

    def __init__(self, *, action_id: Optional[str]=None, initial_date_time: Optional[int]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            i = 10
            return i + 15
        '\n        An element that allows the selection of a time of day formatted as a UNIX timestamp.\n        On desktop clients, this time picker will take the form of a dropdown list and the\n        date picker will take the form of a dropdown calendar. Both options will have free-text\n        entry for precise choices. On mobile clients, the time picker and date\n        picker will use native UIs.\n        https://api.slack.com/reference/block-kit/block-elements#datetimepicker\n\n        Args:\n            action_id (required): An identifier for the action triggered when a time is selected. You can use this\n                when you receive an interaction payload to identify the source of the action. Should be unique among\n                all other action_ids in the containing block. Maximum length for this field is 255 characters.\n            initial_date_time: The initial date and time that is selected when the element is loaded, represented as\n                a UNIX timestamp in seconds. This should be in the format of 10 digits, for example 1628633820\n                represents the date and time August 10th, 2021 at 03:17pm PST.\n                and mm is minutes with leading zeros (00 to 59), for example 22:25 for 10:25pm.\n            confirm: A confirm object that defines an optional confirmation dialog\n                that appears after a time is selected.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_date_time = initial_date_time

    @JsonValidator('initial_date_time attribute must be between 0 and 99999999 seconds')
    def _validate_initial_date_time_valid(self) -> bool:
        if False:
            while True:
                i = 10
        return self.initial_date_time is None or 0 <= self.initial_date_time <= 9999999999

class ImageElement(BlockElement):
    type = 'image'
    image_url_max_length = 3000
    alt_text_max_length = 2000

    @property
    def attributes(self) -> Set[str]:
        if False:
            while True:
                i = 10
        return super().attributes.union({'alt_text', 'image_url'})

    def __init__(self, *, image_url: Optional[str]=None, alt_text: Optional[str]=None, **others: dict):
        if False:
            for i in range(10):
                print('nop')
        "An element to insert an image - this element can be used in section and\n        context blocks only. If you want a block with only an image in it,\n        you're looking for the image block.\n        https://api.slack.com/reference/block-kit/block-elements#image\n\n        Args:\n            image_url (required): The URL of the image to be displayed.\n            alt_text (required): A plain-text summary of the image. This should not contain any markup.\n        "
        super().__init__(type=self.type)
        show_unknown_key_warning(self, others)
        self.image_url = image_url
        self.alt_text = alt_text

    @JsonValidator(f'image_url attribute cannot exceed {image_url_max_length} characters')
    def _validate_image_url_length(self) -> bool:
        if False:
            print('Hello World!')
        return len(self.image_url) <= self.image_url_max_length

    @JsonValidator(f'alt_text attribute cannot exceed {alt_text_max_length} characters')
    def _validate_alt_text_length(self) -> bool:
        if False:
            return 10
        return len(self.alt_text) <= self.alt_text_max_length

class StaticSelectElement(InputInteractiveElement):
    type = 'static_select'
    options_max_length = 100
    option_groups_max_length = 100

    @property
    def attributes(self) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        return super().attributes.union({'options', 'option_groups', 'initial_option'})

    def __init__(self, *, placeholder: Optional[Union[str, dict, TextObject]]=None, action_id: Optional[str]=None, options: Optional[Sequence[Union[dict, Option]]]=None, option_groups: Optional[Sequence[Union[dict, OptionGroup]]]=None, initial_option: Optional[Union[dict, Option]]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            for i in range(10):
                print('nop')
        'This is the simplest form of select menu, with a static list of options passed in when defining the element.\n        https://api.slack.com/reference/block-kit/block-elements#static_select\n\n        Args:\n            placeholder (required): A plain_text only text object that defines the placeholder text shown on the menu.\n                Maximum length for the text in this field is 150 characters.\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            options (either options or option_groups is required): An array of option objects.\n                Maximum number of options is 100.\n                If option_groups is specified, this field should not be.\n            option_groups (either options or option_groups is required): An array of option group objects.\n                Maximum number of option groups is 100.\n                If options is specified, this field should not be.\n            initial_option: A single option that exactly matches one of the options or option_groups.\n                This option will be selected when the menu initially loads.\n            confirm: A confirm object that defines an optional confirmation dialog\n                that appears after a menu item is selected.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.options = options
        self.option_groups = option_groups
        self.initial_option = initial_option

    @JsonValidator(f'options attribute cannot exceed {options_max_length} elements')
    def _validate_options_length(self) -> bool:
        if False:
            return 10
        return self.options is None or len(self.options) <= self.options_max_length

    @JsonValidator(f'option_groups attribute cannot exceed {option_groups_max_length} elements')
    def _validate_option_groups_length(self) -> bool:
        if False:
            return 10
        return self.option_groups is None or len(self.option_groups) <= self.option_groups_max_length

    @JsonValidator('options and option_groups cannot both be specified')
    def _validate_options_and_option_groups_both_specified(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not (self.options is not None and self.option_groups is not None)

    @JsonValidator('options or option_groups must be specified')
    def _validate_neither_options_or_option_groups_is_specified(self) -> bool:
        if False:
            while True:
                i = 10
        return self.options is not None or self.option_groups is not None

class StaticMultiSelectElement(InputInteractiveElement):
    type = 'multi_static_select'
    options_max_length = 100
    option_groups_max_length = 100

    @property
    def attributes(self) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        return super().attributes.union({'options', 'option_groups', 'initial_options', 'max_selected_items'})

    def __init__(self, *, placeholder: Optional[Union[str, dict, TextObject]]=None, action_id: Optional[str]=None, options: Optional[Sequence[Option]]=None, option_groups: Optional[Sequence[OptionGroup]]=None, initial_options: Optional[Sequence[Option]]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, max_selected_items: Optional[int]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            for i in range(10):
                print('nop')
        '\n        This is the simplest form of select menu, with a static list of options passed in when defining the element.\n        https://api.slack.com/reference/block-kit/block-elements#static_multi_select\n\n        Args:\n            placeholder (required): A plain_text only text object that defines the placeholder text shown on the menu.\n                Maximum length for the text in this field is 150 characters.\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            options (either options or option_groups is required): An array of option objects.\n                Maximum number of options is 100.\n                If option_groups is specified, this field should not be.\n            option_groups (either options or option_groups is required): An array of option group objects.\n                Maximum number of option groups is 100.\n                If options is specified, this field should not be.\n            initial_options: An array of option objects that exactly match one or more of the options\n                within options or option_groups. These options will be selected when the menu initially loads.\n            confirm: A confirm object that defines an optional confirmation dialog\n                that appears before the multi-select choices are submitted.\n            max_selected_items: Specifies the maximum number of items that can be selected in the menu.\n                Minimum number is 1.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.options = Option.parse_all(options)
        self.option_groups = OptionGroup.parse_all(option_groups)
        self.initial_options = Option.parse_all(initial_options)
        self.max_selected_items = max_selected_items

    @JsonValidator(f'options attribute cannot exceed {options_max_length} elements')
    def _validate_options_length(self) -> bool:
        if False:
            return 10
        return self.options is None or len(self.options) <= self.options_max_length

    @JsonValidator(f'option_groups attribute cannot exceed {option_groups_max_length} elements')
    def _validate_option_groups_length(self) -> bool:
        if False:
            print('Hello World!')
        return self.option_groups is None or len(self.option_groups) <= self.option_groups_max_length

    @JsonValidator('options and option_groups cannot both be specified')
    def _validate_options_and_option_groups_both_specified(self) -> bool:
        if False:
            return 10
        return self.options is None or self.option_groups is None

    @JsonValidator('options or option_groups must be specified')
    def _validate_neither_options_or_option_groups_is_specified(self) -> bool:
        if False:
            while True:
                i = 10
        return self.options is not None or self.option_groups is not None

class SelectElement(InputInteractiveElement):
    type = 'static_select'
    options_max_length = 100
    option_groups_max_length = 100

    @property
    def attributes(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return super().attributes.union({'options', 'option_groups', 'initial_option'})

    def __init__(self, *, action_id: Optional[str]=None, placeholder: Optional[str]=None, options: Optional[Sequence[Option]]=None, option_groups: Optional[Sequence[OptionGroup]]=None, initial_option: Optional[Option]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            while True:
                i = 10
        'This is the simplest form of select menu, with a static list of options passed in when defining the element.\n        https://api.slack.com/reference/block-kit/block-elements#static_select\n\n        Args:\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            placeholder (required): A plain_text only text object that defines the placeholder text shown on the menu.\n                Maximum length for the text in this field is 150 characters.\n            options (either options or option_groups is required): An array of option objects.\n                Maximum number of options is 100.\n                If option_groups is specified, this field should not be.\n            option_groups (either options or option_groups is required): An array of option group objects.\n                Maximum number of option groups is 100.\n                If options is specified, this field should not be.\n            initial_option: A single option that exactly matches one of the options or option_groups.\n                This option will be selected when the menu initially loads.\n            confirm: A confirm object that defines an optional confirmation dialog\n                that appears after a menu item is selected.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.options = options
        self.option_groups = option_groups
        self.initial_option = initial_option

    @JsonValidator(f'options attribute cannot exceed {options_max_length} elements')
    def _validate_options_length(self) -> bool:
        if False:
            return 10
        return self.options is None or len(self.options) <= self.options_max_length

    @JsonValidator(f'option_groups attribute cannot exceed {option_groups_max_length} elements')
    def _validate_option_groups_length(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.option_groups is None or len(self.option_groups) <= self.option_groups_max_length

    @JsonValidator('options and option_groups cannot both be specified')
    def _validate_options_and_option_groups_both_specified(self) -> bool:
        if False:
            return 10
        return not (self.options is not None and self.option_groups is not None)

    @JsonValidator('options or option_groups must be specified')
    def _validate_neither_options_or_option_groups_is_specified(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.options is not None or self.option_groups is not None

class ExternalDataSelectElement(InputInteractiveElement):
    type = 'external_select'

    @property
    def attributes(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return super().attributes.union({'min_query_length', 'initial_option'})

    def __init__(self, *, action_id: Optional[str]=None, placeholder: Optional[Union[str, TextObject]]=None, initial_option: Union[Optional[Option], Optional[OptionGroup]]=None, min_query_length: Optional[int]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            print('Hello World!')
        '\n        This select menu will load its options from an external data source, allowing\n        for a dynamic list of options.\n        https://api.slack.com/reference/block-kit/block-elements#external_select\n\n        Args:\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            placeholder (required): A plain_text only text object that defines the placeholder text shown on the menu.\n                Maximum length for the text in this field is 150 characters.\n            initial_option: A single option that exactly matches one of the options\n                within the options or option_groups loaded from the external data source.\n                This option will be selected when the menu initially loads.\n            min_query_length: When the typeahead field is used, a request will be sent on every character change.\n                If you prefer fewer requests or more fully ideated queries,\n                use the min_query_length attribute to tell Slack\n                the fewest number of typed characters required before dispatch.\n                The default value is 3.\n            confirm: A confirm object that defines an optional confirmation dialog\n                that appears after a menu item is selected.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.min_query_length = min_query_length
        self.initial_option = initial_option

class ExternalDataMultiSelectElement(InputInteractiveElement):
    type = 'multi_external_select'

    @property
    def attributes(self) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        return super().attributes.union({'min_query_length', 'initial_options', 'max_selected_items'})

    def __init__(self, *, placeholder: Optional[Union[str, dict, TextObject]]=None, action_id: Optional[str]=None, min_query_length: Optional[int]=None, initial_options: Optional[Sequence[Union[dict, Option]]]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, max_selected_items: Optional[int]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            i = 10
            return i + 15
        '\n        This select menu will load its options from an external data source, allowing\n        for a dynamic list of options.\n        https://api.slack.com/reference/block-kit/block-elements#external_multi_select\n\n        Args:\n            placeholder (required): A plain_text only text object that defines the placeholder text shown on the menu.\n                Maximum length for the text in this field is 150 characters.\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            min_query_length: When the typeahead field is used, a request will be sent on every character change.\n                If you prefer fewer requests or more fully ideated queries,\n                use the min_query_length attribute to tell Slack\n                the fewest number of typed characters required before dispatch.\n                The default value is 3\n            initial_options: An array of option objects that exactly match one or more of the options\n                within options or option_groups. These options will be selected when the menu initially loads.\n            confirm: A confirm object that defines an optional confirmation dialog that appears\n                before the multi-select choices are submitted.\n            max_selected_items: Specifies the maximum number of items that can be selected in the menu.\n                Minimum number is 1.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.min_query_length = min_query_length
        self.initial_options = Option.parse_all(initial_options)
        self.max_selected_items = max_selected_items

class UserSelectElement(InputInteractiveElement):
    type = 'users_select'

    @property
    def attributes(self) -> Set[str]:
        if False:
            return 10
        return super().attributes.union({'initial_user'})

    def __init__(self, *, placeholder: Optional[Union[str, dict, TextObject]]=None, action_id: Optional[str]=None, initial_user: Optional[str]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            return 10
        '\n        This select menu will populate its options with a list of Slack users visible to\n        the current user in the active workspace.\n        https://api.slack.com/reference/block-kit/block-elements#users_select\n\n        Args:\n            placeholder (required): A plain_text only text object that defines the placeholder text shown on the menu.\n                Maximum length for the text in this field is 150 characters.\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            initial_user: The user ID of any valid user to be pre-selected when the menu loads.\n            confirm: A confirm object that defines an optional confirmation dialog\n                that appears after a menu item is selected.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_user = initial_user

class UserMultiSelectElement(InputInteractiveElement):
    type = 'multi_users_select'

    @property
    def attributes(self) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        return super().attributes.union({'initial_users', 'max_selected_items'})

    def __init__(self, *, action_id: Optional[str]=None, placeholder: Optional[Union[str, dict, TextObject]]=None, initial_users: Optional[Sequence[str]]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, max_selected_items: Optional[int]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            i = 10
            return i + 15
        '\n        This select menu will populate its options with a list of Slack users visible to\n        the current user in the active workspace.\n        https://api.slack.com/reference/block-kit/block-elements#users_multi_select\n\n        Args:\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            placeholder (required): A plain_text only text object that defines the placeholder text shown on the menu.\n                Maximum length for the text in this field is 150 characters.\n            initial_users: An array of user IDs of any valid users to be pre-selected when the menu loads.\n            confirm: A confirm object that defines an optional confirmation dialog that appears\n                before the multi-select choices are submitted.\n            max_selected_items: Specifies the maximum number of items that can be selected in the menu.\n                Minimum number is 1.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_users = initial_users
        self.max_selected_items = max_selected_items

class ConversationFilter(JsonObject):
    attributes = {'include', 'exclude_bot_users', 'exclude_external_shared_channels'}
    logger = logging.getLogger(__name__)

    def __init__(self, *, include: Optional[Sequence[str]]=None, exclude_bot_users: Optional[bool]=None, exclude_external_shared_channels: Optional[bool]=None):
        if False:
            i = 10
            return i + 15
        'Provides a way to filter the list of options in a conversations select menu\n        or conversations multi-select menu.\n        https://api.slack.com/reference/block-kit/composition-objects#filter_conversations\n\n        Args:\n            include: Indicates which type of conversations should be included in the list.\n                When this field is provided, any conversations that do not match will be excluded.\n                You should provide an array of strings from the following options:\n                "im", "mpim", "private", and "public". The array cannot be empty.\n            exclude_bot_users: Indicates whether to exclude bot users from conversation lists. Defaults to false.\n            exclude_external_shared_channels: Indicates whether to exclude external shared channels\n                from conversation lists. Defaults to false.\n        '
        self.include = include
        self.exclude_bot_users = exclude_bot_users
        self.exclude_external_shared_channels = exclude_external_shared_channels

    @classmethod
    def parse(cls, filter: Union[dict, 'ConversationFilter']):
        if False:
            return 10
        if filter is None:
            return None
        elif isinstance(filter, ConversationFilter):
            return filter
        elif isinstance(filter, dict):
            d = copy.copy(filter)
            return ConversationFilter(**d)
        else:
            cls.logger.warning(f'Unknown conversation filter object detected and skipped ({filter})')
            return None

class ConversationSelectElement(InputInteractiveElement):
    type = 'conversations_select'

    @property
    def attributes(self) -> Set[str]:
        if False:
            print('Hello World!')
        return super().attributes.union({'initial_conversation', 'response_url_enabled', 'filter', 'default_to_current_conversation'})

    def __init__(self, *, placeholder: Optional[Union[str, dict, TextObject]]=None, action_id: Optional[str]=None, initial_conversation: Optional[str]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, response_url_enabled: Optional[bool]=None, default_to_current_conversation: Optional[bool]=None, filter: Optional[ConversationFilter]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            for i in range(10):
                print('nop')
        "\n        This select menu will populate its options with a list of public and private\n        channels, DMs, and MPIMs visible to the current user in the active workspace.\n        https://api.slack.com/reference/block-kit/block-elements#conversation_select\n\n        Args:\n            placeholder (required): A plain_text only text object that defines the placeholder text shown on the menu.\n                Maximum length for the text in this field is 150 characters.\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            initial_conversation: The ID of any valid conversation to be pre-selected when the menu loads.\n                If default_to_current_conversation is also supplied, initial_conversation will take precedence.\n            confirm: A confirm object that defines an optional confirmation dialog\n                that appears after a menu item is selected.\n            response_url_enabled: This field only works with menus in input blocks in modals.\n                When set to true, the view_submission payload from the menu's parent view will contain a response_url.\n                This response_url can be used for message responses. The target conversation for the message\n                will be determined by the value of this select menu.\n            default_to_current_conversation: Pre-populates the select menu with the conversation\n                that the user was viewing when they opened the modal, if available. Default is false.\n            filter: A filter object that reduces the list of available conversations using the specified criteria.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        "
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_conversation = initial_conversation
        self.response_url_enabled = response_url_enabled
        self.default_to_current_conversation = default_to_current_conversation
        self.filter = filter

class ConversationMultiSelectElement(InputInteractiveElement):
    type = 'multi_conversations_select'

    @property
    def attributes(self) -> Set[str]:
        if False:
            while True:
                i = 10
        return super().attributes.union({'initial_conversations', 'max_selected_items', 'default_to_current_conversation', 'filter'})

    def __init__(self, *, placeholder: Optional[Union[str, dict, TextObject]]=None, action_id: Optional[str]=None, initial_conversations: Optional[Sequence[str]]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, max_selected_items: Optional[int]=None, default_to_current_conversation: Optional[bool]=None, filter: Optional[Union[dict, ConversationFilter]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            return 10
        '\n        This multi-select menu will populate its options with a list of public and private channels,\n        DMs, and MPIMs visible to the current user in the active workspace.\n        https://api.slack.com/reference/block-kit/block-elements#conversation_multi_select\n\n        Args:\n            placeholder (required): A plain_text only text object that defines the placeholder text shown on the menu.\n                Maximum length for the text in this field is 150 characters.\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            initial_conversations: An array of one or more IDs of any valid conversations to be pre-selected\n                when the menu loads. If default_to_current_conversation is also supplied,\n                initial_conversations will be ignored.\n            confirm: A confirm object that defines an optional confirmation dialog that appears\n                before the multi-select choices are submitted.\n            max_selected_items: Specifies the maximum number of items that can be selected in the menu.\n                Minimum number is 1.\n            default_to_current_conversation: Pre-populates the select menu with the conversation that\n                the user was viewing when they opened the modal, if available. Default is false.\n            filter: A filter object that reduces the list of available conversations using the specified criteria.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_conversations = initial_conversations
        self.max_selected_items = max_selected_items
        self.default_to_current_conversation = default_to_current_conversation
        self.filter = ConversationFilter.parse(filter)

class ChannelSelectElement(InputInteractiveElement):
    type = 'channels_select'

    @property
    def attributes(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return super().attributes.union({'initial_channel', 'response_url_enabled'})

    def __init__(self, *, placeholder: Optional[Union[str, dict, TextObject]]=None, action_id: Optional[str]=None, initial_channel: Optional[str]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, response_url_enabled: Optional[bool]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            i = 10
            return i + 15
        "\n        This select menu will populate its options with a list of public channels\n        visible to the current user in the active workspace.\n        https://api.slack.com/reference/block-kit/block-elements#channel_select\n\n        Args:\n            placeholder (required): A plain_text only text object that defines the placeholder text shown on the menu.\n                Maximum length for the text in this field is 150 characters.\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            initial_channel: The ID of any valid public channel to be pre-selected when the menu loads.\n            confirm: A confirm object that defines an optional confirmation dialog that appears\n                after a menu item is selected.\n            response_url_enabled: This field only works with menus in input blocks in modals.\n                When set to true, the view_submission payload from the menu's parent view will contain a response_url.\n                This response_url can be used for message responses.\n                The target channel for the message will be determined by the value of this select menu\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        "
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_channel = initial_channel
        self.response_url_enabled = response_url_enabled

class ChannelMultiSelectElement(InputInteractiveElement):
    type = 'multi_channels_select'

    @property
    def attributes(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return super().attributes.union({'initial_channels', 'max_selected_items'})

    def __init__(self, *, placeholder: Optional[Union[str, dict, TextObject]]=None, action_id: Optional[str]=None, initial_channels: Optional[Sequence[str]]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, max_selected_items: Optional[int]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            i = 10
            return i + 15
        '\n        This multi-select menu will populate its options with a list of public channels visible\n        to the current user in the active workspace.\n        https://api.slack.com/reference/block-kit/block-elements#channel_multi_select\n\n        Args:\n            placeholder (required): A plain_text only text object that defines the placeholder text shown on the menu.\n                Maximum length for the text in this field is 150 characters.\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            initial_channels: An array of one or more IDs of any valid public channel\n                to be pre-selected when the menu loads.\n            confirm: A confirm object that defines an optional confirmation dialog that appears\n                before the multi-select choices are submitted.\n            max_selected_items: Specifies the maximum number of items that can be selected in the menu.\n                Minimum number is 1.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_channels = initial_channels
        self.max_selected_items = max_selected_items

class RichTextInputElement(InputInteractiveElement):
    type = 'rich_text_input'

    @property
    def attributes(self) -> Set[str]:
        if False:
            print('Hello World!')
        return super().attributes.union({'initial_value', 'dispatch_action_config'})

    def __init__(self, *, action_id: Optional[str]=None, placeholder: Optional[Union[str, dict, TextObject]]=None, initial_value: Optional[Dict[str, Any]]=None, dispatch_action_config: Optional[Union[dict, DispatchActionConfig]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_value = initial_value
        self.dispatch_action_config = dispatch_action_config

class PlainTextInputElement(InputInteractiveElement):
    type = 'plain_text_input'

    @property
    def attributes(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return super().attributes.union({'initial_value', 'multiline', 'min_length', 'max_length', 'dispatch_action_config'})

    def __init__(self, *, action_id: Optional[str]=None, placeholder: Optional[Union[str, dict, TextObject]]=None, initial_value: Optional[str]=None, multiline: Optional[bool]=None, min_length: Optional[int]=None, max_length: Optional[int]=None, dispatch_action_config: Optional[Union[dict, DispatchActionConfig]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            while True:
                i = 10
        '\n        A plain-text input, similar to the HTML <input> tag, creates a field\n        where a user can enter freeform data. It can appear as a single-line\n        field or a larger textarea using the multiline flag. Plain-text input\n        elements can be used inside of SectionBlocks and ActionsBlocks.\n        https://api.slack.com/reference/block-kit/block-elements#input\n\n        Args:\n            action_id (required): An identifier for the input value when the parent modal is submitted.\n                You can use this when you receive a view_submission payload to identify the value of the input element.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            placeholder: A plain_text only text object that defines the placeholder text shown\n                in the plain-text input. Maximum length for the text in this field is 150 characters.\n            initial_value: The initial value in the plain-text input when it is loaded.\n            multiline: Indicates whether the input will be a single line (false) or a larger textarea (true).\n                Defaults to false.\n            min_length: The minimum length of input that the user must provide. If the user provides less,\n                they will receive an error. Maximum value is 3000.\n            max_length: The maximum length of input that the user can provide. If the user provides more,\n                they will receive an error.\n            dispatch_action_config: A dispatch configuration object that determines when\n                during text input the element returns a block_actions payload.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_value = initial_value
        self.multiline = multiline
        self.min_length = min_length
        self.max_length = max_length
        self.dispatch_action_config = dispatch_action_config

class EmailInputElement(InputInteractiveElement):
    type = 'email_text_input'

    @property
    def attributes(self) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        return super().attributes.union({'initial_value', 'dispatch_action_config'})

    def __init__(self, *, action_id: Optional[str]=None, initial_value: Optional[str]=None, dispatch_action_config: Optional[Union[dict, DispatchActionConfig]]=None, focus_on_load: Optional[bool]=None, placeholder: Optional[Union[str, dict, TextObject]]=None, **others: dict):
        if False:
            while True:
                i = 10
        '\n        https://api.slack.com/reference/block-kit/block-elements#email\n\n        Args:\n            action_id (required): An identifier for the input value when the parent modal is submitted.\n                You can use this when you receive a view_submission payload to identify the value of the input element.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            initial_value: The initial value in the email input when it is loaded.\n            dispatch_action_config:  dispatch configuration object that determines when during\n                text input the element returns a block_actions payload.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n            placeholder: A plain_text only text object that defines the placeholder text shown in the\n                email input. Maximum length for the text in this field is 150 characters.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_value = initial_value
        self.dispatch_action_config = dispatch_action_config

class UrlInputElement(InputInteractiveElement):
    type = 'url_text_input'

    @property
    def attributes(self) -> Set[str]:
        if False:
            print('Hello World!')
        return super().attributes.union({'initial_value', 'dispatch_action_config'})

    def __init__(self, *, action_id: Optional[str]=None, initial_value: Optional[str]=None, dispatch_action_config: Optional[Union[dict, DispatchActionConfig]]=None, focus_on_load: Optional[bool]=None, placeholder: Optional[Union[str, dict, TextObject]]=None, **others: dict):
        if False:
            print('Hello World!')
        '\n        A URL input element, similar to the Plain-text input element,\n        creates a single line field where a user can enter URL-encoded data.\n        https://api.slack.com/reference/block-kit/block-elements#url\n\n        Args:\n            action_id (required): An identifier for the input value when the parent modal is submitted.\n                You can use this when you receive a view_submission payload to identify the value of the input element.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            initial_value: The initial value in the URL input when it is loaded.\n            dispatch_action_config: A dispatch configuration object that determines when during text input\n                the element returns a block_actions payload.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n            placeholder: A plain_text only text object that defines the placeholder text shown in the URL input.\n                Maximum length for the text in this field is 150 characters.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_value = initial_value
        self.dispatch_action_config = dispatch_action_config

class NumberInputElement(InputInteractiveElement):
    type = 'number_input'

    @property
    def attributes(self) -> Set[str]:
        if False:
            print('Hello World!')
        return super().attributes.union({'initial_value', 'is_decimal_allowed', 'min_value', 'max_value', 'dispatch_action_config'})

    def __init__(self, *, action_id: Optional[str]=None, is_decimal_allowed: Optional[bool]=False, initial_value: Optional[Union[int, float, str]]=None, min_value: Optional[Union[int, float, str]]=None, max_value: Optional[Union[int, float, str]]=None, dispatch_action_config: Optional[Union[dict, DispatchActionConfig]]=None, focus_on_load: Optional[bool]=None, placeholder: Optional[Union[str, dict, TextObject]]=None, **others: dict):
        if False:
            i = 10
            return i + 15
        '\n        https://api.slack.com/reference/block-kit/block-elements#number\n\n        Args:\n            action_id (required): An identifier for the input value when the parent modal is submitted.\n                You can use this when you receive a view_submission payload to identify the value of the input element.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            is_decimal_allowed (required): Decimal numbers are allowed if is_decimal_allowed= true, set the value to\n                false otherwise.\n            initial_value: The initial value in the number input when it is loaded.\n            min_value: The minimum value, cannot be greater than max_value.\n            max_value: The maximum value, cannot be less than min_value.\n            dispatch_action_config: A dispatch configuration object that determines when\n                during text input the element returns a block_actions payload.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n            placeholder: A plain_text only text object that defines the placeholder text shown\n                in the plain-text input. Maximum length for the text in this field is 150 characters.\n        '
        super().__init__(type=self.type, action_id=action_id, placeholder=TextObject.parse(placeholder, PlainTextObject.type), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.initial_value = str(initial_value) if initial_value is not None else None
        self.is_decimal_allowed = is_decimal_allowed
        self.min_value = str(min_value) if min_value is not None else None
        self.max_value = str(max_value) if max_value is not None else None
        self.dispatch_action_config = dispatch_action_config

class FileInputElement(InputInteractiveElement):
    type = 'file_input'

    @property
    def attributes(self) -> Set[str]:
        if False:
            return 10
        return super().attributes.union({'filetypes', 'max_files'})

    def __init__(self, *, action_id: Optional[str]=None, filetypes: Optional[List[str]]=None, max_files: Optional[int]=None, **others: dict):
        if False:
            while True:
                i = 10
        '\n        https://api.slack.com/reference/block-kit/block-elements#file_input\n\n        Args:\n            action_id (required): An identifier for the input value when the parent modal is submitted.\n                You can use this when you receive a view_submission payload to identify the value of the input element.\n                Should be unique among all other action_ids in the containing block. Maximum length is 255 characters.\n            filetypes: An array of valid file extensions that will be accepted for this element.\n                All file extensions will be accepted if filetypes is not specified.\n                This validation is provided for convenience only,\n                and you should perform your own file type validation based on what you expect to receive.\n            max_files: Maximum number of files that can be uploaded for this file_input element.\n                Minimum of 1, maximum of 10. Defaults to 10 if not specified.\n        '
        super().__init__(type=self.type, action_id=action_id)
        show_unknown_key_warning(self, others)
        self.filetypes = filetypes
        self.max_files = max_files

class RadioButtonsElement(InputInteractiveElement):
    type = 'radio_buttons'

    @property
    def attributes(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return super().attributes.union({'options', 'initial_option'})

    def __init__(self, *, action_id: Optional[str]=None, options: Optional[Sequence[Union[dict, Option]]]=None, initial_option: Optional[Union[dict, Option]]=None, confirm: Optional[Union[dict, ConfirmObject]]=None, focus_on_load: Optional[bool]=None, **others: dict):
        if False:
            for i in range(10):
                print('nop')
        'A radio button group that allows a user to choose one item from a list of possible options.\n        https://api.slack.com/reference/block-kit/block-elements#radio\n\n        Args:\n            action_id (required): An identifier for the action triggered when the radio button group is changed.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            options (required): An array of option objects. A maximum of 10 options are allowed.\n            initial_option: An option object that exactly matches one of the options.\n                This option will be selected when the radio button group initially loads.\n            confirm: A confirm object that defines an optional confirmation dialog that appears\n                after clicking one of the radio buttons in this element.\n            focus_on_load: Indicates whether the element will be set to auto focus within the view object.\n                Only one element can be set to true. Defaults to false.\n        '
        super().__init__(type=self.type, action_id=action_id, confirm=ConfirmObject.parse(confirm), focus_on_load=focus_on_load)
        show_unknown_key_warning(self, others)
        self.options = options
        self.initial_option = initial_option

class OverflowMenuElement(InteractiveElement):
    type = 'overflow'
    options_min_length = 1
    options_max_length = 5

    @property
    def attributes(self) -> Set[str]:
        if False:
            return 10
        return super().attributes.union({'confirm', 'options'})

    def __init__(self, *, action_id: Optional[str]=None, options: Sequence[Option], confirm: Optional[Union[dict, ConfirmObject]]=None, **others: dict):
        if False:
            print('Hello World!')
        '\n        This is like a cross between a button and a select menu - when a user clicks\n        on this overflow button, they will be presented with a list of options to\n        choose from. Unlike the select menu, there is no typeahead field, and the\n        button always appears with an ellipsis ("…") rather than customisable text.\n\n        As such, it is usually used if you want a more compact layout than a select\n        menu, or to supply a list of less visually important actions after a row of\n        buttons. You can also specify simple URL links as overflow menu options,\n        instead of actions.\n\n        https://api.slack.com/reference/block-kit/block-elements#overflow\n\n        Args:\n            action_id (required): An identifier for the action triggered when a menu option is selected.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            options (required): An array of option objects to display in the menu.\n                Maximum number of options is 5, minimum is 1.\n            confirm: A confirm object that defines an optional confirmation dialog that appears\n                after a menu item is selected.\n        '
        super().__init__(action_id=action_id, type=self.type)
        show_unknown_key_warning(self, others)
        self.options = options
        self.confirm = ConfirmObject.parse(confirm)

    @JsonValidator(f'options attribute must have between {options_min_length} and {options_max_length} items')
    def _validate_options_length(self) -> bool:
        if False:
            while True:
                i = 10
        return self.options_min_length <= len(self.options) <= self.options_max_length

class WorkflowButtonElement(InteractiveElement):
    type = 'workflow_button'

    @property
    def attributes(self) -> Set[str]:
        if False:
            print('Hello World!')
        return super().attributes.union({'text', 'workflow', 'style', 'accessibility_label'})

    def __init__(self, *, text: Union[str, dict, TextObject], action_id: Optional[str]=None, workflow: Optional[Union[dict, Workflow]]=None, style: Optional[str]=None, accessibility_label: Optional[str]=None, **others: dict):
        if False:
            print('Hello World!')
        'Allows users to run a link trigger with customizable inputs\n        Interactive component - but interactions with workflow button elements will not send block_actions events,\n        since these are used to start new workflow runs.\n        https://api.slack.com/reference/block-kit/block-elements#workflow_button\n\n        Args:\n            text (required): A text object that defines the button\'s text.\n                Can only be of type: plain_text. text may truncate with ~30 characters.\n                Maximum length for the text in this field is 75 characters.\n            action_id (required): An identifier for this action.\n                You can use this when you receive an interaction payload to identify the source of the action.\n                Should be unique among all other action_ids in the containing block.\n                Maximum length for this field is 255 characters.\n            workflow: A workflow object that contains details about the workflow\n                that will run when the button is clicked.\n            style: Decorates buttons with alternative visual color schemes. Use this option with restraint.\n                "primary" gives buttons a green outline and text, ideal for affirmation or confirmation actions.\n                "primary" should only be used for one button within a set.\n                "danger" gives buttons a red outline and text, and should be used when the action is destructive.\n                Use "danger" even more sparingly than "primary".\n                If you don\'t include this field, the default button style will be used.\n            accessibility_label: A label for longer descriptive text about a button element.\n                This label will be read out by screen readers instead of the button text object.\n                Maximum length for this field is 75 characters.\n        '
        super().__init__(action_id=action_id, type=self.type)
        show_unknown_key_warning(self, others)
        self.text = TextObject.parse(text, default_type=PlainTextObject.type)
        self.workflow = workflow
        self.style = style
        self.accessibility_label = accessibility_label