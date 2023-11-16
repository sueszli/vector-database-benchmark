"""Helpers for creating schema based data entry flows."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Mapping
import copy
from dataclasses import dataclass
import types
from typing import Any, cast
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback, split_entity_id
from homeassistant.data_entry_flow import FlowResult, UnknownHandler
from . import entity_registry as er, selector
from .typing import UNDEFINED, UndefinedType

class SchemaFlowError(Exception):
    """Validation failed."""

@dataclass
class SchemaFlowStep:
    """Define a config or options flow step."""

@dataclass(slots=True)
class SchemaFlowFormStep(SchemaFlowStep):
    """Define a config or options flow form step."""
    schema: vol.Schema | Callable[[SchemaCommonFlowHandler], Coroutine[Any, Any, vol.Schema | None]] | None = None
    'Optional voluptuous schema, or function which returns a schema or None, for\n    requesting and validating user input.\n\n    - If a function is specified, the function will be passed the current\n    `SchemaCommonFlowHandler`.\n    - If schema validation fails, the step will be retried. If the schema is None, no\n    user input is requested.\n    '
    validate_user_input: Callable[[SchemaCommonFlowHandler, dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]] | None = None
    'Optional function to validate user input.\n\n    - The `validate_user_input` function is called if the schema validates successfully.\n    - The first argument is a reference to the current `SchemaCommonFlowHandler`.\n    - The second argument is the user input from the current step.\n    - The `validate_user_input` should raise `SchemaFlowError` if user input is invalid.\n    '
    next_step: Callable[[dict[str, Any]], Coroutine[Any, Any, str | None]] | str | None = None
    'Optional property to identify next step.\n\n    - If `next_step` is a function, it is called if the schema validates successfully or\n      if no schema is defined. The `next_step` function is passed the union of\n      config entry options and user input from previous steps. If the function returns\n      None, the flow is ended with `FlowResultType.CREATE_ENTRY`.\n    - If `next_step` is None, the flow is ended with `FlowResultType.CREATE_ENTRY`.\n    '
    suggested_values: Callable[[SchemaCommonFlowHandler], Coroutine[Any, Any, dict[str, Any]]] | None | UndefinedType = UNDEFINED
    'Optional property to populate suggested values.\n\n    - If `suggested_values` is UNDEFINED, each key in the schema will get a suggested\n      value from an option with the same key.\n\n    Note: if a step is retried due to a validation failure, then the user input will\n    have priority over the suggested values.\n    '
    preview: str | None = None
    'Optional preview component.'

@dataclass(slots=True)
class SchemaFlowMenuStep(SchemaFlowStep):
    """Define a config or options flow menu step."""
    options: list[str] | dict[str, str]

class SchemaCommonFlowHandler:
    """Handle a schema based config or options flow."""

    def __init__(self, handler: SchemaConfigFlowHandler | SchemaOptionsFlowHandler, flow: Mapping[str, SchemaFlowStep], options: dict[str, Any] | None) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize a common handler.'
        self._flow = flow
        self._handler = handler
        self._options = options if options is not None else {}
        self._flow_state: dict[str, Any] = {}

    @property
    def parent_handler(self) -> SchemaConfigFlowHandler | SchemaOptionsFlowHandler:
        if False:
            return 10
        'Return parent handler.'
        return self._handler

    @property
    def options(self) -> dict[str, Any]:
        if False:
            print('Hello World!')
        'Return the options linked to the current flow handler.'
        return self._options

    @property
    def flow_state(self) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return the flow state, used to store temporary data.\n\n        It can be used for example to store the key or the index of a sub-item\n        that will be edited in the next step.\n        '
        return self._flow_state

    async def async_step(self, step_id: str, user_input: dict[str, Any] | None=None) -> FlowResult:
        """Handle a step."""
        if isinstance(self._flow[step_id], SchemaFlowFormStep):
            return await self._async_form_step(step_id, user_input)
        return await self._async_menu_step(step_id, user_input)

    async def _get_schema(self, form_step: SchemaFlowFormStep) -> vol.Schema | None:
        if form_step.schema is None:
            return None
        if isinstance(form_step.schema, vol.Schema):
            return form_step.schema
        return await form_step.schema(self)

    async def _async_form_step(self, step_id: str, user_input: dict[str, Any] | None=None) -> FlowResult:
        """Handle a form step."""
        form_step: SchemaFlowFormStep = cast(SchemaFlowFormStep, self._flow[step_id])
        if user_input is not None and (data_schema := (await self._get_schema(form_step))) and data_schema.schema and (not self._handler.show_advanced_options):
            for key in data_schema.schema:
                if isinstance(key, (vol.Optional, vol.Required)):
                    if key.description and key.description.get('advanced') and (key.default is not vol.UNDEFINED) and (key not in self._options):
                        user_input[str(key.schema)] = key.default()
        if user_input is not None and form_step.validate_user_input is not None:
            try:
                user_input = await form_step.validate_user_input(self, user_input)
            except SchemaFlowError as exc:
                return await self._show_next_step(step_id, exc, user_input)
        if user_input is not None:
            self._update_and_remove_omitted_optional_keys(self._options, user_input, data_schema)
        if user_input is not None or form_step.schema is None:
            return await self._show_next_step_or_create_entry(form_step)
        return await self._show_next_step(step_id)

    def _update_and_remove_omitted_optional_keys(self, values: dict[str, Any], user_input: dict[str, Any], data_schema: vol.Schema | None) -> None:
        if False:
            print('Hello World!')
        values.update(user_input)
        if data_schema and data_schema.schema:
            for key in data_schema.schema:
                if isinstance(key, vol.Optional) and key not in user_input and (not (key.description and key.description.get('advanced') and (not self._handler.show_advanced_options))):
                    values.pop(key, None)

    async def _show_next_step_or_create_entry(self, form_step: SchemaFlowFormStep) -> FlowResult:
        next_step_id_or_end_flow: str | None
        if callable(form_step.next_step):
            next_step_id_or_end_flow = await form_step.next_step(self._options)
        else:
            next_step_id_or_end_flow = form_step.next_step
        if next_step_id_or_end_flow is None:
            return self._handler.async_create_entry(data=self._options)
        return await self._show_next_step(next_step_id_or_end_flow)

    async def _show_next_step(self, next_step_id: str, error: SchemaFlowError | None=None, user_input: dict[str, Any] | None=None) -> FlowResult:
        """Show form for next step."""
        if isinstance(self._flow[next_step_id], SchemaFlowMenuStep):
            menu_step = cast(SchemaFlowMenuStep, self._flow[next_step_id])
            return self._handler.async_show_menu(step_id=next_step_id, menu_options=menu_step.options)
        form_step = cast(SchemaFlowFormStep, self._flow[next_step_id])
        if (data_schema := (await self._get_schema(form_step))) is None:
            return await self._show_next_step_or_create_entry(form_step)
        suggested_values: dict[str, Any] = {}
        if form_step.suggested_values is UNDEFINED:
            suggested_values = self._options
        elif form_step.suggested_values:
            suggested_values = await form_step.suggested_values(self)
        if user_input:
            suggested_values = copy.deepcopy(suggested_values)
            self._update_and_remove_omitted_optional_keys(suggested_values, user_input, await self._get_schema(form_step))
        if data_schema.schema:
            data_schema = self._handler.add_suggested_values_to_schema(data_schema, suggested_values)
        errors = {'base': str(error)} if error else None
        last_step = None
        if not callable(form_step.next_step):
            last_step = form_step.next_step is None
        return self._handler.async_show_form(step_id=next_step_id, data_schema=data_schema, errors=errors, last_step=last_step, preview=form_step.preview)

    async def _async_menu_step(self, step_id: str, user_input: dict[str, Any] | None=None) -> FlowResult:
        """Handle a menu step."""
        menu_step: SchemaFlowMenuStep = cast(SchemaFlowMenuStep, self._flow[step_id])
        return self._handler.async_show_menu(step_id=step_id, menu_options=menu_step.options)

class SchemaConfigFlowHandler(config_entries.ConfigFlow, ABC):
    """Handle a schema based config flow."""
    config_flow: Mapping[str, SchemaFlowStep]
    options_flow: Mapping[str, SchemaFlowStep] | None = None
    VERSION = 1

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Initialize a subclass.'
        super().__init_subclass__(**kwargs)

        @callback
        def _async_get_options_flow(config_entry: config_entries.ConfigEntry) -> config_entries.OptionsFlow:
            if False:
                while True:
                    i = 10
            'Get the options flow for this handler.'
            if cls.options_flow is None:
                raise UnknownHandler
            return SchemaOptionsFlowHandler(config_entry, cls.options_flow, cls.async_options_flow_finished, cls.async_setup_preview)
        cls.async_get_options_flow = _async_get_options_flow
        for step in cls.config_flow:
            setattr(cls, f'async_step_{step}', cls._async_step(step))

    def __init__(self) -> None:
        if False:
            return 10
        'Initialize config flow.'
        self._common_handler = SchemaCommonFlowHandler(self, self.config_flow, None)

    @staticmethod
    async def async_setup_preview(hass: HomeAssistant) -> None:
        """Set up preview."""

    @classmethod
    @callback
    def async_supports_options_flow(cls, config_entry: config_entries.ConfigEntry) -> bool:
        if False:
            print('Hello World!')
        'Return options flow support for this handler.'
        return cls.options_flow is not None

    @staticmethod
    def _async_step(step_id: str) -> Callable:
        if False:
            i = 10
            return i + 15
        'Generate a step handler.'

        async def _async_step(self: SchemaConfigFlowHandler, user_input: dict[str, Any] | None=None) -> FlowResult:
            """Handle a config flow step."""
            result = await self._common_handler.async_step(step_id, user_input)
            return result
        return _async_step

    @abstractmethod
    @callback
    def async_config_entry_title(self, options: Mapping[str, Any]) -> str:
        if False:
            i = 10
            return i + 15
        'Return config entry title.\n\n        The options parameter contains config entry options, which is the union of user\n        input from the config flow steps.\n        '

    @callback
    def async_config_flow_finished(self, options: Mapping[str, Any]) -> None:
        if False:
            while True:
                i = 10
        'Take necessary actions after the config flow is finished, if needed.\n\n        The options parameter contains config entry options, which is the union of user\n        input from the config flow steps.\n        '

    @callback
    @staticmethod
    def async_options_flow_finished(hass: HomeAssistant, options: Mapping[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Take necessary actions after the options flow is finished, if needed.\n\n        The options parameter contains config entry options, which is the union of\n        stored options and user input from the options flow steps.\n        '

    @callback
    def async_create_entry(self, data: Mapping[str, Any], **kwargs: Any) -> FlowResult:
        if False:
            return 10
        'Finish config flow and create a config entry.'
        self.async_config_flow_finished(data)
        return super().async_create_entry(data={}, options=data, title=self.async_config_entry_title(data), **kwargs)

class SchemaOptionsFlowHandler(config_entries.OptionsFlowWithConfigEntry):
    """Handle a schema based options flow."""

    def __init__(self, config_entry: config_entries.ConfigEntry, options_flow: Mapping[str, SchemaFlowStep], async_options_flow_finished: Callable[[HomeAssistant, Mapping[str, Any]], None] | None=None, async_setup_preview: Callable[[HomeAssistant], Coroutine[Any, Any, None]] | None=None) -> None:
        if False:
            print('Hello World!')
        'Initialize options flow.\n\n        If needed, `async_options_flow_finished` can be set to take necessary actions\n        after the options flow is finished. The second parameter contains config entry\n        options, which is the union of stored options and user input from the options\n        flow steps.\n        '
        super().__init__(config_entry)
        self._common_handler = SchemaCommonFlowHandler(self, options_flow, self._options)
        self._async_options_flow_finished = async_options_flow_finished
        for step in options_flow:
            setattr(self, f'async_step_{step}', types.MethodType(self._async_step(step), self))
        if async_setup_preview:
            setattr(self, 'async_setup_preview', async_setup_preview)

    @staticmethod
    def _async_step(step_id: str) -> Callable:
        if False:
            return 10
        'Generate a step handler.'

        async def _async_step(self: SchemaConfigFlowHandler, user_input: dict[str, Any] | None=None) -> FlowResult:
            """Handle an options flow step."""
            result = await self._common_handler.async_step(step_id, user_input)
            return result
        return _async_step

    @callback
    def async_create_entry(self, data: Mapping[str, Any], **kwargs: Any) -> FlowResult:
        if False:
            while True:
                i = 10
        'Finish config flow and create a config entry.'
        if self._async_options_flow_finished:
            self._async_options_flow_finished(self.hass, data)
        return super().async_create_entry(data=data, **kwargs)

@callback
def wrapped_entity_config_entry_title(hass: HomeAssistant, entity_id_or_uuid: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    "Generate title for a config entry wrapping a single entity.\n\n    If the entity is registered, use the registry entry's name.\n    If the entity is in the state machine, use the name from the state.\n    Otherwise, fall back to the object ID.\n    "
    registry = er.async_get(hass)
    entity_id = er.async_validate_entity_id(registry, entity_id_or_uuid)
    object_id = split_entity_id(entity_id)[1]
    entry = registry.async_get(entity_id)
    if entry:
        return entry.name or entry.original_name or object_id
    state = hass.states.get(entity_id)
    if state:
        return state.name or object_id
    return object_id

@callback
def entity_selector_without_own_entities(handler: SchemaOptionsFlowHandler, entity_selector_config: selector.EntitySelectorConfig) -> vol.Schema:
    if False:
        return 10
    'Return an entity selector which excludes own entities.'
    entity_registry = er.async_get(handler.hass)
    entities = er.async_entries_for_config_entry(entity_registry, handler.config_entry.entry_id)
    entity_ids = [ent.entity_id for ent in entities]
    final_selector_config = entity_selector_config.copy()
    final_selector_config['exclude_entities'] = entity_ids
    return selector.EntitySelector(final_selector_config)