"""Helper to check the configuration file."""
from __future__ import annotations
from collections import OrderedDict
import logging
import os
from pathlib import Path
from typing import NamedTuple, Self
import voluptuous as vol
from homeassistant import loader
from homeassistant.config import CONF_CORE, CONF_PACKAGES, CORE_CONFIG_SCHEMA, YAML_CONFIG_FILE, config_per_platform, extract_domain_configs, format_homeassistant_error, format_schema_error, load_yaml_config_file, merge_packages_config
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.requirements import RequirementsNotFound, async_clear_install_history, async_get_integration_with_requirements
import homeassistant.util.yaml.loader as yaml_loader
from .typing import ConfigType

class CheckConfigError(NamedTuple):
    """Configuration check error."""
    message: str
    domain: str | None
    config: ConfigType | None

class HomeAssistantConfig(OrderedDict):
    """Configuration result with errors attribute."""

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        'Initialize HA config.'
        super().__init__()
        self.errors: list[CheckConfigError] = []
        self.warnings: list[CheckConfigError] = []

    def add_error(self, message: str, domain: str | None=None, config: ConfigType | None=None) -> Self:
        if False:
            print('Hello World!')
        'Add an error.'
        self.errors.append(CheckConfigError(str(message), domain, config))
        return self

    @property
    def error_str(self) -> str:
        if False:
            i = 10
            return i + 15
        'Concatenate all errors to a string.'
        return '\n'.join([err.message for err in self.errors])

    def add_warning(self, message: str, domain: str | None=None, config: ConfigType | None=None) -> Self:
        if False:
            while True:
                i = 10
        'Add a warning.'
        self.warnings.append(CheckConfigError(str(message), domain, config))
        return self

    @property
    def warning_str(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Concatenate all warnings to a string.'
        return '\n'.join([err.message for err in self.warnings])

async def async_check_ha_config_file(hass: HomeAssistant) -> HomeAssistantConfig:
    """Load and check if Home Assistant configuration file is valid.

    This method is a coroutine.
    """
    result = HomeAssistantConfig()
    async_clear_install_history(hass)

    def _pack_error(package: str, component: str, config: ConfigType, message: str) -> None:
        if False:
            i = 10
            return i + 15
        'Handle errors from packages.'
        message = f'Package {package} setup failed. Component {component} {message}'
        domain = f'homeassistant.packages.{package}.{component}'
        pack_config = core_config[CONF_PACKAGES].get(package, config)
        result.add_warning(message, domain, pack_config)

    def _comp_error(ex: vol.Invalid | HomeAssistantError, domain: str, component_config: ConfigType) -> None:
        if False:
            return 10
        'Handle errors from components.'
        if isinstance(ex, vol.Invalid):
            message = format_schema_error(ex, domain, component_config)
        else:
            message = format_homeassistant_error(ex, domain, component_config)
        if domain in frontend_dependencies:
            result.add_error(message, domain, component_config)
        else:
            result.add_warning(message, domain, component_config)

    async def _get_integration(hass: HomeAssistant, domain: str) -> loader.Integration | None:
        """Get an integration."""
        integration: loader.Integration | None = None
        try:
            integration = await async_get_integration_with_requirements(hass, domain)
        except loader.IntegrationNotFound as ex:
            if not hass.config.recovery_mode and (not hass.config.safe_mode):
                result.add_warning(f'Integration error: {domain} - {ex}')
        except RequirementsNotFound as ex:
            result.add_warning(f'Integration error: {domain} - {ex}')
        return integration
    config_path = hass.config.path(YAML_CONFIG_FILE)
    try:
        if not await hass.async_add_executor_job(os.path.isfile, config_path):
            return result.add_error('File configuration.yaml not found.')
        config = await hass.async_add_executor_job(load_yaml_config_file, config_path, yaml_loader.Secrets(Path(hass.config.config_dir)))
    except FileNotFoundError:
        return result.add_error(f'File not found: {config_path}')
    except HomeAssistantError as err:
        return result.add_error(f'Error loading {config_path}: {err}')
    try:
        core_config = config.pop(CONF_CORE, {})
        core_config = CORE_CONFIG_SCHEMA(core_config)
        result[CONF_CORE] = core_config
    except vol.Invalid as err:
        result.add_error(format_schema_error(err, CONF_CORE, core_config), CONF_CORE, core_config)
        core_config = {}
    await merge_packages_config(hass, config, core_config.get(CONF_PACKAGES, {}), _pack_error)
    core_config.pop(CONF_PACKAGES, None)
    components = {key.partition(' ')[0] for key in config}
    frontend_dependencies: set[str] = set()
    if 'frontend' in components or 'default_config' in components:
        frontend = await _get_integration(hass, 'frontend')
        if frontend:
            await frontend.resolve_dependencies()
            frontend_dependencies = frontend.all_dependencies | {'frontend'}
    for domain in components:
        if not (integration := (await _get_integration(hass, domain))):
            continue
        try:
            component = integration.get_component()
        except ImportError as ex:
            result.add_warning(f'Component error: {domain} - {ex}')
            continue
        config_validator = None
        try:
            config_validator = integration.get_platform('config')
        except ImportError as err:
            if err.name != f'{integration.pkg_path}.config':
                result.add_error(f'Error importing config platform {domain}: {err}')
                continue
        if config_validator is not None and hasattr(config_validator, 'async_validate_config'):
            try:
                result[domain] = (await config_validator.async_validate_config(hass, config))[domain]
                continue
            except (vol.Invalid, HomeAssistantError) as ex:
                _comp_error(ex, domain, config)
                continue
            except Exception as err:
                logging.getLogger(__name__).exception('Unexpected error validating config')
                result.add_error(f'Unexpected error calling config validator: {err}', domain, config.get(domain))
                continue
        config_schema = getattr(component, 'CONFIG_SCHEMA', None)
        if config_schema is not None:
            try:
                config = config_schema(config)
                if domain in config:
                    result[domain] = config[domain]
            except vol.Invalid as ex:
                _comp_error(ex, domain, config)
                continue
        component_platform_schema = getattr(component, 'PLATFORM_SCHEMA_BASE', getattr(component, 'PLATFORM_SCHEMA', None))
        if component_platform_schema is None:
            continue
        platforms = []
        for (p_name, p_config) in config_per_platform(config, domain):
            try:
                p_validated = component_platform_schema(p_config)
            except vol.Invalid as ex:
                _comp_error(ex, domain, p_config)
                continue
            if p_name is None:
                platforms.append(p_validated)
                continue
            try:
                p_integration = await async_get_integration_with_requirements(hass, p_name)
                platform = p_integration.get_platform(domain)
            except loader.IntegrationNotFound as ex:
                if not hass.config.recovery_mode and (not hass.config.safe_mode):
                    result.add_warning(f'Platform error {domain}.{p_name} - {ex}')
                continue
            except (RequirementsNotFound, ImportError) as ex:
                result.add_warning(f'Platform error {domain}.{p_name} - {ex}')
                continue
            platform_schema = getattr(platform, 'PLATFORM_SCHEMA', None)
            if platform_schema is not None:
                try:
                    p_validated = platform_schema(p_validated)
                except vol.Invalid as ex:
                    _comp_error(ex, f'{domain}.{p_name}', p_config)
                    continue
            platforms.append(p_validated)
        for filter_comp in extract_domain_configs(config, domain):
            del config[filter_comp]
        result[domain] = platforms
    return result