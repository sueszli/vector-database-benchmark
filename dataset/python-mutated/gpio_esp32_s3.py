import logging
from esphome.const import CONF_INPUT, CONF_MODE, CONF_NUMBER
import esphome.config_validation as cv
from esphome.pins import check_strapping_pin
_ESP_32S3_SPI_PSRAM_PINS = {26: 'SPICS1', 27: 'SPIHD', 28: 'SPIWP', 29: 'SPICS0', 30: 'SPICLK', 31: 'SPIQ', 32: 'SPID'}
_ESP_32_ESP32_S3R8_PSRAM_PINS = {33: 'SPIIO4', 34: 'SPIIO5', 35: 'SPIIO6', 36: 'SPIIO7', 37: 'SPIDQS'}
_ESP_32S3_STRAPPING_PINS = {0, 3, 45, 46}
_LOGGER = logging.getLogger(__name__)

def esp32_s3_validate_gpio_pin(value):
    if False:
        while True:
            i = 10
    if value < 0 or value > 48:
        raise cv.Invalid(f'Invalid pin number: {value} (must be 0-46)')
    if value in _ESP_32S3_SPI_PSRAM_PINS:
        raise cv.Invalid(f'This pin cannot be used on ESP32-S3s and is already used by the SPI/PSRAM interface(function: {_ESP_32S3_SPI_PSRAM_PINS[value]})')
    if value in _ESP_32_ESP32_S3R8_PSRAM_PINS:
        _LOGGER.warning('GPIO%d is used by the PSRAM interface on ESP32-S3R8 / ESP32-S3R8V and should be avoided on these models', value)
    if value in (22, 23, 24, 25):
        raise cv.Invalid(f'The pin GPIO{value} is not usable on ESP32-S3s.')
    return value

def esp32_s3_validate_supports(value):
    if False:
        i = 10
        return i + 15
    num = value[CONF_NUMBER]
    mode = value[CONF_MODE]
    is_input = mode[CONF_INPUT]
    if num < 0 or num > 48:
        raise cv.Invalid(f'Invalid pin number: {num} (must be 0-46)')
    if is_input:
        pass
    check_strapping_pin(value, _ESP_32S3_STRAPPING_PINS, _LOGGER)
    return value