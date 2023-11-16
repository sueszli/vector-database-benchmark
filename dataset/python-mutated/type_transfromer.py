import logging
from datetime import datetime
from typing import Any, Dict
from airbyte_cdk.sources.utils.transform import TypeTransformer
logger = logging.getLogger('airbyte')

class DateTimeTransformer(TypeTransformer):
    api_date_time_format = '%Y-%m-%dT%H:%M:%S.%f%z'

    @staticmethod
    def default_convert(original_item: Any, subschema: Dict[str, Any]) -> Any:
        if False:
            for i in range(10):
                print('nop')
        target_format = subschema.get('format', '')
        if target_format == 'date-time':
            if isinstance(original_item, str):
                try:
                    date = datetime.strptime(original_item, DateTimeTransformer.api_date_time_format)
                    return date.isoformat()
                except ValueError:
                    logger.warning(f"{original_item}: doesn't match expected format.")
                    return original_item
        return original_item