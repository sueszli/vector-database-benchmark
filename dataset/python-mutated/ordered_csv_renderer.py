from collections import OrderedDict
from typing import Any, Dict, Generator
from more_itertools import unique_everseen
from rest_framework_csv.renderers import CSVRenderer

class OrderedCsvRenderer(CSVRenderer):

    def tablize(self, data: Any, header: Any=None, labels: Any=None) -> Generator:
        if False:
            return 10
        '\n        Convert a list of data into a table.\n        '
        if not header and hasattr(data, 'header'):
            header = data.header
        if data:
            data = self.flatten_data(data)
            if not header:
                data = tuple(data)
                headers = []
                for item in data:
                    headers.extend(item.keys())
                unique_fields = list(unique_everseen(headers))
                ordered_fields: Dict[str, Any] = OrderedDict()
                for item in unique_fields:
                    field = item.split('.')
                    field = field[0]
                    if field in ordered_fields:
                        ordered_fields[field].append(item)
                    else:
                        ordered_fields[field] = [item]
                header = []
                for fields in ordered_fields.values():
                    for field in fields:
                        header.append(field)
            if labels:
                yield [labels.get(x, x) for x in header]
            else:
                yield header
            for item in data:
                row = [item.get(key, None) for key in header]
                yield row
        else:
            return []