import six
import csv
import json
import itertools
from io import StringIO, BytesIO
from six import iteritems

def result_formater(results):
    if False:
        i = 10
        return i + 15
    common_fields = None
    for result in results:
        result.setdefault('result', None)
        if isinstance(result['result'], dict):
            if common_fields is None:
                common_fields = set(result['result'].keys())
            else:
                common_fields &= set(result['result'].keys())
        else:
            common_fields = set()
    for result in results:
        result['result_formated'] = {}
        if not common_fields:
            result['others'] = result['result']
        elif not isinstance(result['result'], dict):
            result['others'] = result['result']
        else:
            result_formated = {}
            others = {}
            for (key, value) in iteritems(result['result']):
                if key in common_fields:
                    result_formated[key] = value
                else:
                    others[key] = value
            result['result_formated'] = result_formated
            result['others'] = others
    return (common_fields or set(), results)

def dump_as_json(results, valid=False):
    if False:
        return 10
    first = True
    if valid:
        yield '['
    for result in results:
        if valid:
            if first:
                first = False
            else:
                yield ', '
        yield (json.dumps(result, ensure_ascii=False) + '\n')
    if valid:
        yield ']'

def dump_as_txt(results):
    if False:
        print('Hello World!')
    for result in results:
        yield (result.get('url', None) + '\t' + json.dumps(result.get('result', None), ensure_ascii=False) + '\n')

def dump_as_csv(results):
    if False:
        return 10

    def toString(obj):
        if False:
            print('Hello World!')
        if isinstance(obj, six.binary_type):
            if six.PY2:
                return obj
            else:
                return obj.decode('utf8')
        elif isinstance(obj, six.text_type):
            if six.PY2:
                return obj.encode('utf8')
            else:
                return obj
        elif six.PY2:
            return json.dumps(obj, ensure_ascii=False).encode('utf8')
        else:
            return json.dumps(obj, ensure_ascii=False)
    if six.PY2:
        stringio = BytesIO()
    else:
        stringio = StringIO()
    csv_writer = csv.writer(stringio)
    it = iter(results)
    first_30 = []
    for result in it:
        first_30.append(result)
        if len(first_30) >= 30:
            break
    (common_fields, _) = result_formater(first_30)
    common_fields_l = sorted(common_fields)
    csv_writer.writerow([toString('url')] + [toString(x) for x in common_fields_l] + [toString('...')])
    for result in itertools.chain(first_30, it):
        result['result_formated'] = {}
        if not common_fields:
            result['others'] = result['result']
        elif not isinstance(result['result'], dict):
            result['others'] = result['result']
        else:
            result_formated = {}
            others = {}
            for (key, value) in iteritems(result['result']):
                if key in common_fields:
                    result_formated[key] = value
                else:
                    others[key] = value
            result['result_formated'] = result_formated
            result['others'] = others
        csv_writer.writerow([toString(result['url'])] + [toString(result['result_formated'].get(k, '')) for k in common_fields_l] + [toString(result['others'])])
        yield stringio.getvalue()
        stringio.truncate(0)
        stringio.seek(0)