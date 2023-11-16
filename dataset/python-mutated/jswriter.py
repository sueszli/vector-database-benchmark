from robot.htmldata import JsonWriter

class JsResultWriter:
    _output_attr = 'window.output'
    _settings_attr = 'window.settings'
    _suite_key = 'suite'
    _strings_key = 'strings'

    def __init__(self, output, start_block='<script type="text/javascript">\n', end_block='</script>\n', split_threshold=9500):
        if False:
            while True:
                i = 10
        writer = JsonWriter(output, separator=end_block + start_block)
        self._write = writer.write
        self._write_json = writer.write_json
        self._start_block = start_block
        self._end_block = end_block
        self._split_threshold = split_threshold

    def write(self, result, settings):
        if False:
            for i in range(10):
                print('nop')
        self._start_output_block()
        self._write_suite(result.suite)
        self._write_strings(result.strings)
        self._write_data(result.data)
        self._write_settings_and_end_output_block(settings)

    def _start_output_block(self):
        if False:
            print('Hello World!')
        self._write(self._start_block, postfix='', separator=False)
        self._write('%s = {}' % self._output_attr)

    def _write_suite(self, suite):
        if False:
            i = 10
            return i + 15
        writer = SuiteWriter(self._write_json, self._split_threshold)
        writer.write(suite, self._output_var(self._suite_key))

    def _write_strings(self, strings):
        if False:
            return 10
        variable = self._output_var(self._strings_key)
        self._write('%s = []' % variable)
        prefix = '%s = %s.concat(' % (variable, variable)
        postfix = ');\n'
        threshold = self._split_threshold
        for index in range(0, len(strings), threshold):
            self._write_json(prefix, strings[index:index + threshold], postfix)

    def _write_data(self, data):
        if False:
            return 10
        for key in data:
            self._write_json('%s = ' % self._output_var(key), data[key])

    def _write_settings_and_end_output_block(self, settings):
        if False:
            for i in range(10):
                print('nop')
        self._write_json('%s = ' % self._settings_attr, settings, separator=False)
        self._write(self._end_block, postfix='', separator=False)

    def _output_var(self, key):
        if False:
            while True:
                i = 10
        return '%s["%s"]' % (self._output_attr, key)

class SuiteWriter:

    def __init__(self, write_json, split_threshold):
        if False:
            i = 10
            return i + 15
        self._write_json = write_json
        self._split_threshold = split_threshold

    def write(self, suite, variable):
        if False:
            while True:
                i = 10
        mapping = {}
        self._write_parts_over_threshold(suite, mapping)
        self._write_json('%s = ' % variable, suite, mapping=mapping)

    def _write_parts_over_threshold(self, data, mapping):
        if False:
            while True:
                i = 10
        if not isinstance(data, tuple):
            return 1
        not_written = 1 + sum((self._write_parts_over_threshold(item, mapping) for item in data))
        if not_written > self._split_threshold:
            self._write_part(data, mapping)
            return 1
        return not_written

    def _write_part(self, data, mapping):
        if False:
            while True:
                i = 10
        part_name = 'window.sPart%d' % len(mapping)
        self._write_json('%s = ' % part_name, data, mapping=mapping)
        mapping[data] = part_name

class SplitLogWriter:

    def __init__(self, output):
        if False:
            print('Hello World!')
        self._writer = JsonWriter(output)

    def write(self, keywords, strings, index, notify):
        if False:
            while True:
                i = 10
        self._writer.write_json('window.keywords%d = ' % index, keywords)
        self._writer.write_json('window.strings%d = ' % index, strings)
        self._writer.write('window.fileLoading.notify("%s")' % notify)