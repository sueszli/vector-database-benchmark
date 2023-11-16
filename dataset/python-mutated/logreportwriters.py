from os.path import basename, splitext
from robot.htmldata import HtmlFileWriter, ModelWriter, LOG, REPORT
from robot.utils import file_writer, is_string
from .jswriter import JsResultWriter, SplitLogWriter

class _LogReportWriter:
    usage = None

    def __init__(self, js_model):
        if False:
            while True:
                i = 10
        self._js_model = js_model

    def _write_file(self, path, config, template):
        if False:
            return 10
        outfile = file_writer(path, usage=self.usage) if is_string(path) else path
        with outfile:
            model_writer = RobotModelWriter(outfile, self._js_model, config)
            writer = HtmlFileWriter(outfile, model_writer)
            writer.write(template)

class LogWriter(_LogReportWriter):
    usage = 'log'

    def write(self, path, config):
        if False:
            while True:
                i = 10
        self._write_file(path, config, LOG)
        if self._js_model.split_results:
            self._write_split_logs(splitext(path)[0])

    def _write_split_logs(self, base):
        if False:
            i = 10
            return i + 15
        for (index, (keywords, strings)) in enumerate(self._js_model.split_results, start=1):
            self._write_split_log(index, keywords, strings, '%s-%d.js' % (base, index))

    def _write_split_log(self, index, keywords, strings, path):
        if False:
            i = 10
            return i + 15
        with file_writer(path, usage=self.usage) as outfile:
            writer = SplitLogWriter(outfile)
            writer.write(keywords, strings, index, basename(path))

class ReportWriter(_LogReportWriter):
    usage = 'report'

    def write(self, path, config):
        if False:
            print('Hello World!')
        self._write_file(path, config, REPORT)

class RobotModelWriter(ModelWriter):

    def __init__(self, output, model, config):
        if False:
            while True:
                i = 10
        self._output = output
        self._model = model
        self._config = config

    def write(self, line):
        if False:
            for i in range(10):
                print('nop')
        JsResultWriter(self._output).write(self._model, self._config)