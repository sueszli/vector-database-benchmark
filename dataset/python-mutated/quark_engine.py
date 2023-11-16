from api_app.analyzers_manager.classes import FileAnalyzer
from api_app.analyzers_manager.exceptions import AnalyzerRunException

class QuarkEngine(FileAnalyzer):

    @classmethod
    def _update(cls):
        if False:
            print('Hello World!')
        from quark import freshquark
        freshquark.download()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        from quark.config import DIR_PATH
        from quark.report import Report
        report = Report()
        report.analysis(self.filepath, DIR_PATH)
        json_report = report.get_report('json')
        if not json_report:
            raise AnalyzerRunException('json report can not be empty')
        return json_report