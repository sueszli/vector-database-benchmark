from opensfm import report
from opensfm.dataset import DataSet

def run_dataset(data: DataSet) -> None:
    if False:
        i = 10
        return i + 15
    'Export a nice report based on previously generated statistics\n\n    Args:\n        data: dataset object\n\n    '
    pdf_report = report.Report(data)
    pdf_report.generate_report()
    pdf_report.save_report('report.pdf')