from __future__ import annotations
from pathlib import Path
from tabulate import tabulate
AIRFLOW_SOURCES = Path(__file__).resolve().parent.parent.parent.parent
HEADERS = ('Version', 'Current Patch/Minor', 'State', 'First Release', 'Limited Support', 'EOL/Terminated')
SUPPORTED_VERSIONS = (('2', '2.7.3', 'Supported', 'Dec 17, 2020', 'TBD', 'TBD'), ('1.10', '1.10.15', 'EOL', 'Aug 27, 2018', 'Dec 17, 2020', 'June 17, 2021'), ('1.9', '1.9.0', 'EOL', 'Jan 03, 2018', 'Aug 27, 2018', 'Aug 27, 2018'), ('1.8', '1.8.2', 'EOL', 'Mar 19, 2017', 'Jan 03, 2018', 'Jan 03, 2018'), ('1.7', '1.7.1.2', 'EOL', 'Mar 28, 2016', 'Mar 19, 2017', 'Mar 19, 2017'))

def replace_text_between(file: Path, start: str, end: str, replacement_text: str):
    if False:
        while True:
            i = 10
    original_text = file.read_text()
    leading_text = original_text.split(start)[0]
    trailing_text = original_text.split(end)[1]
    file.write_text(leading_text + start + replacement_text + end + trailing_text)
if __name__ == '__main__':
    replace_text_between(file=AIRFLOW_SOURCES / 'README.md', start='<!-- Beginning of auto-generated table -->\n', end='<!-- End of auto-generated table -->\n', replacement_text='\n' + tabulate(SUPPORTED_VERSIONS, tablefmt='github', headers=HEADERS, stralign='left', disable_numparse=True) + '\n\n')
    replace_text_between(file=AIRFLOW_SOURCES / 'docs' / 'apache-airflow' / 'installation' / 'supported-versions.rst', start=' .. Beginning of auto-generated table\n', end=' .. End of auto-generated table\n', replacement_text='\n' + tabulate(SUPPORTED_VERSIONS, tablefmt='rst', headers=HEADERS, stralign='left', disable_numparse=True) + '\n\n')