"""
Tests for importwizard.py
"""
import pytest
from spyder.plugins.variableexplorer.widgets.importwizard import ImportWizard

@pytest.fixture
def importwizard(qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Set up ImportWizard.'
    text = u'17/11/1976\t1.34\n14/05/09\t3.14'
    importwizard = ImportWizard(None, text)
    qtbot.addWidget(importwizard)
    return importwizard

def test_importwizard(importwizard):
    if False:
        print('Hello World!')
    'Run ImportWizard dialog.'
    importwizard.show()
    assert importwizard
if __name__ == '__main__':
    pytest.main()