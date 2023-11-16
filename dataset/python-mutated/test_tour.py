"""
Tests for tour.py
"""
import pytest
from spyder.plugins.tours.widgets import TourTestWindow

@pytest.fixture
def tour(qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Setup the QMainWindow for the tour.'
    tour = TourTestWindow()
    qtbot.addWidget(tour)
    return tour

def test_tour(tour, qtbot):
    if False:
        while True:
            i = 10
    'Test tour.'
    tour.show()
    assert tour
if __name__ == '__main__':
    pytest.main()