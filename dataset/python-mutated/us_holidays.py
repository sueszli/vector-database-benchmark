""" Calendar file of US Holidays from Mozilla provided by `icalendar`_.

License `CC BY-SA 3.0`_

Sourced from: https://www.mozilla.org/en-US/projects/calendar/holidays/

This module contains one list: ``us_holidays``.

.. rubric:: ``us_holidays``

.. code-block::

    [
        (datetime.date(1966, 12, 26), 'Kwanzaa'),
        (datetime.date(2000, 1, 1), "New Year's Day"),
        ...
        (datetime.date(2020, 12, 25), 'Christmas Day (US-OPM)')
    ]

.. bokeh-sampledata-xref:: us_holidays

.. _icalendar: https://pypi.org/project/icalendar/
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from datetime import datetime
from ..util.dependencies import import_required
from ..util.sampledata import package_path
__all__ = ('us_holidays',)

def _read_data() -> list[tuple[datetime, str]]:
    if False:
        while True:
            i = 10
    '\n\n    '
    ic = import_required('icalendar', 'us_holidays data requires icalendar (http://icalendar.readthedocs.org) to be installed')
    with open(package_path('USHolidays.ics')) as f:
        data = ic.Calendar.from_ical(f.read())
    return sorted(((comp.get('dtstart').dt, str(comp.get('summary'))) for comp in data.walk() if comp.name == 'VEVENT'))
us_holidays = _read_data()