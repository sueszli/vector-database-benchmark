from odoo.tests import common
from odoo.addons.calendar.models.calendar import calendar_id2real_id

class TestRecurrentEvent(common.TransactionCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestRecurrentEvent, self).setUp()
        self.CalendarEvent = self.env['calendar.event']

    def test_recurrent_meeting1(self):
        if False:
            return 10
        self.CalendarEvent.create({'count': 5, 'start': '2011-04-13 11:04:00', 'stop': '2011-04-13 12:04:00', 'duration': 1.0, 'name': 'Test Meeting', 'recurrency': True, 'rrule_type': 'daily'})
        meetings_count = self.CalendarEvent.with_context({'virtual_id': True}).search_count([('start', '>=', '2011-03-13'), ('stop', '<=', '2011-05-13')])
        self.assertEqual(meetings_count, 5, 'Recurrent daily meetings are not created !')

    def test_recurrent_meeting2(self):
        if False:
            return 10
        self.CalendarEvent.create({'start': '2011-04-18 11:47:00', 'stop': '2011-04-18 12:47:00', 'day': 0.0, 'duration': 1.0, 'final_date': '2011-04-30', 'end_type': 'end_date', 'fr': True, 'mo': True, 'th': True, 'tu': True, 'we': True, 'name': 'Review code with programmer', 'recurrency': True, 'rrule_type': 'weekly'})
        meetings_count = self.CalendarEvent.with_context({'virtual_id': True}).search_count([('start', '>=', '2011-03-13'), ('stop', '<=', '2011-05-13')])
        self.assertEqual(meetings_count, 10, 'Recurrent weekly meetings are not created !')

    def test_recurrent_meeting3(self):
        if False:
            i = 10
            return i + 15
        self.calendar_event_sprint_review = self.CalendarEvent.create({'count': 12, 'start': '2011-04-01 12:01:00', 'stop': '2011-04-01 13:01:00', 'day': 1, 'duration': 1.0, 'name': 'Sprint Review', 'recurrency': True, 'rrule_type': 'monthly'})
        meetings_count = self.CalendarEvent.with_context({'virtual_id': True}).search_count([('start', '>=', '2011-03-01'), ('stop', '<=', '2012-05-13')])
        self.assertEqual(meetings_count, 12, 'Recurrent weekly meetings are not created !')
        idval = '%d-%s' % (self.calendar_event_sprint_review.id, '20110901130100')
        self.CalendarEvent.browse(idval).write({'name': 'Sprint Review for google modules'})
        meetings = self.CalendarEvent.with_context({'virtual_id': True}).search([('start', '>=', '2011-03-01'), ('stop', '<=', '2012-05-13')])
        for meeting in meetings:
            self.assertEqual(meeting.name, 'Sprint Review for google modules', 'Name not changed for id: %s' % meeting.id)
        idval = '%d-%s' % (self.calendar_event_sprint_review.id, '20110425124700')
        self.CalendarEvent.browse(idval).write({'description': 'Review code of the module: sync_google_calendar.'})
        meetings = self.CalendarEvent.search([('recurrent_id', '=', self.calendar_event_sprint_review.id)])
        for meeting in meetings:
            self.assertEqual(meeting.description, 'Review code of the module: sync_google_calendar.', 'Description not changed for id: %s' % meeting.id)
        self.calendar_event_sprint_review.write({'description': 'Some description'})
        self.assertEqual(self.calendar_event_sprint_review.description, 'Some description', 'Event %d has not been updated' % self.calendar_event_sprint_review.id)

    def test_recurrent_meeting4(self):
        if False:
            i = 10
            return i + 15
        self.CalendarEvent.create({'start': '2017-01-22 11:47:00', 'stop': '2017-01-22 12:47:00', 'day': 0.0, 'duration': 1.0, 'final_date': '2017-06-30', 'end_type': 'end_date', 'fr': True, 'mo': True, 'th': True, 'tu': True, 'we': True, 'name': 'Review code with programmer', 'recurrency': True, 'rrule_type': 'weekly'})
        meetings_count = self.CalendarEvent.with_context({'virtual_id': True}).search_count([('start', '<=', '2017-01-24'), ('stop', '>=', '2017-01-24'), ('name', '=', 'Review code with programmer')])
        self.assertEqual(meetings_count, 1, 'Recurrent weekly meetings are not found using date filter !')
        meetings_count = self.CalendarEvent.with_context({'virtual_id': True}).search_count([('start', '<=', '2017-01-24 11:55:00'), ('stop', '>=', '2017-01-24 11:55:00'), ('name', '=', 'Review code with programmer')])
        self.assertEqual(meetings_count, 1, 'Recurrent weekly meetings are not found using time filter !')
        meetings_count = self.CalendarEvent.with_context({'virtual_id': True}).search_count([('start', '!=', False), ('stop', '>=', '2017-06-30 11:55:00'), ('name', '=', 'Review code with programmer')])
        self.assertEqual(meetings_count, 1, "Last recurrent weekly meetings are not found using 'is set' filter !")
        meetings_count = self.CalendarEvent.with_context({'virtual_id': True}).search_count([('start', '<=', '2017-01-24 11:55:00'), ('stop', '>=', '2017-01-24 15:55:00')])
        self.assertEqual(meetings_count, 0, 'Too late recurrent meetings are found using time filter !')
        meetings_count = self.CalendarEvent.with_context({'virtual_id': True}).search_count([('start', '>=', '2017-06-30 08:00:00'), ('name', '=', 'Review code with programmer')])
        self.assertEqual(meetings_count, 1, 'Last recurrent weekly meetings are not found without stop filter !')

    def test_recurrent_meeting5(self):
        if False:
            return 10
        self.CalendarEvent.create({'count': 5, 'start': '2012-04-13 11:00:00', 'stop': '2012-04-13 12:00:00', 'duration': 1.0, 'name': 'Test Meeting', 'recurrency': True, 'rrule_type': 'daily'})
        meeting = self.CalendarEvent.with_context({'virtual_id': True}).search([('start', '=', '2012-04-13 11:00:00'), ('stop', '=', '2012-04-13 12:00:00')])
        virutal_dates = calendar_id2real_id(meeting.id, with_date=True)
        self.assertEqual(virutal_dates[1], '2012-04-13 11:00:00', "The virtual event doesn't have the correct start date !")
        self.assertEqual(virutal_dates[2], '2012-04-13 12:00:00', "The virtual event doesn't have the correct stop date !")