from odoo import fields
from odoo.tests.common import TransactionCase

class TestCalendar(TransactionCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestCalendar, self).setUp()
        self.CalendarEvent = self.env['calendar.event']
        self.event_tech_presentation = self.CalendarEvent.create({'privacy': 'private', 'start': '2011-04-30 16:00:00', 'stop': '2011-04-30 18:30:00', 'description': 'The Technical Presentation will cover following topics:\n* Creating Odoo class\n* Views\n* Wizards\n* Workflows', 'duration': 2.5, 'location': 'Odoo S.A.', 'name': 'Technical Presentation'})

    def test_calender_event(self):
        if False:
            return 10
        data = {'fr': 1, 'mo': 1, 'interval': 1, 'rrule_type': 'weekly', 'end_type': 'end_date', 'final_date': '2011-05-31 00:00:00', 'recurrency': True}
        self.event_tech_presentation.write(data)
        self.CalendarEvent.fields_view_get(False, 'calendar')
        rec_events = self.CalendarEvent.with_context({'virtual_id': True}).search([('start', '>=', '2011-04-30 16:00:00'), ('start', '<=', '2011-05-31 00:00:00')])
        self.assertEqual(len(rec_events), 9, 'Wrong number of events found')
        before = self.CalendarEvent.with_context({'virtual_id': False}).search([('start', '>=', '2011-04-30 16:00:00'), ('start', '<=', '2011-05-31 00:00:00')])
        newevent = rec_events[1].detach_recurring_event()
        newevent.with_context({'virtual_id': True}).write({'name': 'New Name', 'recurrency': True})
        after = self.CalendarEvent.with_context({'virtual_id': False}).search([('start', '>=', '2011-04-30 16:00:00'), ('start', '<=', '2011-05-31 00:00:00')])
        self.assertEqual(len(after), len(before) + 1, 'Wrong number of events found, after to have moved a virtual event')
        new_event = after - before
        self.assertEqual(new_event[0].recurrent_id, before.id, 'Recurrent_id not correctly passed to the new event')
        allday_event = self.CalendarEvent.create({'allday': 1, 'privacy': 'confidential', 'start': '2011-04-30 00:00:00', 'stop': '2011-04-30 00:00:00', 'description': 'All day technical test', 'location': 'School', 'name': 'All day test event'})
        res_alarm_day_before_event_starts = self.env['calendar.alarm'].create({'name': '1 Day before event starts', 'duration': 1, 'interval': 'days', 'type': 'notification'})
        allday_event.write({'alarm_ids': [(6, 0, [res_alarm_day_before_event_starts.id])]})
        calendar_event_sprint_review = self.CalendarEvent.create({'name': 'Begin of month meeting', 'start': fields.Date.today() + ' 12:00:00', 'stop': fields.Date.today() + ' 18:00:00', 'recurrency': True, 'rrule': 'FREQ=MONTHLY;INTERVAL=1;COUNT=12;BYDAY=1MO'})
        self.assertEqual(calendar_event_sprint_review.rrule_type, 'monthly', 'rrule_type should be mothly')
        self.assertEqual(calendar_event_sprint_review.count, 12, 'rrule_type should be mothly')
        self.assertEqual(calendar_event_sprint_review.month_by, 'day', 'rrule_type should be mothly')
        self.assertEqual(calendar_event_sprint_review.byday, '1', 'rrule_type should be mothly')
        self.assertEqual(calendar_event_sprint_review.week_list, 'MO', 'rrule_type should be mothly')