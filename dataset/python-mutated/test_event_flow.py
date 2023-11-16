import datetime
from dateutil.relativedelta import relativedelta
from odoo.addons.event.tests.common import TestEventCommon
from odoo.exceptions import ValidationError, UserError, AccessError
from odoo.tools import mute_logger
from odoo.fields import Datetime
from mock import patch

class TestEventFlow(TestEventCommon):

    @mute_logger('odoo.addons.base.ir.ir_model', 'odoo.models')
    def test_00_basic_event_auto_confirm(self):
        if False:
            return 10
        ' Basic event management with auto confirmation '
        event_config = self.env['event.config.settings'].sudo(self.user_eventmanager).create({'auto_confirmation': 1})
        event_config.execute()
        test_event = self.Event.sudo(self.user_eventmanager).create({'name': 'TestEvent', 'date_begin': datetime.datetime.now() + relativedelta(days=-1), 'date_end': datetime.datetime.now() + relativedelta(days=1), 'seats_max': 2, 'seats_availability': 'limited'})
        self.assertEqual(test_event.state, 'confirm', 'Event: auto_confirmation of event failed')
        test_reg1 = self.Registration.sudo(self.user_eventuser).create({'name': 'TestReg1', 'event_id': test_event.id})
        self.assertEqual(test_reg1.state, 'open', 'Event: auto_confirmation of registration failed')
        self.assertEqual(test_event.seats_reserved, 1, 'Event: wrong number of reserved seats after confirmed registration')
        test_reg2 = self.Registration.sudo(self.user_eventuser).create({'name': 'TestReg2', 'event_id': test_event.id})
        self.assertEqual(test_reg2.state, 'open', 'Event: auto_confirmation of registration failed')
        self.assertEqual(test_event.seats_reserved, 2, 'Event: wrong number of reserved seats after confirmed registration')
        with self.assertRaises(ValidationError):
            self.Registration.sudo(self.user_eventuser).create({'name': 'TestReg3', 'event_id': test_event.id})
        test_reg1.button_reg_close()
        self.assertEqual(test_reg1.state, 'done', 'Event: wrong state of attended registration')
        self.assertEqual(test_event.seats_used, 1, 'Event: incorrect number of attendees after closing registration')
        test_reg2.button_reg_close()
        self.assertEqual(test_reg1.state, 'done', 'Event: wrong state of attended registration')
        self.assertEqual(test_event.seats_used, 2, 'Event: incorrect number of attendees after closing registration')
        test_event.button_done()
        with self.assertRaises(UserError):
            test_event.button_cancel()

    @mute_logger('odoo.addons.base.ir.ir_model', 'odoo.models')
    def test_10_advanced_event_flow(self):
        if False:
            while True:
                i = 10
        ' Avanced event flow: no auto confirmation, manage minimum / maximum\n        seats, ... '
        self.env['ir.values'].set_default('event.config.settings', 'auto_confirmation', False)
        test_event = self.Event.sudo(self.user_eventmanager).create({'name': 'TestEvent', 'date_begin': datetime.datetime.now() + relativedelta(days=-1), 'date_end': datetime.datetime.now() + relativedelta(days=1), 'seats_max': 10})
        self.assertEqual(test_event.state, 'draft', 'Event: new event should be in draft state, no auto confirmation')
        test_reg1 = self.Registration.sudo(self.user_eventuser).create({'name': 'TestReg1', 'event_id': test_event.id})
        self.assertEqual(test_reg1.state, 'draft', 'Event: new registration should not be confirmed with auto_confirmation parameter being False')

    def test_event_access_rights(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AccessError):
            self.Event.sudo(self.user_eventuser).create({'name': 'TestEvent', 'date_begin': datetime.datetime.now() + relativedelta(days=-1), 'date_end': datetime.datetime.now() + relativedelta(days=1), 'seats_max': 10})
        with self.assertRaises(AccessError):
            self.event_0.sudo(self.user_eventuser).write({'name': 'TestEvent Modified'})
        self.user_eventmanager.write({'groups_id': [(3, self.env.ref('base.group_system').id), (4, self.env.ref('base.group_erp_manager').id)]})
        with self.assertRaises(AccessError):
            event_config = self.env['event.config.settings'].sudo(self.user_eventmanager).create({'auto_confirmation': 1})
            event_config.execute()

    def test_event_data(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.event_0.registration_ids.get_date_range_str(), u'Tomorrow')

    def test_event_date_range(self):
        if False:
            print('Hello World!')
        self.patcher = patch('odoo.addons.event.models.event.fields.Datetime', wraps=Datetime)
        self.mock_datetime = self.patcher.start()
        self.mock_datetime.now.return_value = Datetime.to_string(datetime.datetime(2015, 12, 31, 12, 0))
        self.event_0.registration_ids.event_begin_date = datetime.datetime(2015, 12, 31, 18, 0)
        self.assertEqual(self.event_0.registration_ids.get_date_range_str(), u'Today')
        self.event_0.registration_ids.event_begin_date = datetime.datetime(2016, 1, 1, 6, 0)
        self.assertEqual(self.event_0.registration_ids.get_date_range_str(), u'Tomorrow')
        self.event_0.registration_ids.event_begin_date = datetime.datetime(2016, 1, 2, 6, 0)
        self.assertEqual(self.event_0.registration_ids.get_date_range_str(), u'This week')
        self.event_0.registration_ids.event_begin_date = datetime.datetime(2016, 2, 1, 6, 0)
        self.assertTrue('T' in self.event_0.registration_ids.get_date_range_str())
        self.mock_datetime.now.return_value = Datetime.to_string(datetime.datetime(2015, 12, 15, 12, 0))
        self.event_0.registration_ids.event_begin_date = datetime.datetime(2015, 12, 31, 6, 0)
        self.assertEqual(self.event_0.registration_ids.get_date_range_str(), u'This month')
        self.patcher.stop()