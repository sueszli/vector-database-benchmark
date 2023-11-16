import calendar
import datetime
import logging
import random
from golem_messages import constants
from golem_messages import factories as msg_factories
from golem_messages import helpers
from golem_messages import message
from golem_messages.factories.helpers import fake_golem_uuid
from golem.network.concent import exceptions as concent_exceptions
from ..base import SCIBaseTest
reasons = message.concents.ForceSubtaskResultsRejected.REASON
logger = logging.getLogger(__name__)
moment = datetime.timedelta(seconds=2)

class TestBase(SCIBaseTest):

    def prepare_report_computed_task(self, mode, ttc_kwargs, rct_kwargs, ttc=None):
        if False:
            i = 10
            return i + 15
        'Returns ReportComputedTask with open force acceptance window\n\n        Can be modified by delta\n        '
        _rct_kwargs = self.gen_rtc_kwargs()
        _rct_kwargs.update(rct_kwargs)
        report_computed_task = msg_factories.tasks.ReportComputedTaskFactory(task_to_compute=ttc if ttc else self.gen_ttc(**ttc_kwargs), **_rct_kwargs)
        deadline_delta = 3600
        deadline_timedelta = datetime.timedelta(seconds=deadline_delta)
        report_computed_task.task_to_compute.compute_task_def['deadline'] = report_computed_task.task_to_compute.timestamp + deadline_delta
        svt = helpers.subtask_verification_time(report_computed_task)
        now = datetime.datetime.utcnow()
        if mode == 'before':
            ttc_dt = now - deadline_timedelta - svt + moment
        elif mode == 'after':
            ttc_dt = now - deadline_timedelta - svt - constants.FAT - moment
        else:
            ttc_dt = now - deadline_timedelta - svt - moment
        ttc_timestamp = calendar.timegm(ttc_dt.utctimetuple())
        msg_factories.helpers.override_timestamp(msg=report_computed_task.task_to_compute, timestamp=ttc_timestamp)
        report_computed_task.task_to_compute.compute_task_def['deadline'] = report_computed_task.task_to_compute.timestamp + deadline_delta
        report_computed_task.task_to_compute.sig = None
        report_computed_task.sig = None
        report_computed_task.task_to_compute.sign_message(self.requestor_priv_key)
        report_computed_task.sign_message(self.provider_priv_key)
        self.assertTrue(report_computed_task.verify_owners(provider_public_key=self.provider_pub_key, requestor_public_key=self.requestor_pub_key, concent_public_key=self.variant['pubkey']))
        print('*' * 80)
        print('TTC:', ttc_dt)
        print('WINDOW: {} ---------- {}'.format(ttc_dt + deadline_timedelta + svt, ttc_dt + deadline_timedelta + svt + constants.FAT))
        print('NOW:', now)
        print('*' * 80)
        return report_computed_task

    def provider_send_force(self, mode='within', ttc_kwargs=None, ttc=None, rct_kwargs=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ttc_kwargs = ttc_kwargs or {}
        rct_kwargs = rct_kwargs or {}
        if ttc:
            price = ttc.price
        else:
            price = random.randint(1 << 20, 10 << 20)
            ttc_kwargs['price'] = price
        self.requestor_put_deposit(helpers.requestor_deposit_amount(price)[0])
        report_computed_task = self.prepare_report_computed_task(mode=mode, ttc_kwargs=ttc_kwargs, rct_kwargs=rct_kwargs, ttc=ttc)
        fsr = msg_factories.concents.ForceSubtaskResultsFactory(ack_report_computed_task__report_computed_task=report_computed_task, **kwargs)
        fsr.task_to_compute.generate_ethsig(private_key=self.requestor_priv_key)
        fsr.task_to_compute.sign_message(private_key=self.requestor_priv_key)
        fsr.ack_report_computed_task.report_computed_task.sign_message(private_key=self.provider_priv_key)
        fsr.ack_report_computed_task.sign_message(private_key=self.requestor_priv_key)
        fsr.sign_message(private_key=self.provider_priv_key)
        self.assertTrue(fsr.task_to_compute.verify_ethsig())
        self.assertEqual(fsr.task_to_compute.price, price)
        self.assertTrue(fsr.validate_ownership_chain(concent_public_key=self.variant['pubkey']))
        self.assertTrue(fsr.verify_owners(provider_public_key=self.provider_keys.raw_pubkey, requestor_public_key=self.requestor_keys.raw_pubkey, concent_public_key=self.variant['pubkey']))
        fsr.sig = None
        self.provider_fsr = fsr
        response = self.provider_load_response(self.provider_send(fsr))
        self.assertIn(type(response), [type(None), message.concents.ServiceRefused, message.concents.ForceSubtaskResultsRejected])
        return response

class RequestorDoesntSendTestCase(TestBase):
    """Requestor doesn't send Ack/Reject of SubtaskResults"""

    def test_provider_insufficient_funds(self):
        if False:
            print('Hello World!')
        pass

    def test_provider_before_start(self):
        if False:
            i = 10
            return i + 15
        response = self.provider_send_force(mode='before')
        self.assertIsInstance(response, message.concents.ForceSubtaskResultsRejected)
        self.assertIs(response.reason, reasons.RequestPremature)

    def test_provider_after_deadline(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.provider_send_force(mode='after')
        self.assertIsInstance(response, message.concents.ForceSubtaskResultsRejected)
        self.assertIs(response.reason, reasons.RequestTooLate)

    def test_already_processed(self):
        if False:
            for i in range(10):
                print('nop')
        requestor_id = '1234'
        task_id = fake_golem_uuid(requestor_id)
        subtask_id = fake_golem_uuid(requestor_id)
        kwargs = {'requestor_id': requestor_id, 'task_id': task_id, 'subtask_id': subtask_id}
        self.assertIsNone(self.provider_send_force(ttc_kwargs=kwargs))
        second_response = self.provider_send_force(ttc_kwargs=kwargs)
        self.assertIsInstance(second_response, message.concents.ServiceRefused)

    def test_no_response_from_requestor(self):
        if False:
            print('Hello World!')
        pass

    def test_requestor_responds_with_invalid_accept(self):
        if False:
            while True:
                i = 10
        self.assertIsNone(self.provider_send_force())
        fsrr = msg_factories.concents.ForceSubtaskResultsResponseFactory()
        fsrr.subtask_results_rejected = None
        with self.assertRaises(concent_exceptions.ConcentRequestError):
            self.requestor_send(fsrr)

    def test_requestor_responds_with_invalid_reject(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(self.provider_send_force())
        fsrr = msg_factories.concents.ForceSubtaskResultsResponseFactory()
        fsrr.subtask_results_accepted = None
        with self.assertRaises(concent_exceptions.ConcentRequestError):
            self.requestor_send(fsrr)

    def test_requestor_responds_with_accept(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(self.provider_send_force())
        fsr = self.requestor_receive()
        self.assertTrue(fsr.verify_owners(provider_public_key=self.provider_pub_key, requestor_public_key=self.requestor_pub_key, concent_public_key=self.variant['pubkey']))
        self.assertEqual(self.provider_fsr.subtask_id, fsr.subtask_id)
        accept_msg = msg_factories.tasks.SubtaskResultsAcceptedFactory(report_computed_task=fsr.ack_report_computed_task.report_computed_task)
        accept_msg.sign_message(self.requestor_priv_key)
        self.assertTrue(accept_msg.verify_owners(provider_public_key=self.provider_pub_key, requestor_public_key=self.requestor_pub_key, concent_public_key=self.variant['pubkey']))
        fsrr = message.concents.ForceSubtaskResultsResponse(subtask_results_accepted=accept_msg)
        self.requestor_send(fsrr)
        received = self.provider_receive()
        self.assertIsInstance(received, message.concents.ForceSubtaskResultsResponse)
        self.assertIsNone(received.subtask_results_rejected)
        self.assertEqual(received.subtask_results_accepted, accept_msg)

    def test_requestor_responds_with_reject(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(self.provider_send_force())
        fsr = self.requestor_receive()
        self.assertTrue(fsr.verify_owners(provider_public_key=self.provider_pub_key, requestor_public_key=self.requestor_pub_key, concent_public_key=self.variant['pubkey']))
        reject_msg = msg_factories.tasks.SubtaskResultsRejectedFactory(report_computed_task=fsr.ack_report_computed_task.report_computed_task)
        reject_msg.sign_message(self.requestor_priv_key)
        self.assertTrue(reject_msg.verify_owners(provider_public_key=self.provider_pub_key, requestor_public_key=self.requestor_pub_key, concent_public_key=self.variant['pubkey']))
        fsrr = message.concents.ForceSubtaskResultsResponse(subtask_results_rejected=reject_msg)
        self.requestor_send(fsrr)
        received = self.provider_receive()
        self.assertIsInstance(received, message.concents.ForceSubtaskResultsResponse)
        self.assertIsNone(received.subtask_results_accepted)
        self.assertEqual(received.subtask_results_rejected, reject_msg)

    def test_requestor_responds_with_invalid_reject_reason(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(self.provider_send_force())
        fsr = self.requestor_receive()
        self.assertTrue(fsr.verify_owners(provider_public_key=self.provider_pub_key, requestor_public_key=self.requestor_pub_key, concent_public_key=self.variant['pubkey']))
        reject_msg = msg_factories.tasks.SubtaskResultsRejectedFactory(report_computed_task=fsr.ack_report_computed_task.report_computed_task, reason=message.tasks.SubtaskResultsRejected.REASON.ConcentVerificationNegative)
        reject_msg.sign_message(self.requestor_priv_key)
        self.assertTrue(reject_msg.verify_owners(provider_public_key=self.provider_pub_key, requestor_public_key=self.requestor_pub_key, concent_public_key=self.variant['pubkey']))
        fsrr = message.concents.ForceSubtaskResultsResponse(subtask_results_rejected=reject_msg)
        with self.assertRaises(concent_exceptions.ConcentRequestError):
            self.requestor_send(fsrr)