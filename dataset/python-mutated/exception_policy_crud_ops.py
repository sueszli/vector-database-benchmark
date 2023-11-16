"""
FILE: exception_policy_crud_ops.py
DESCRIPTION:
    These samples demonstrates how to create Exception Policy used in ACS JobRouter.
    You need a valid connection string to an Azure Communication Service to execute the sample

USAGE:
    python exception_policy_crud_ops.py
    Set the environment variables with your own values before running the sample:
    1) AZURE_COMMUNICATION_SERVICE_ENDPOINT - Communication Service endpoint url
"""
import os

class ExceptionPolicySamples(object):
    endpoint = os.environ['AZURE_COMMUNICATION_SERVICE_ENDPOINT']
    _ep_policy_id = 'sample_ep_policy'
    _cp_policy_ids = ['escalation-on-q-over-flow', 'escalation-on-wait-time-exceeded']

    def setup(self):
        if False:
            return 10
        connection_string = self.endpoint
        from azure.communication.jobrouter import JobRouterAdministrationClient
        from azure.communication.jobrouter.models import ClassificationPolicy, StaticRouterRule, ExpressionRouterRule, StaticQueueSelectorAttachment, ConditionalQueueSelectorAttachment, RouterQueueSelector, StaticWorkerSelectorAttachment, RouterWorkerSelector, LabelOperator
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        for _id in self._cp_policy_ids:
            classification_policy: ClassificationPolicy = router_admin_client.upsert_classification_policy(_id, ClassificationPolicy(prioritization_rule=StaticRouterRule(value=100), queue_selector_attachments=[StaticQueueSelectorAttachment(queue_selector=RouterQueueSelector(key='Escalate', label_operator=LabelOperator.EQUAL, value=True))], worker_selector_attachments=[StaticWorkerSelectorAttachment(worker_selector=RouterWorkerSelector(key='Escalate', label_operator=LabelOperator.EQUAL, value=True))]))

    def create_exception_policy(self):
        if False:
            i = 10
            return i + 15
        connection_string = self.endpoint
        policy_id = self._ep_policy_id
        from azure.communication.jobrouter import JobRouterAdministrationClient
        from azure.communication.jobrouter.models import WaitTimeExceptionTrigger, QueueLengthExceptionTrigger, ReclassifyExceptionAction, ExceptionRule, ExceptionPolicy
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        print('JobRouterAdministrationClient created successfully!')
        queue_length_exception_trigger: QueueLengthExceptionTrigger = QueueLengthExceptionTrigger(threshold=10)
        escalate_job_on_queue_over_flow: ReclassifyExceptionAction = ReclassifyExceptionAction(classification_policy_id='escalation-on-q-over-flow', labels_to_upsert={'EscalateJob': True, 'EscalationReasonCode': 'QueueOverFlow'})
        wait_time_exception_trigger: WaitTimeExceptionTrigger = WaitTimeExceptionTrigger(threshold_seconds=10 * 60)
        escalate_job_on_wait_time_exceeded: ReclassifyExceptionAction = ReclassifyExceptionAction(classification_policy_id='escalation-on-wait-time-exceeded', labels_to_upsert={'EscalateJob': True, 'EscalationReasonCode': 'WaitTimeExceeded'})
        exception_rules = [ExceptionRule(id='EscalateJobOnQueueOverFlowTrigger', trigger=queue_length_exception_trigger, actions=[escalate_job_on_queue_over_flow]), ExceptionRule(id='EscalateJobOnWaitTimeExceededTrigger', trigger=wait_time_exception_trigger, actions=[escalate_job_on_wait_time_exceeded])]
        exception_policy = router_admin_client.upsert_exception_policy(policy_id, ExceptionPolicy(name='TriggerJobCancellationWhenQueueLenIs10', exception_rules=exception_rules))
        print(f'Exception policy has been successfully created with id: {exception_policy.id}')

    def update_exception_policy(self):
        if False:
            i = 10
            return i + 15
        connection_string = self.endpoint
        policy_id = self._ep_policy_id
        from azure.communication.jobrouter import JobRouterAdministrationClient
        from azure.communication.jobrouter.models import WaitTimeExceptionTrigger, ReclassifyExceptionAction, ExceptionPolicy, ExceptionRule, QueueLengthExceptionTrigger
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        print('JobRouterAdministrationClient created successfully!')
        escalate_job_on_wait_time_exceed2: WaitTimeExceptionTrigger = WaitTimeExceptionTrigger(threshold_seconds=2 * 60)
        escalate_job_on_wait_time_exceeded2: ReclassifyExceptionAction = ReclassifyExceptionAction(classification_policy_id='escalation-on-wait-time-exceeded', labels_to_upsert={'EscalateJob': True, 'EscalationReasonCode': 'WaitTimeExceeded2Min'})
        updated_exception_policy: ExceptionPolicy = router_admin_client.upsert_exception_policy(policy_id, exception_rules=[ExceptionRule(id='EscalateJobOnWaitTimeExceededTrigger2Min', trigger=escalate_job_on_wait_time_exceed2, actions=[escalate_job_on_wait_time_exceeded2]), ExceptionRule(id='EscalateJobOnQueueOverFlowTrigger', trigger=QueueLengthExceptionTrigger(threshold=100), actions=[ReclassifyExceptionAction(classification_policy_id='escalation-on-q-over-flow', labels_to_upsert={'EscalateJob': True, 'EscalationReasonCode': 'QueueOverFlow'})])])
        print(f'Exception policy updated with rules: {updated_exception_policy.exception_rules}')
        print('Exception policy has been successfully updated')

    def get_exception_policy(self):
        if False:
            print('Hello World!')
        connection_string = self.endpoint
        policy_id = self._ep_policy_id
        from azure.communication.jobrouter import JobRouterAdministrationClient
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        exception_policy = router_admin_client.get_exception_policy(policy_id)
        print(f'Successfully fetched exception policy with id: {exception_policy.id}')

    def list_exception_policies(self):
        if False:
            print('Hello World!')
        connection_string = self.endpoint
        from azure.communication.jobrouter import JobRouterAdministrationClient
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        exception_policy_iterator = router_admin_client.list_exception_policies()
        for ep in exception_policy_iterator:
            print(f'Retrieved exception policy with id: {ep.id}')
        print(f'Successfully completed fetching exception policies')

    def list_exception_policies_batched(self):
        if False:
            return 10
        connection_string = self.endpoint
        from azure.communication.jobrouter import JobRouterAdministrationClient
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        exception_policy_iterator = router_admin_client.list_exception_policies(results_per_page=10)
        for policy_page in exception_policy_iterator.by_page():
            policies_in_page = list(policy_page)
            print(f'Retrieved {len(policies_in_page)} policies in current page')
            for ep in policies_in_page:
                print(f'Retrieved exception policy with id: {ep.id}')
        print(f'Successfully completed fetching exception policies')

    def clean_up(self):
        if False:
            i = 10
            return i + 15
        connection_string = self.endpoint
        policy_id = self._ep_policy_id
        from azure.communication.jobrouter import JobRouterAdministrationClient
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        router_admin_client.delete_exception_policy(policy_id)
        for _id in self._cp_policy_ids:
            router_admin_client.delete_classification_policy(_id)
if __name__ == '__main__':
    sample = ExceptionPolicySamples()
    sample.setup()
    sample.create_exception_policy()
    sample.update_exception_policy()
    sample.get_exception_policy()
    sample.list_exception_policies()
    sample.list_exception_policies_batched()
    sample.clean_up()