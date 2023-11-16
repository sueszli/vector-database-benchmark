"""
FILE: job_queue_crud_ops.py
DESCRIPTION:
    These samples demonstrates how to create Job Queues used in ACS JobRouter.
    You need a valid connection string to an Azure Communication Service to execute the sample

USAGE:
    python job_queue_crud_ops.py
    Set the environment variables with your own values before running the sample:
    1) AZURE_COMMUNICATION_SERVICE_ENDPOINT - Communication Service endpoint url
"""
import os

class JobQueueSamples(object):
    endpoint = os.environ['AZURE_COMMUNICATION_SERVICE_ENDPOINT']
    _job_queue_id = 'sample_q_policy'
    _distribution_policy_id = 'sample_dp_policy'

    def setup_distribution_policy(self):
        if False:
            i = 10
            return i + 15
        connection_string = self.endpoint
        distribution_policy_id = self._distribution_policy_id
        from azure.communication.jobrouter import JobRouterAdministrationClient
        from azure.communication.jobrouter.models import LongestIdleMode, DistributionPolicy
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        distribution_policy = router_admin_client.upsert_distribution_policy(distribution_policy_id, DistributionPolicy(offer_expires_after_seconds=10 * 60, mode=LongestIdleMode(min_concurrent_offers=1, max_concurrent_offers=1)))
        print(f'Sample setup completed: Created distribution policy')

    def create_queue(self):
        if False:
            i = 10
            return i + 15
        connection_string = self.endpoint
        job_queue_id = self._job_queue_id
        distribution_policy_id = self._distribution_policy_id
        from azure.communication.jobrouter import JobRouterAdministrationClient
        from azure.communication.jobrouter.models import RouterQueue
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        print('JobRouterAdministrationClient created successfully!')
        job_queue: RouterQueue = router_admin_client.upsert_queue(job_queue_id, RouterQueue(distribution_policy_id=distribution_policy_id, name='My job queue'))
        print(f'Job queue successfully created with id: {job_queue.id}')

    def update_queue(self):
        if False:
            i = 10
            return i + 15
        connection_string = self.endpoint
        job_queue_id = self._job_queue_id
        from azure.communication.jobrouter import JobRouterAdministrationClient
        from azure.communication.jobrouter.models import RouterQueue
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        print('JobRouterAdministrationClient created successfully!')
        updated_job_queue: RouterQueue = router_admin_client.upsert_queue(job_queue_id, labels={'Additional-Queue-Label': 'ChatQueue'})
        print(f'Router queue successfully update with labels {updated_job_queue.labels}')

    def get_queue(self):
        if False:
            i = 10
            return i + 15
        connection_string = self.endpoint
        job_queue_id = self._job_queue_id
        from azure.communication.jobrouter import JobRouterAdministrationClient
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        job_queue = router_admin_client.get_queue(job_queue_id)
        print(f'Successfully fetched router queue with id: {job_queue.id}')

    def get_queue_statistics(self):
        if False:
            i = 10
            return i + 15
        connection_string = self.endpoint
        job_queue_id = self._job_queue_id
        from azure.communication.jobrouter import JobRouterClient
        from azure.communication.jobrouter.models import RouterQueueStatistics
        router_client: JobRouterClient = JobRouterClient.from_connection_string(conn_str=connection_string)
        job_queue_statistics: RouterQueueStatistics = router_client.get_queue_statistics(job_queue_id)
        print(f'Successfully fetched queue statistics router queue: {job_queue_statistics}')

    def list_queues(self):
        if False:
            return 10
        connection_string = self.endpoint
        from azure.communication.jobrouter import JobRouterAdministrationClient
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        job_queue_iterator = router_admin_client.list_queues()
        for q in job_queue_iterator:
            print(f'Retrieved queue policy with id: {q.id}')
        print(f'Successfully completed fetching job queues')

    def list_queues_batched(self):
        if False:
            for i in range(10):
                print('nop')
        connection_string = self.endpoint
        from azure.communication.jobrouter import JobRouterAdministrationClient
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        job_queue_iterator = router_admin_client.list_queues(results_per_page=10)
        for queue_page in job_queue_iterator.by_page():
            job_queues_in_page = list(queue_page)
            print(f'Retrieved {len(job_queues_in_page)} queues in current page')
            for q in job_queues_in_page:
                print(f'Retrieved queue policy with id: {q.id}')
        print(f'Successfully completed fetching job queues')

    def clean_up(self):
        if False:
            print('Hello World!')
        connection_string = self.endpoint
        job_queue_id = self._job_queue_id
        from azure.communication.jobrouter import JobRouterAdministrationClient
        router_admin_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        router_admin_client.delete_queue(job_queue_id)
        router_admin_client.delete_distribution_policy(self._distribution_policy_id)
if __name__ == '__main__':
    sample = JobQueueSamples()
    sample.setup_distribution_policy()
    sample.create_queue()
    sample.update_queue()
    sample.get_queue()
    sample.get_queue_statistics()
    sample.list_queues()
    sample.list_queues_batched()
    sample.clean_up()