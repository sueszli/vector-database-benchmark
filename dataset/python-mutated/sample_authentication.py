"""
FILE: sample_authentication.py
DESCRIPTION:
    These samples demonstrates how to create a Router client.
    You need a valid connection string to an Azure Communication Service to execute the sample

USAGE:
    python sample_authentication.py
    Set the environment variables with your own values before running the sample:
    1) AZURE_COMMUNICATION_SERVICE_ENDPOINT - Communication Service endpoint url
"""
import os

class RouterClientAuthenticationSamples(object):
    endpoint = os.environ['AZURE_COMMUNICATION_SERVICE_ENDPOINT']

    def create_router_client(self):
        if False:
            i = 10
            return i + 15
        connection_string = self.endpoint
        from azure.communication.jobrouter import JobRouterClient
        router_client = JobRouterClient.from_connection_string(conn_str=connection_string)
        print('JobRouterClient created successfully!')

    def create_router_admin_client(self):
        if False:
            print('Hello World!')
        connection_string = self.endpoint
        from azure.communication.jobrouter import JobRouterAdministrationClient
        router_client = JobRouterAdministrationClient.from_connection_string(conn_str=connection_string)
        print('JobRouterAdministrationClient created successfully!')
if __name__ == '__main__':
    sample = RouterClientAuthenticationSamples()
    sample.create_router_client()
    sample.create_router_admin_client()