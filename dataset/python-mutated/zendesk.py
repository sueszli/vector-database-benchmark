from __future__ import annotations
from typing import TYPE_CHECKING
from zenpy import Zenpy
from airflow.hooks.base import BaseHook
if TYPE_CHECKING:
    from zenpy.lib.api import BaseApi
    from zenpy.lib.api_objects import JobStatus, Ticket, TicketAudit
    from zenpy.lib.generator import SearchResultGenerator

class ZendeskHook(BaseHook):
    """
    Interact with Zendesk. This hook uses the Zendesk conn_id.

    :param zendesk_conn_id: The Airflow connection used for Zendesk credentials.
    """
    conn_name_attr = 'zendesk_conn_id'
    default_conn_name = 'zendesk_default'
    conn_type = 'zendesk'
    hook_name = 'Zendesk'

    def __init__(self, zendesk_conn_id: str=default_conn_name) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.zendesk_conn_id = zendesk_conn_id
        self.base_api: BaseApi | None = None
        (zenpy_client, url) = self._init_conn()
        self.zenpy_client = zenpy_client
        self.__url = url
        self.get = self.zenpy_client.users._get

    def _init_conn(self) -> tuple[Zenpy, str]:
        if False:
            while True:
                i = 10
        '\n        Create the Zenpy Client for our Zendesk connection.\n\n        :return: zenpy.Zenpy client and the url for the API.\n        '
        conn = self.get_connection(self.zendesk_conn_id)
        url = 'https://' + conn.host
        domain = conn.host
        subdomain: str | None = None
        if conn.host.count('.') >= 2:
            dot_splitted_string = conn.host.rsplit('.', 2)
            subdomain = dot_splitted_string[0]
            domain = '.'.join(dot_splitted_string[1:])
        return (Zenpy(domain=domain, subdomain=subdomain, email=conn.login, password=conn.password), url)

    def get_conn(self) -> Zenpy:
        if False:
            print('Hello World!')
        '\n        Get the underlying Zenpy client.\n\n        :return: zenpy.Zenpy client.\n        '
        return self.zenpy_client

    def get_ticket(self, ticket_id: int) -> Ticket:
        if False:
            return 10
        '\n        Retrieve ticket.\n\n        :return: Ticket object retrieved.\n        '
        return self.zenpy_client.tickets(id=ticket_id)

    def search_tickets(self, **kwargs) -> SearchResultGenerator:
        if False:
            print('Hello World!')
        '\n        Search tickets.\n\n        :param kwargs: (optional) Search fields given to the zenpy search method.\n        :return: SearchResultGenerator of Ticket objects.\n        '
        return self.zenpy_client.search(type='ticket', **kwargs)

    def create_tickets(self, tickets: Ticket | list[Ticket], **kwargs) -> TicketAudit | JobStatus:
        if False:
            print('Hello World!')
        '\n        Create tickets.\n\n        :param tickets: Ticket or List of Ticket to create.\n        :param kwargs: (optional) Additional fields given to the zenpy create method.\n        :return: A TicketAudit object containing information about the Ticket created.\n            When sending bulk request, returns a JobStatus object.\n        '
        return self.zenpy_client.tickets.create(tickets, **kwargs)

    def update_tickets(self, tickets: Ticket | list[Ticket], **kwargs) -> TicketAudit | JobStatus:
        if False:
            i = 10
            return i + 15
        '\n        Update tickets.\n\n        :param tickets: Updated Ticket or List of Ticket object to update.\n        :param kwargs: (optional) Additional fields given to the zenpy update method.\n        :return: A TicketAudit object containing information about the Ticket updated.\n            When sending bulk request, returns a JobStatus object.\n        '
        return self.zenpy_client.tickets.update(tickets, **kwargs)

    def delete_tickets(self, tickets: Ticket | list[Ticket], **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Delete tickets, returns nothing on success and raises APIException on failure.\n\n        :param tickets: Ticket or List of Ticket to delete.\n        :param kwargs: (optional) Additional fields given to the zenpy delete method.\n        :return:\n        '
        return self.zenpy_client.tickets.delete(tickets, **kwargs)