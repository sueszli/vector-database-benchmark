from pylons import app_globals as g

class SupportTicketError(Exception):
    pass

class SupportTickerNotFoundError(SupportTicketError):
    pass

def create_support_ticket(subject, comment_body, comment_is_public=False, group=None, requester_email=None, product=None):
    if False:
        for i in range(10):
            print('nop')
    requester_id = None
    if requester_email == 'contact@reddit.com':
        requester_id = g.live_config['ticket_contact_user_id']
    custom_fields = []
    if product:
        custom_fields.append({'id': g.live_config['ticket_user_fields']['Product'], 'value': product})
    return g.ticket_provider.create(requester_id=requester_id, subject=subject, comment_body=comment_body, comment_is_public=comment_is_public, group_id=g.live_config['ticket_groups'][group], custom_fields=custom_fields)

def get_support_ticket(ticket_id):
    if False:
        for i in range(10):
            print('nop')
    return g.ticket_provider.get(ticket_id)

def get_support_ticket_url(ticket_id):
    if False:
        while True:
            i = 10
    return g.ticket_provider.build_ticket_url_from_id(ticket_id)

def update_support_ticket(ticket=None, ticket_id=None, status=None, comment_body=None, comment_is_public=False, tag_list=None):
    if False:
        print('Hello World!')
    if not ticket and (not ticket_id):
        raise SupportTickerNotFoundError('No ticket provided to update.')
    if not ticket:
        ticket = get_support_ticket(ticket_id)
    return g.ticket_provider.update(ticket=ticket, status=status, comment_body=comment_body, comment_is_public=comment_is_public, tag_list=tag_list)