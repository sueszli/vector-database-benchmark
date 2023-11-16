from chalicelib.core import log_tools
from schemas import schemas
IN_TY = 'rollbar'

def get_all(tenant_id):
    if False:
        for i in range(10):
            print('nop')
    return log_tools.get_all_by_tenant(tenant_id=tenant_id, integration=IN_TY)

def get(project_id):
    if False:
        while True:
            i = 10
    return log_tools.get(project_id=project_id, integration=IN_TY)

def update(tenant_id, project_id, changes):
    if False:
        i = 10
        return i + 15
    options = {}
    if 'accessToken' in changes:
        options['accessToken'] = changes['accessToken']
    return log_tools.edit(project_id=project_id, integration=IN_TY, changes=options)

def add(tenant_id, project_id, access_token):
    if False:
        while True:
            i = 10
    options = {'accessToken': access_token}
    return log_tools.add(project_id=project_id, integration=IN_TY, options=options)

def delete(tenant_id, project_id):
    if False:
        i = 10
        return i + 15
    return log_tools.delete(project_id=project_id, integration=IN_TY)

def add_edit(tenant_id, project_id, data: schemas.IntegrationRollbarSchema):
    if False:
        print('Hello World!')
    s = get(project_id)
    if s is not None:
        return update(tenant_id=tenant_id, project_id=project_id, changes={'accessToken': data.access_token})
    else:
        return add(tenant_id=tenant_id, project_id=project_id, access_token=data.access_token)