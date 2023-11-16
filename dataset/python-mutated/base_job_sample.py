import os
import random
import string
from googleapiclient.discovery import build
from googleapiclient.errors import Error
client_service = build('jobs', 'v3')
parent = 'projects/' + os.environ['GOOGLE_CLOUD_PROJECT']

def generate_job_with_required_fields(company_name):
    if False:
        print('Hello World!')
    requisition_id = 'job_with_required_fields:' + ''.join((random.choice(string.ascii_uppercase + string.digits) for _ in range(16)))
    job_title = 'Software Engineer'
    application_uris = ['http://careers.google.com']
    description = 'Design, develop, test, deploy, maintain and improve software.'
    job = {'requisition_id': requisition_id, 'title': job_title, 'application_info': {'uris': application_uris}, 'description': description, 'company_name': company_name}
    print('Job generated: %s' % job)
    return job

def create_job(client_service, job_to_be_created):
    if False:
        print('Hello World!')
    try:
        request = {'job': job_to_be_created}
        job_created = client_service.projects().jobs().create(parent=parent, body=request).execute()
        print('Job created: %s' % job_created)
        return job_created
    except Error as e:
        print('Got exception while creating job')
        raise e

def get_job(client_service, job_name):
    if False:
        for i in range(10):
            print('nop')
    try:
        job_existed = client_service.projects().jobs().get(name=job_name).execute()
        print('Job existed: %s' % job_existed)
        return job_existed
    except Error as e:
        print('Got exception while getting job')
        raise e

def update_job(client_service, job_name, job_to_be_updated):
    if False:
        for i in range(10):
            print('nop')
    try:
        request = {'job': job_to_be_updated}
        job_updated = client_service.projects().jobs().patch(name=job_name, body=request).execute()
        print('Job updated: %s' % job_updated)
        return job_updated
    except Error as e:
        print('Got exception while updating job')
        raise e

def update_job_with_field_mask(client_service, job_name, job_to_be_updated, field_mask):
    if False:
        return 10
    try:
        request = {'job': job_to_be_updated, 'update_mask': field_mask}
        job_updated = client_service.projects().jobs().patch(name=job_name, body=request).execute()
        print('Job updated: %s' % job_updated)
        return job_updated
    except Error as e:
        print('Got exception while updating job with field mask')
        raise e

def delete_job(client_service, job_name):
    if False:
        print('Hello World!')
    try:
        client_service.projects().jobs().delete(name=job_name).execute()
        print('Job deleted')
    except Error as e:
        print('Got exception while deleting job')
        raise e

def run_sample():
    if False:
        while True:
            i = 10
    import base_company_sample
    company_to_be_created = base_company_sample.generate_company()
    company_created = base_company_sample.create_company(client_service, company_to_be_created)
    company_name = company_created.get('name')
    job_to_be_created = generate_job_with_required_fields(company_name)
    job_created = create_job(client_service, job_to_be_created)
    job_name = job_created.get('name')
    get_job(client_service, job_name)
    job_to_be_updated = job_created
    job_to_be_updated.update({'description': 'changedDescription'})
    update_job(client_service, job_name, job_to_be_updated)
    update_job_with_field_mask(client_service, job_name, {'title': 'changedJobTitle'}, 'title')
    delete_job(client_service, job_name)
    base_company_sample.delete_company(client_service, company_name)
if __name__ == '__main__':
    run_sample()