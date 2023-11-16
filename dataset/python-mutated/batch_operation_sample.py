import os
from googleapiclient.discovery import build
client_service = build('jobs', 'v3')
parent = 'projects/' + os.environ['GOOGLE_CLOUD_PROJECT']

def batch_job_create(client_service, company_name):
    if False:
        for i in range(10):
            print('nop')
    import base_job_sample
    created_jobs = []

    def job_create_callback(request_id, response, exception):
        if False:
            return 10
        if exception is not None:
            print('Got exception while creating job: %s' % exception)
        else:
            print('Job created: %s' % response)
            created_jobs.append(response)
    batch = client_service.new_batch_http_request()
    job_to_be_created1 = base_job_sample.generate_job_with_required_fields(company_name)
    request1 = {'job': job_to_be_created1}
    batch.add(client_service.projects().jobs().create(parent=parent, body=request1), callback=job_create_callback)
    job_to_be_created2 = base_job_sample.generate_job_with_required_fields(company_name)
    request2 = {'job': job_to_be_created2}
    batch.add(client_service.projects().jobs().create(parent=parent, body=request2), callback=job_create_callback)
    batch.execute()
    return created_jobs

def batch_job_update(client_service, jobs_to_be_updated):
    if False:
        for i in range(10):
            print('nop')
    updated_jobs = []

    def job_update_callback(request_id, response, exception):
        if False:
            while True:
                i = 10
        if exception is not None:
            print('Got exception while updating job: %s' % exception)
        else:
            print('Job updated: %s' % response)
            updated_jobs.append(response)
    batch = client_service.new_batch_http_request()
    for index in range(0, len(jobs_to_be_updated)):
        job_to_be_updated = jobs_to_be_updated[index]
        job_to_be_updated.update({'title': 'Engineer in Mountain View'})
        request = {'job': job_to_be_updated}
        if index % 2 == 0:
            batch.add(client_service.projects().jobs().patch(name=job_to_be_updated.get('name'), body=request), callback=job_update_callback)
        else:
            request.update({'update_mask': 'title'})
            batch.add(client_service.projects().jobs().patch(name=job_to_be_updated.get('name'), body=request), callback=job_update_callback)
    batch.execute()
    return updated_jobs

def batch_job_delete(client_service, jobs_to_be_deleted):
    if False:
        while True:
            i = 10

    def job_delete_callback(request_id, response, exception):
        if False:
            i = 10
            return i + 15
        if exception is not None:
            print('Got exception while deleting job: %s' % exception)
        else:
            print('Job deleted')
    batch = client_service.new_batch_http_request()
    for job_to_be_deleted in jobs_to_be_deleted:
        batch.add(client_service.projects().jobs().delete(name=job_to_be_deleted.get('name')), callback=job_delete_callback)
    batch.execute()

def run_sample():
    if False:
        i = 10
        return i + 15
    import base_company_sample
    company_to_be_created = base_company_sample.generate_company()
    company_created = base_company_sample.create_company(client_service, company_to_be_created)
    company_name = company_created.get('name')
    created_jobs = batch_job_create(client_service, company_name)
    updated_jobs = batch_job_update(client_service, created_jobs)
    batch_job_delete(client_service, updated_jobs)
    base_company_sample.delete_company(client_service, company_name)
if __name__ == '__main__':
    run_sample()