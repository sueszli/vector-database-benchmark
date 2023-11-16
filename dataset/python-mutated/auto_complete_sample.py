import os
import time
from googleapiclient.discovery import build
client_service = build('jobs', 'v3')
name = 'projects/' + os.environ['GOOGLE_CLOUD_PROJECT']

def job_title_auto_complete(client_service, query, company_name):
    if False:
        while True:
            i = 10
    complete = client_service.projects().complete(name=name, query=query, languageCode='en-US', type='JOB_TITLE', pageSize=10)
    if company_name is not None:
        complete.companyName = company_name
    results = complete.execute()
    print(results)

def auto_complete_default(client_service, query, company_name):
    if False:
        i = 10
        return i + 15
    complete = client_service.projects().complete(name=name, query=query, languageCode='en-US', pageSize=10)
    if company_name is not None:
        complete.companyName = company_name
    results = complete.execute()
    print(results)

def set_up():
    if False:
        print('Hello World!')
    import base_company_sample
    import base_job_sample
    company_to_be_created = base_company_sample.generate_company()
    company_created = base_company_sample.create_company(client_service, company_to_be_created)
    company_name = company_created.get('name')
    job_to_be_created = base_job_sample.generate_job_with_required_fields(company_name)
    job_to_be_created.update({'title': 'Software engineer'})
    job_name = base_job_sample.create_job(client_service, job_to_be_created).get('name')
    return (company_name, job_name)

def tear_down(company_name, job_name):
    if False:
        for i in range(10):
            print('nop')
    import base_company_sample
    import base_job_sample
    base_job_sample.delete_job(client_service, job_name)
    base_company_sample.delete_company(client_service, company_name)

def run_sample(company_name):
    if False:
        for i in range(10):
            print('nop')
    auto_complete_default(client_service, 'goo', company_name)
    auto_complete_default(client_service, 'sof', company_name)
    job_title_auto_complete(client_service, 'sof', company_name)
if __name__ == '__main__':
    (company_name, job_name) = set_up()
    time.sleep(10)
    run_sample(company_name)
    tear_down(company_name, job_name)