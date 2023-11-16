import os
import time
from googleapiclient.discovery import build
client_service = build('jobs', 'v3')
parent = 'projects/' + os.environ['GOOGLE_CLOUD_PROJECT']

def basic_keyword_search(client_service, company_name, keyword):
    if False:
        for i in range(10):
            print('nop')
    request_metadata = {'user_id': 'HashedUserId', 'session_id': 'HashedSessionId', 'domain': 'www.google.com'}
    job_query = {'query': keyword}
    if company_name is not None:
        job_query.update({'company_names': [company_name]})
    request = {'search_mode': 'JOB_SEARCH', 'request_metadata': request_metadata, 'job_query': job_query}
    response = client_service.projects().jobs().search(parent=parent, body=request).execute()
    print(response)

def category_search(client_service, company_name, categories):
    if False:
        i = 10
        return i + 15
    request_metadata = {'user_id': 'HashedUserId', 'session_id': 'HashedSessionId', 'domain': 'www.google.com'}
    job_query = {'job_categories': categories}
    if company_name is not None:
        job_query.update({'company_names': [company_name]})
    request = {'search_mode': 'JOB_SEARCH', 'request_metadata': request_metadata, 'job_query': job_query}
    response = client_service.projects().jobs().search(parent=parent, body=request).execute()
    print(response)

def employment_types_search(client_service, company_name, employment_types):
    if False:
        while True:
            i = 10
    request_metadata = {'user_id': 'HashedUserId', 'session_id': 'HashedSessionId', 'domain': 'www.google.com'}
    job_query = {'employment_types': employment_types}
    if company_name is not None:
        job_query.update({'company_names': [company_name]})
    request = {'search_mode': 'JOB_SEARCH', 'request_metadata': request_metadata, 'job_query': job_query}
    response = client_service.projects().jobs().search(parent=parent, body=request).execute()
    print(response)

def date_range_search(client_service, company_name, date_range):
    if False:
        i = 10
        return i + 15
    request_metadata = {'user_id': 'HashedUserId', 'session_id': 'HashedSessionId', 'domain': 'www.google.com'}
    job_query = {'publish_time_range': date_range}
    if company_name is not None:
        job_query.update({'company_names': [company_name]})
    request = {'search_mode': 'JOB_SEARCH', 'request_metadata': request_metadata, 'job_query': job_query}
    response = client_service.projects().jobs().search(parent=parent, body=request).execute()
    print(response)

def language_code_search(client_service, company_name, language_codes):
    if False:
        return 10
    request_metadata = {'user_id': 'HashedUserId', 'session_id': 'HashedSessionId', 'domain': 'www.google.com'}
    job_query = {'language_codes': language_codes}
    if company_name is not None:
        job_query.update({'company_names': [company_name]})
    request = {'search_mode': 'JOB_SEARCH', 'request_metadata': request_metadata, 'job_query': job_query}
    response = client_service.projects().jobs().search(parent=parent, body=request).execute()
    print(response)

def company_display_name_search(client_service, company_name, company_display_names):
    if False:
        while True:
            i = 10
    request_metadata = {'user_id': 'HashedUserId', 'session_id': 'HashedSessionId', 'domain': 'www.google.com'}
    job_query = {'company_display_names': company_display_names}
    if company_name is not None:
        job_query.update({'company_names': [company_name]})
    request = {'search_mode': 'JOB_SEARCH', 'request_metadata': request_metadata, 'job_query': job_query}
    response = client_service.projects().jobs().search(parent=parent, body=request).execute()
    print(response)

def compensation_search(client_service, company_name):
    if False:
        i = 10
        return i + 15
    request_metadata = {'user_id': 'HashedUserId', 'session_id': 'HashedSessionId', 'domain': 'www.google.com'}
    compensation_range = {'max_compensation': {'currency_code': 'USD', 'units': 15}, 'min_compensation': {'currency_code': 'USD', 'units': 10, 'nanos': 500000000}}
    compensation_filter = {'type': 'UNIT_AND_AMOUNT', 'units': ['HOURLY'], 'range': compensation_range}
    job_query = {'compensation_filter': compensation_filter}
    if company_name is not None:
        job_query.update({'company_names': [company_name]})
    request = {'search_mode': 'JOB_SEARCH', 'request_metadata': request_metadata, 'job_query': job_query}
    response = client_service.projects().jobs().search(parent=parent, body=request).execute()
    print(response)

def set_up():
    if False:
        return 10
    import base_company_sample
    import base_job_sample
    company_to_be_created = base_company_sample.generate_company()
    company_to_be_created.update({'display_name': 'Google'})
    company_created = base_company_sample.create_company(client_service, company_to_be_created)
    company_name = company_created.get('name')
    job_to_be_created = base_job_sample.generate_job_with_required_fields(company_name)
    amount = {'currency_code': 'USD', 'units': 12}
    compensation_info = {'entries': [{'type': 'BASE', 'unit': 'HOURLY', 'amount': amount}]}
    job_to_be_created.update({'title': 'Systems Administrator', 'employment_types': 'FULL_TIME', 'language_code': 'en-US', 'compensation_info': compensation_info})
    job_name = base_job_sample.create_job(client_service, job_to_be_created).get('name')
    return (company_name, job_name)

def tear_down(company_name, job_name):
    if False:
        while True:
            i = 10
    import base_company_sample
    import base_job_sample
    base_job_sample.delete_job(client_service, job_name)
    base_company_sample.delete_company(client_service, company_name)

def run_sample(company_name, job_name):
    if False:
        while True:
            i = 10
    basic_keyword_search(client_service, company_name, 'Systems Administrator')
    category_search(client_service, company_name, ['COMPUTER_AND_IT'])
    date_range = {'start_time': '2018-07-01T00:00:00Z'}
    date_range_search(client_service, company_name, date_range)
    employment_types_search(client_service, company_name, ['FULL_TIME', 'CONTRACTOR', 'PER_DIEM'])
    company_display_name_search(client_service, company_name, ['Google'])
    compensation_search(client_service, company_name)
    language_code_search(client_service, company_name, ['pt-BR', 'en-US'])
if __name__ == '__main__':
    (company_name, job_name) = set_up()
    time.sleep(10)
    run_sample(company_name, job_name)
    tear_down(company_name, job_name)