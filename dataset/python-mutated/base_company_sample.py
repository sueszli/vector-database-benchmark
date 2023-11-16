import os
import random
import string
from googleapiclient.discovery import build
from googleapiclient.errors import Error
client_service = build('jobs', 'v3')
parent = 'projects/' + os.environ['GOOGLE_CLOUD_PROJECT']

def generate_company():
    if False:
        i = 10
        return i + 15
    external_id = 'company:' + ''.join((random.choice(string.ascii_uppercase + string.digits) for _ in range(16)))
    display_name = 'Google'
    headquarters_address = '1600 Amphitheatre Parkway Mountain View, CA 94043'
    company = {'display_name': display_name, 'external_id': external_id, 'headquarters_address': headquarters_address}
    print('Company generated: %s' % company)
    return company

def create_company(client_service, company_to_be_created):
    if False:
        for i in range(10):
            print('nop')
    try:
        request = {'company': company_to_be_created}
        company_created = client_service.projects().companies().create(parent=parent, body=request).execute()
        print('Company created: %s' % company_created)
        return company_created
    except Error as e:
        print('Got exception while creating company')
        raise e

def get_company(client_service, company_name):
    if False:
        while True:
            i = 10
    try:
        company_existed = client_service.projects().companies().get(name=company_name).execute()
        print('Company existed: %s' % company_existed)
        return company_existed
    except Error as e:
        print('Got exception while getting company')
        raise e

def update_company(client_service, company_name, company_to_be_updated):
    if False:
        i = 10
        return i + 15
    try:
        request = {'company': company_to_be_updated}
        company_updated = client_service.projects().companies().patch(name=company_name, body=request).execute()
        print('Company updated: %s' % company_updated)
        return company_updated
    except Error as e:
        print('Got exception while updating company')
        raise e

def update_company_with_field_mask(client_service, company_name, company_to_be_updated, field_mask):
    if False:
        i = 10
        return i + 15
    try:
        request = {'company': company_to_be_updated, 'update_mask': field_mask}
        company_updated = client_service.projects().companies().patch(name=company_name, body=request).execute()
        print('Company updated: %s' % company_updated)
        return company_updated
    except Error as e:
        print('Got exception while updating company with field mask')
        raise e

def delete_company(client_service, company_name):
    if False:
        return 10
    try:
        client_service.projects().companies().delete(name=company_name).execute()
        print('Company deleted')
    except Error as e:
        print('Got exception while deleting company')
        raise e

def run_sample():
    if False:
        return 10
    company_to_be_created = generate_company()
    company_created = create_company(client_service, company_to_be_created)
    company_name = company_created.get('name')
    get_company(client_service, company_name)
    company_to_be_updated = company_created
    company_to_be_updated.update({'websiteUri': 'https://elgoog.im/'})
    update_company(client_service, company_name, company_to_be_updated)
    update_company_with_field_mask(client_service, company_name, {'displayName': 'changedTitle', 'externalId': company_created.get('externalId')}, 'displayName')
    delete_company(client_service, company_name)
if __name__ == '__main__':
    run_sample()