import os
import frappe

def execute():
    if False:
        print('Hello World!')
    site = frappe.local.site
    log_folder = os.path.join(site, 'logs')
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)