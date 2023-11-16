from click import secho
import frappe

def execute():
    if False:
        for i in range(10):
            print('nop')
    if frappe.get_hooks('jenv'):
        print()
        secho('WARNING: The hook "jenv" is deprecated. Follow the migration guide to use the new "jinja" hook.', fg='yellow')
        secho('https://github.com/frappe/frappe/wiki/Migrating-to-Version-13', fg='yellow')
        print()