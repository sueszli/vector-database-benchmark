import frappe
from frappe.utils import cint

def execute():
    if False:
        while True:
            i = 10
    "\n\tThe motive of this patch is to increase the overall security in frappe framework\n\n\tExisting passwords won't be affected, however, newly created accounts\n\twill have to adheare to the new password policy guidelines. Once can always\n\tloosen up the security by modifying the values in System Settings, however,\n\twe strongly advice against doing so!\n\n\tSecurity is something we take very seriously at frappe,\n\tand hence we chose to make security tighter by default.\n\t"
    doc = frappe.get_single('System Settings')
    if cint(doc.enable_password_policy) == 0:
        doc.enable_password_policy = 1
    if cint(doc.minimum_password_score) <= 2:
        doc.minimum_password_score = 2
    if cint(doc.allow_consecutive_login_attempts) <= 3:
        doc.allow_consecutive_login_attempts = 3
    doc.flags.ignore_mandatory = True
    doc.save()