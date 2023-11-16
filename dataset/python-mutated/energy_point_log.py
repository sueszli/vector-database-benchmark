import json
import frappe
from frappe import _
from frappe.desk.doctype.notification_log.notification_log import enqueue_create_notification, get_title, get_title_html
from frappe.desk.doctype.notification_settings.notification_settings import is_email_notifications_enabled, is_email_notifications_enabled_for_type
from frappe.model.document import Document
from frappe.utils import cint, get_fullname, get_link_to_form, getdate

class EnergyPointLog(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        points: DF.Int
        reason: DF.Text | None
        reference_doctype: DF.Link | None
        reference_name: DF.Data | None
        revert_of: DF.Link | None
        reverted: DF.Check
        rule: DF.Link | None
        seen: DF.Check
        type: DF.Literal['Auto', 'Appreciation', 'Criticism', 'Review', 'Revert']
        user: DF.Link

    def validate(self):
        if False:
            while True:
                i = 10
        self.map_milestone_reference()
        if self.type in ['Appreciation', 'Criticism'] and self.user == self.owner:
            frappe.throw(_('You cannot give review points to yourself'))

    def map_milestone_reference(self):
        if False:
            return 10
        if self.reference_doctype == 'Milestone':
            (self.reference_doctype, self.reference_name) = frappe.db.get_value('Milestone', self.reference_name, ['reference_type', 'reference_name'])

    def after_insert(self):
        if False:
            i = 10
            return i + 15
        alert_dict = get_alert_dict(self)
        if alert_dict:
            frappe.publish_realtime('energy_point_alert', message=alert_dict, user=self.user, after_commit=True)
        frappe.cache.hdel('energy_points', self.user)
        if self.type != 'Review' and frappe.get_cached_value('Notification Settings', self.user, 'energy_points_system_notifications'):
            reference_user = self.user if self.type == 'Auto' else self.owner
            notification_doc = {'type': 'Energy Point', 'document_type': self.reference_doctype, 'document_name': self.reference_name, 'subject': get_notification_message(self), 'from_user': reference_user, 'email_content': f'<div>{self.reason}</div>' if self.reason else None}
            enqueue_create_notification(self.user, notification_doc)

    def on_trash(self):
        if False:
            print('Hello World!')
        if self.type == 'Revert':
            reference_log = frappe.get_doc('Energy Point Log', self.revert_of)
            reference_log.reverted = 0
            reference_log.save()

    @frappe.whitelist()
    def revert(self, reason, ignore_permissions=False):
        if False:
            i = 10
            return i + 15
        if not ignore_permissions:
            frappe.only_for('System Manager')
        if self.type != 'Auto':
            frappe.throw(_('This document cannot be reverted'))
        if self.get('reverted'):
            return
        self.reverted = 1
        self.save(ignore_permissions=True)
        return frappe.get_doc({'doctype': 'Energy Point Log', 'points': -self.points, 'type': 'Revert', 'user': self.user, 'reason': reason, 'reference_doctype': self.reference_doctype, 'reference_name': self.reference_name, 'revert_of': self.name}).insert(ignore_permissions=True)

def get_notification_message(doc):
    if False:
        for i in range(10):
            print('nop')
    owner_name = get_fullname(doc.owner)
    points = doc.points
    title = get_title(doc.reference_doctype, doc.reference_name)
    if doc.type == 'Auto':
        owner_name = frappe.bold('You')
        if points == 1:
            message = _('{0} gained {1} point for {2} {3}')
        else:
            message = _('{0} gained {1} points for {2} {3}')
        message = message.format(owner_name, frappe.bold(points), doc.rule, get_title_html(title))
    elif doc.type == 'Appreciation':
        if points == 1:
            message = _('{0} appreciated your work on {1} with {2} point')
        else:
            message = _('{0} appreciated your work on {1} with {2} points')
        message = message.format(frappe.bold(owner_name), get_title_html(title), frappe.bold(points))
    elif doc.type == 'Criticism':
        if points == 1:
            message = _('{0} criticized your work on {1} with {2} point')
        else:
            message = _('{0} criticized your work on {1} with {2} points')
        message = message.format(frappe.bold(owner_name), get_title_html(title), frappe.bold(points))
    elif doc.type == 'Revert':
        if points == 1:
            message = _('{0} reverted your point on {1}')
        else:
            message = _('{0} reverted your points on {1}')
        message = message.format(frappe.bold(owner_name), get_title_html(title))
    return message

def get_alert_dict(doc):
    if False:
        print('Hello World!')
    alert_dict = frappe._dict()
    owner_name = get_fullname(doc.owner)
    if doc.reference_doctype:
        doc_link = get_link_to_form(doc.reference_doctype, doc.reference_name)
    points = doc.points
    bold_points = frappe.bold(doc.points)
    if doc.type == 'Auto':
        if points == 1:
            message = _('You gained {0} point')
        else:
            message = _('You gained {0} points')
        alert_dict.message = message.format(bold_points)
        alert_dict.indicator = 'green'
    elif doc.type == 'Appreciation':
        if points == 1:
            message = _('{0} appreciated your work on {1} with {2} point')
        else:
            message = _('{0} appreciated your work on {1} with {2} points')
        alert_dict.message = message.format(owner_name, doc_link, bold_points)
        alert_dict.indicator = 'green'
    elif doc.type == 'Criticism':
        if points == 1:
            message = _('{0} criticized your work on {1} with {2} point')
        else:
            message = _('{0} criticized your work on {1} with {2} points')
        alert_dict.message = message.format(owner_name, doc_link, bold_points)
        alert_dict.indicator = 'red'
    elif doc.type == 'Revert':
        if points == 1:
            message = _('{0} reverted your point on {1}')
        else:
            message = _('{0} reverted your points on {1}')
        alert_dict.message = message.format(owner_name, doc_link)
        alert_dict.indicator = 'red'
    return alert_dict

def create_energy_points_log(ref_doctype, ref_name, doc, apply_only_once=False):
    if False:
        i = 10
        return i + 15
    doc = frappe._dict(doc)
    log_exists = check_if_log_exists(ref_doctype, ref_name, doc.rule, None if apply_only_once else doc.user)
    if log_exists:
        return frappe.get_doc('Energy Point Log', log_exists)
    new_log = frappe.new_doc('Energy Point Log')
    new_log.reference_doctype = ref_doctype
    new_log.reference_name = ref_name
    new_log.update(doc)
    new_log.insert(ignore_permissions=True)
    return new_log

def check_if_log_exists(ref_doctype, ref_name, rule, user=None):
    if False:
        while True:
            i = 10
    "'Checks if Energy Point Log already exists"
    filters = frappe._dict({'rule': rule, 'reference_doctype': ref_doctype, 'reference_name': ref_name, 'reverted': 0})
    if user:
        filters.user = user
    return frappe.db.exists('Energy Point Log', filters)

def create_review_points_log(user, points, reason=None, doctype=None, docname=None):
    if False:
        i = 10
        return i + 15
    return frappe.get_doc({'doctype': 'Energy Point Log', 'points': points, 'type': 'Review', 'user': user, 'reason': reason, 'reference_doctype': doctype, 'reference_name': docname}).insert(ignore_permissions=True)

@frappe.whitelist()
def add_review_points(user, points):
    if False:
        print('Hello World!')
    frappe.only_for('System Manager')
    create_review_points_log(user, points)

@frappe.whitelist()
def get_energy_points(user):
    if False:
        return 10
    points = get_user_energy_and_review_points(user)
    return frappe._dict(points.get(user, {}))

@frappe.whitelist()
def get_user_energy_and_review_points(user=None, from_date=None, as_dict=True):
    if False:
        return 10
    conditions = ''
    given_points_condition = ''
    values = frappe._dict()
    if user:
        conditions = 'WHERE `user` = %(user)s'
        values.user = user
    if from_date:
        conditions += 'WHERE' if not conditions else 'AND'
        given_points_condition += 'AND `creation` >= %(from_date)s'
        conditions += " `creation` >= %(from_date)s OR `type`='Review'"
        values.from_date = from_date
    points_list = frappe.db.sql("\n\t\tSELECT\n\t\t\tSUM(CASE WHEN `type` != 'Review' THEN `points` ELSE 0 END) AS energy_points,\n\t\t\tSUM(CASE WHEN `type` = 'Review' THEN `points` ELSE 0 END) AS review_points,\n\t\t\tSUM(CASE\n\t\t\t\tWHEN `type`='Review' AND `points` < 0 {given_points_condition}\n\t\t\t\tTHEN ABS(`points`)\n\t\t\t\tELSE 0\n\t\t\tEND) as given_points,\n\t\t\t`user`\n\t\tFROM `tabEnergy Point Log`\n\t\t{conditions}\n\t\tGROUP BY `user`\n\t\tORDER BY `energy_points` DESC\n\t".format(conditions=conditions, given_points_condition=given_points_condition), values=values, as_dict=1)
    if not as_dict:
        return points_list
    dict_to_return = frappe._dict()
    for d in points_list:
        dict_to_return[d.pop('user')] = d
    return dict_to_return

@frappe.whitelist()
def review(doc, points, to_user, reason, review_type='Appreciation'):
    if False:
        for i in range(10):
            print('nop')
    current_review_points = get_energy_points(frappe.session.user).review_points
    doc = doc.as_dict() if hasattr(doc, 'as_dict') else frappe._dict(json.loads(doc))
    points = abs(cint(points))
    if current_review_points < points:
        frappe.msgprint(_('You do not have enough review points'))
        return
    review_doc = create_energy_points_log(doc.doctype, doc.name, {'type': review_type, 'reason': reason, 'points': points if review_type == 'Appreciation' else -points, 'user': to_user})
    create_review_points_log(user=frappe.session.user, points=-points, reason=reason, doctype=review_doc.doctype, docname=review_doc.name)
    return review_doc

@frappe.whitelist()
def get_reviews(doctype, docname):
    if False:
        return 10
    return frappe.get_all('Energy Point Log', filters={'reference_doctype': doctype, 'reference_name': docname, 'type': ['in', ('Appreciation', 'Criticism')]}, fields=['points', 'owner', 'type', 'user', 'reason', 'creation'])

def send_weekly_summary():
    if False:
        print('Hello World!')
    send_summary('Weekly')

def send_monthly_summary():
    if False:
        return 10
    send_summary('Monthly')

def send_summary(timespan):
    if False:
        i = 10
        return i + 15
    from frappe.social.doctype.energy_point_settings.energy_point_settings import is_energy_point_enabled
    from frappe.utils.user import get_enabled_system_users
    if not is_energy_point_enabled():
        return
    if not is_email_notifications_enabled_for_type(frappe.session.user, 'Energy Point'):
        return
    from_date = frappe.utils.add_to_date(None, weeks=-1)
    if timespan == 'Monthly':
        from_date = frappe.utils.add_to_date(None, months=-1)
    user_points = get_user_energy_and_review_points(from_date=from_date, as_dict=False)
    if not user_points or not user_points[0].energy_points:
        return
    from_date = getdate(from_date)
    to_date = getdate()
    all_users = [user.email for user in get_enabled_system_users() if is_email_notifications_enabled_for_type(user.name, 'Energy Point')]
    frappe.sendmail(subject=f'{timespan} energy points summary', recipients=all_users, template='energy_points_summary', args={'top_performer': user_points[0], 'top_reviewer': max(user_points, key=lambda x: x['given_points']), 'standings': user_points[:10], 'footer_message': get_footer_message(timespan).format(from_date, to_date)}, with_container=1)

def get_footer_message(timespan):
    if False:
        for i in range(10):
            print('nop')
    if timespan == 'Monthly':
        return _("Stats based on last month's performance (from {0} to {1})")
    else:
        return _("Stats based on last week's performance (from {0} to {1})")