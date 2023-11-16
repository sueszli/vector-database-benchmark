from functools import partial
from types import FunctionType, MethodType, ModuleType
import frappe
from frappe import _
from frappe.model.document import Document
from frappe.rate_limiter import rate_limit
from frappe.utils.safe_exec import NamespaceDict, get_safe_globals, is_safe_exec_enabled, safe_exec

class ServerScript(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        allow_guest: DF.Check
        api_method: DF.Data | None
        cron_format: DF.Data | None
        disabled: DF.Check
        doctype_event: DF.Literal['Before Insert', 'Before Validate', 'Before Save', 'After Insert', 'After Save', 'Before Submit', 'After Submit', 'Before Cancel', 'After Cancel', 'Before Delete', 'After Delete', 'Before Save (Submitted Document)', 'After Save (Submitted Document)', 'On Payment Authorization']
        enable_rate_limit: DF.Check
        event_frequency: DF.Literal['All', 'Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly', 'Hourly Long', 'Daily Long', 'Weekly Long', 'Monthly Long', 'Cron']
        module: DF.Link | None
        rate_limit_count: DF.Int
        rate_limit_seconds: DF.Int
        reference_doctype: DF.Link | None
        script: DF.Code
        script_type: DF.Literal['DocType Event', 'Scheduler Event', 'Permission Query', 'API']

    def validate(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.only_for('Script Manager', True)
        self.sync_scheduled_jobs()
        self.clear_scheduled_events()
        self.check_if_compilable_in_restricted_context()

    def on_update(self):
        if False:
            for i in range(10):
                print('nop')
        self.sync_scheduler_events()

    def clear_cache(self):
        if False:
            print('Hello World!')
        frappe.cache.delete_value('server_script_map')
        return super().clear_cache()

    def on_trash(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.cache.delete_value('server_script_map')
        if self.script_type == 'Scheduler Event':
            for job in self.scheduled_jobs:
                frappe.delete_doc('Scheduled Job Type', job.name)

    def get_code_fields(self):
        if False:
            while True:
                i = 10
        return {'script': 'py'}

    @property
    def scheduled_jobs(self) -> list[dict[str, str]]:
        if False:
            print('Hello World!')
        return frappe.get_all('Scheduled Job Type', filters={'server_script': self.name}, fields=['name', 'stopped'])

    def sync_scheduled_jobs(self):
        if False:
            while True:
                i = 10
        "Sync Scheduled Job Type statuses if Server Script's disabled status is changed"
        if self.script_type != 'Scheduler Event' or not self.has_value_changed('disabled'):
            return
        for scheduled_job in self.scheduled_jobs:
            if bool(scheduled_job.stopped) != bool(self.disabled):
                job = frappe.get_doc('Scheduled Job Type', scheduled_job.name)
                job.stopped = self.disabled
                job.save()

    def sync_scheduler_events(self):
        if False:
            i = 10
            return i + 15
        'Create or update Scheduled Job Type documents for Scheduler Event Server Scripts'
        if not self.disabled and self.event_frequency and (self.script_type == 'Scheduler Event'):
            cron_format = self.cron_format if self.event_frequency == 'Cron' else None
            setup_scheduler_events(script_name=self.name, frequency=self.event_frequency, cron_format=cron_format)

    def clear_scheduled_events(self):
        if False:
            for i in range(10):
                print('nop')
        'Deletes existing scheduled jobs by Server Script if self.event_frequency or self.cron_format has changed'
        if self.script_type == 'Scheduler Event' and (self.has_value_changed('event_frequency') or self.has_value_changed('cron_format')):
            for scheduled_job in self.scheduled_jobs:
                frappe.delete_doc('Scheduled Job Type', scheduled_job.name)

    def check_if_compilable_in_restricted_context(self):
        if False:
            for i in range(10):
                print('nop')
        'Check compilation errors and send them back as warnings.'
        from RestrictedPython import compile_restricted
        try:
            compile_restricted(self.script)
        except Exception as e:
            frappe.msgprint(str(e), title=_('Compilation warning'))

    def execute_method(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        "Specific to API endpoint Server Scripts\n\n\t\tRaises:\n\t\t        frappe.DoesNotExistError: If self.script_type is not API\n\t\t        frappe.PermissionError: If self.allow_guest is unset for API accessed by Guest user\n\n\t\tReturns:\n\t\t        dict: Evaluates self.script with frappe.utils.safe_exec.safe_exec and returns the flags set in it's safe globals\n\t\t"
        if self.enable_rate_limit:
            limit = self.rate_limit_count or 5
            seconds = self.rate_limit_seconds or 24 * 60 * 60
            _fn = partial(execute_api_server_script, script=self)
            return rate_limit(limit=limit, seconds=seconds)(_fn)()
        else:
            return execute_api_server_script(self)

    def execute_doc(self, doc: Document):
        if False:
            for i in range(10):
                print('nop')
        "Specific to Document Event triggered Server Scripts\n\n\t\tArgs:\n\t\t        doc (Document): Executes script with for a certain document's events\n\t\t"
        safe_exec(self.script, _locals={'doc': doc}, restrict_commit_rollback=True)

    def execute_scheduled_method(self):
        if False:
            i = 10
            return i + 15
        'Specific to Scheduled Jobs via Server Scripts\n\n\t\tRaises:\n\t\t        frappe.DoesNotExistError: If script type is not a scheduler event\n\t\t'
        if self.script_type != 'Scheduler Event':
            raise frappe.DoesNotExistError
        safe_exec(self.script)

    def get_permission_query_conditions(self, user: str) -> list[str]:
        if False:
            while True:
                i = 10
        'Specific to Permission Query Server Scripts\n\n\t\tArgs:\n\t\t        user (str): Takes user email to execute script and return list of conditions\n\n\t\tReturns:\n\t\t        list: Returns list of conditions defined by rules in self.script\n\t\t'
        locals = {'user': user, 'conditions': ''}
        safe_exec(self.script, None, locals)
        if locals['conditions']:
            return locals['conditions']

    @frappe.whitelist()
    def get_autocompletion_items(self):
        if False:
            return 10
        'Generates a list of a autocompletion strings from the context dict\n\t\tthat is used while executing a Server Script.\n\n\t\tReturns:\n\t\t        list: Returns list of autocompletion items.\n\t\t        For e.g., ["frappe.utils.cint", "frappe.get_all", ...]\n\t\t'

        def get_keys(obj):
            if False:
                for i in range(10):
                    print('nop')
            out = []
            for key in obj:
                if key.startswith('_'):
                    continue
                value = obj[key]
                if isinstance(value, (NamespaceDict, dict)) and value:
                    if key == 'form_dict':
                        out.append(['form_dict', 7])
                        continue
                    for (subkey, score) in get_keys(value):
                        fullkey = f'{key}.{subkey}'
                        out.append([fullkey, score])
                else:
                    if isinstance(value, type) and issubclass(value, Exception):
                        score = 0
                    elif isinstance(value, ModuleType):
                        score = 10
                    elif isinstance(value, (FunctionType, MethodType)):
                        score = 9
                    elif isinstance(value, type):
                        score = 8
                    elif isinstance(value, dict):
                        score = 7
                    else:
                        score = 6
                    out.append([key, score])
            return out
        items = frappe.cache.get_value('server_script_autocompletion_items')
        if not items:
            items = get_keys(get_safe_globals())
            items = [{'value': d[0], 'score': d[1]} for d in items]
            frappe.cache.set_value('server_script_autocompletion_items', items)
        return items

def setup_scheduler_events(script_name: str, frequency: str, cron_format: str | None=None):
    if False:
        i = 10
        return i + 15
    'Creates or Updates Scheduled Job Type documents based on the specified script name and frequency\n\n\tArgs:\n\t        script_name (str): Name of the Server Script document\n\t        frequency (str): Event label compatible with the Frappe scheduler\n\t'
    method = frappe.scrub(f'{script_name}-{frequency}')
    scheduled_script = frappe.db.get_value('Scheduled Job Type', {'method': method})
    if not scheduled_script:
        frappe.get_doc({'doctype': 'Scheduled Job Type', 'method': method, 'frequency': frequency, 'server_script': script_name, 'cron_format': cron_format}).insert()
        frappe.msgprint(_('Enabled scheduled execution for script {0}').format(script_name))
    else:
        doc = frappe.get_doc('Scheduled Job Type', scheduled_script)
        if doc.frequency == frequency:
            return
        doc.frequency = frequency
        doc.cron_format = cron_format
        doc.save()
        frappe.msgprint(_('Scheduled execution for script {0} has updated').format(script_name))

def execute_api_server_script(script=None, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    del args
    del kwargs
    if script.script_type != 'API':
        raise frappe.DoesNotExistError
    if frappe.session.user == 'Guest' and (not script.allow_guest):
        raise frappe.PermissionError
    (_globals, _locals) = safe_exec(script.script)
    return _globals.frappe.flags

@frappe.whitelist()
def enabled() -> bool | None:
    if False:
        for i in range(10):
            print('nop')
    if frappe.has_permission('Server Script'):
        return is_safe_exec_enabled()