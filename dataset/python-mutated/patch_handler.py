""" Patch Handler.

This file manages execution of manaully written patches. Patches are script
that apply changes in database schema or data to accomodate for changes in the
code.

Ways to specify patches:

1. patches.txt file specifies patches that run before doctype schema
migration. Each line represents one patch (old format).
2. patches.txt can alternatively also separate pre and post model sync
patches by using INI like file format:
	```patches.txt
	[pre_model_sync]
	app.module.patch1
	app.module.patch2


	[post_model_sync]
	app.module.patch3
	```

	When different sections are specified patches are executed in this order:
		1. Run pre_model_sync patches
		2. Reload/resync all doctype schema
		3. Run post_model_sync patches

	Hence any patch that just needs to modify data but doesn't depend on
	old schema should be added to post_model_sync section of file.

3. simple python commands can be added by starting line with `execute:`
`execute:` example: `execute:print("hello world")`
"""
import configparser
import time
from enum import Enum
from textwrap import dedent, indent
import frappe

class PatchError(Exception):
    pass

class PatchType(Enum):
    pre_model_sync = 'pre_model_sync'
    post_model_sync = 'post_model_sync'

def run_all(skip_failing: bool=False, patch_type: PatchType | None=None) -> None:
    if False:
        while True:
            i = 10
    'run all pending patches'
    executed = set(frappe.get_all('Patch Log', filters={'skipped': 0}, fields='patch', pluck='patch'))
    frappe.flags.final_patches = []

    def run_patch(patch):
        if False:
            while True:
                i = 10
        try:
            if not run_single(patchmodule=patch):
                print(patch + ': failed: STOPPED')
                raise PatchError(patch)
        except Exception:
            if not skip_failing:
                raise
            print('Failed to execute patch')
            update_patch_log(patch, skipped=True)
    patches = get_all_patches(patch_type=patch_type)
    for patch in patches:
        if patch and patch not in executed:
            run_patch(patch)
    for patch in frappe.flags.final_patches:
        patch = patch.replace('finally:', '')
        run_patch(patch)

def get_all_patches(patch_type: PatchType | None=None) -> list[str]:
    if False:
        while True:
            i = 10
    if patch_type and (not isinstance(patch_type, PatchType)):
        frappe.throw(f'Unsupported patch type specified: {patch_type}')
    patches = []
    for app in frappe.get_installed_apps():
        patches.extend(get_patches_from_app(app, patch_type=patch_type))
    return patches

def get_patches_from_app(app: str, patch_type: PatchType | None=None) -> list[str]:
    if False:
        print('Hello World!')
    "Get patches from an app's patches.txt\n\n\tpatches.txt can be:\n\t        1. ini like file with section for different patch_type\n\t        2. plain text file with each line representing a patch.\n\t"
    patches_file = frappe.get_app_path(app, 'patches.txt')
    try:
        return parse_as_configfile(patches_file, patch_type)
    except configparser.MissingSectionHeaderError:
        if not patch_type or patch_type == PatchType.pre_model_sync:
            return frappe.get_file_items(patches_file)
    return []

def parse_as_configfile(patches_file: str, patch_type: PatchType | None=None) -> list[str]:
    if False:
        return 10
    parser = configparser.ConfigParser(allow_no_value=True, delimiters='\n')
    parser.optionxform = str
    parser.read(patches_file)
    if not parser.sections():
        return []
    if not patch_type:
        return [patch for patch in parser[PatchType.pre_model_sync.value]] + [patch for patch in parser[PatchType.post_model_sync.value]]
    if patch_type.value in parser.sections():
        return [patch for patch in parser[patch_type.value]]
    else:
        frappe.throw(frappe._('Patch type {} not found in patches.txt').format(patch_type))

def reload_doc(args):
    if False:
        for i in range(10):
            print('nop')
    import frappe.modules
    run_single(method=frappe.modules.reload_doc, methodargs=args)

def run_single(patchmodule=None, method=None, methodargs=None, force=False):
    if False:
        for i in range(10):
            print('nop')
    from frappe import conf
    conf.developer_mode = 0
    if force or method or (not executed(patchmodule)):
        return execute_patch(patchmodule, method, methodargs)
    else:
        return True

def execute_patch(patchmodule: str, method=None, methodargs=None):
    if False:
        return 10
    'execute the patch'
    _patch_mode(True)
    if patchmodule.startswith('execute:'):
        has_patch_file = False
        patch = patchmodule.split('execute:')[1]
        docstring = ''
    else:
        has_patch_file = True
        patch = f'{patchmodule.split(maxsplit=1)[0]}.execute'
        _patch = frappe.get_attr(patch)
        docstring = _patch.__doc__ or ''
        if docstring:
            docstring = '\n' + indent(dedent(docstring), '\t')
    print(f'Executing {patchmodule or methodargs} in {frappe.local.site} ({frappe.db.cur_db_name}){docstring}')
    start_time = time.monotonic()
    frappe.db.begin()
    frappe.db.auto_commit_on_many_writes = 0
    try:
        if patchmodule:
            if patchmodule.startswith('finally:'):
                frappe.flags.final_patches.append(patchmodule)
            else:
                if has_patch_file:
                    _patch()
                else:
                    exec(patch, globals())
                update_patch_log(patchmodule)
        elif method:
            method(**methodargs)
    except Exception:
        frappe.db.rollback()
        raise
    else:
        frappe.db.commit()
        end_time = time.monotonic()
        _patch_mode(False)
        print(f'Success: Done in {round(end_time - start_time, 3)}s')
    return True

def update_patch_log(patchmodule, skipped=False):
    if False:
        print('Hello World!')
    'update patch_file in patch log'
    patch = frappe.get_doc({'doctype': 'Patch Log', 'patch': patchmodule})
    if skipped:
        traceback = frappe.get_traceback(with_context=True)
        patch.skipped = 1
        patch.traceback = traceback
        print(traceback, end='\n\n')
    patch.insert(ignore_permissions=True)

def executed(patchmodule):
    if False:
        print('Hello World!')
    'return True if is executed'
    if patchmodule.startswith('finally:'):
        patchmodule = patchmodule.replace('finally:', '')
    return frappe.db.get_value('Patch Log', {'patch': patchmodule, 'skipped': 0})

def _patch_mode(enable):
    if False:
        return 10
    'stop/start execution till patch is run'
    frappe.local.flags.in_patch = enable
    frappe.db.commit()