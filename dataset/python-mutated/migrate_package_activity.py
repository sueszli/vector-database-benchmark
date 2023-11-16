"""
Migrates revisions into the activity stream, to allow you to view old versions
of datasets and changes (diffs) between them.

This should be run once you've upgraded to CKAN 2.9.

This script is not part of the main migrations because it takes a long time to
run, and you don't want it to delay a site going live again after an upgrade.
In the period between upgrading CKAN and this script completes, the Activity
Stream's view of old versions of datasets and diffs between them will be
incomplete - it won't show resources, extras or tags.

This script is idempotent - there is no harm in running this multiple times, or
stopping and restarting it.

We won't delete the revision tables in the database yet, since we haven't
converted the group, package_relationship to activity objects yet.

(In a future version of CKAN we will remove the 'package_revision' table from
the codebase. We'll need a step in the main migration which checks that
migrate_package_activity.py has been done, before it removes the
package_revision table.)
"""
from __future__ import print_function
from __future__ import absolute_import
import argparse
from collections import defaultdict
from typing import Any
import sys
_context: Any = None

def get_context():
    if False:
        return 10
    from ckan import model
    import ckan.logic as logic
    global _context
    if not _context:
        user = logic.get_action(u'get_site_user')({u'model': model, u'ignore_auth': True}, {})
        _context = {u'model': model, u'session': model.Session, u'user': user[u'name']}
    return _context

def num_unmigrated(engine):
    if False:
        print('Hello World!')
    num_unmigrated = engine.execute('\n        SELECT count(*) FROM activity a JOIN package p ON a.object_id=p.id\n        WHERE a.activity_type IN (\'new package\', \'changed package\')\n        AND a.data NOT LIKE \'%%{"actor"%%\'\n        AND p.private = false;\n    ').fetchone()[0]
    return num_unmigrated

def num_activities_migratable():
    if False:
        while True:
            i = 10
    from ckan import model
    num_activities = model.Session.execute(u"\n    SELECT count(*) FROM activity a JOIN package p ON a.object_id=p.id\n    WHERE a.activity_type IN ('new package', 'changed package')\n    AND p.private = false;\n    ").fetchall()[0][0]
    return num_activities

def migrate_all_datasets():
    if False:
        while True:
            i = 10
    import ckan.logic as logic
    dataset_names = logic.get_action(u'package_list')(get_context(), {})
    num_datasets = len(dataset_names)
    errors = defaultdict(int)
    with PackageDictizeMonkeyPatch():
        for (i, dataset_name) in enumerate(dataset_names):
            print(u'\n{}/{} dataset: {}'.format(i + 1, num_datasets, dataset_name))
            migrate_dataset(dataset_name, errors)
    print(u'Migrated:')
    print(u'  {} datasets'.format(len(dataset_names)))
    num_activities = num_activities_migratable()
    print(u'  with {} activities'.format(num_activities))
    print_errors(errors)

class PackageDictizeMonkeyPatch(object):
    """Patches package_dictize to add back in the revision functionality. This
    allows you to specify context['revision_id'] and see the old revisions of
    a package.

    This works as a context object. We could have used mock.patch and saved a
    couple of lines here, but we'd have had to add mock to requirements.txt.
    """

    def __enter__(self):
        if False:
            while True:
                i = 10
        import ckan.lib.dictization.model_dictize as model_dictize
        try:
            import ckan.migration.revision_legacy_code as revision_legacy_code
        except ImportError:
            from . import revision_legacy_code
        self.existing_function = model_dictize.package_dictize
        model_dictize.package_dictize = revision_legacy_code.package_dictize_with_revisions

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        import ckan.lib.dictization.model_dictize as model_dictize
        model_dictize.package_dictize = self.existing_function

def migrate_dataset(dataset_name, errors):
    if False:
        while True:
            i = 10
    '\n    Migrates a single dataset.\n\n    NB this function should be run in a `with PackageDictizeMonkeyPatch():`\n    '
    import ckan.logic as logic
    from ckan import model
    from ckanext.activity.model import Activity
    package_activity_stream = logic.get_action(u'package_activity_list')(get_context(), {u'id': dataset_name, u'include_hidden_activity': True})
    num_activities = len(package_activity_stream)
    if not num_activities:
        print(u'  No activities')
    for (i, activity) in enumerate(reversed(package_activity_stream)):
        print(u'  activity {}/{} {}'.format(i + 1, num_activities, activity[u'timestamp']))
        activity_obj = model.Session.query(Activity).get(activity[u'id'])
        if u'resources' in activity_obj.data.get(u'package', {}):
            print(u'    activity has full dataset already recorded - no action')
            continue
        context = dict(get_context(), for_view=False, revision_id=activity_obj.revision_id, use_cache=False)
        try:
            assert activity_obj.revision_id, u'Revision missing on the activity'
            dataset = logic.get_action(u'package_show')(context, {u'id': activity[u'object_id'], u'include_tracking': False})
        except Exception as exc:
            if isinstance(exc, logic.NotFound):
                error_msg = u'Revision missing'
            else:
                error_msg = str(exc)
            print(u'    Error: {}! Skipping this version (revision_id={}, timestamp={})'.format(error_msg, activity_obj.revision_id, activity_obj.timestamp))
            errors[error_msg] += 1
            try:
                dataset = {u'title': activity_obj.data['package']['title']}
            except KeyError:
                dataset = {u'title': u'unknown'}
        if u'revision_timestamp' in (dataset.get(u'organization') or {}):
            del dataset[u'organization'][u'revision_timestamp']
        for res in dataset.get(u'resources', []):
            if u'revision_timestamp' in res:
                del res[u'revision_timestamp']
        actor = model.Session.query(model.User).get(activity[u'user_id'])
        actor_name = actor.name if actor else activity[u'user_id']
        data = {u'package': dataset, u'actor': actor_name}
        activity_obj.data = data
    if model.Session.dirty:
        model.Session.commit()
        print(u'  saved')
    print(u"  This package's {} activities are migrated".format(len(package_activity_stream)))

def wipe_activity_detail(delete_activity_detail):
    if False:
        print('Hello World!')
    from ckan import model
    activity_detail_has_rows = bool(model.Session.execute(u'SELECT count(*) FROM (SELECT * FROM "activity_detail" LIMIT 1) as t;').fetchall()[0][0])
    if not activity_detail_has_rows:
        print(u'\nactivity_detail table is aleady emptied')
        return
    print(u'\nNow the migration is done, the history of datasets is now stored\nin the activity table. As a result, the contents of the\nactivity_detail table will no longer be used after CKAN 2.8.x, and\nyou can delete it to save space (this is safely done before or\nafter the CKAN upgrade).')
    if delete_activity_detail is None:
        delete_activity_detail = input(u'Delete activity_detail table content? (y/n):')
    if delete_activity_detail.lower()[:1] != u'y':
        return
    from ckan import model
    model.Session.execute(u'DELETE FROM "activity_detail";')
    model.Session.commit()
    print(u'activity_detail deleted')

def print_errors(errors):
    if False:
        print('Hello World!')
    if errors:
        print(u'Error summary:')
        for (error_msg, count) in errors.items():
            print(u'  {} {}'.format(count, error_msg))
        print(u'\nThese errors are unusual - maybe a dataset was deleted, purged and then\nrecreated, or the revisions corrupted for some reason. These activity items now\ndon\'t have a package_dict recorded against them, which means that when a user\nclicks "View this version" or "Changes" in the Activity Stream for it, it will\nbe missing. Hopefully that\'s acceptable enough to just ignore, because these\nerrors are really hard to fix.\n            ')
if __name__ == u'__main__':
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument(u'-c', u'--config', help=u'CKAN config file (.ini)')
    parser.add_argument(u'--delete', choices=[u'yes', u'no'], help=u'Delete activity detail')
    parser.add_argument(u'--dataset', help=u'just migrate this particular dataset - specify its name')
    args = parser.parse_args()
    assert args.config, u'You must supply a --config'
    print(u'Loading config')
    from ckan.plugins import plugin_loaded
    try:
        from ckan.cli import load_config
        from ckan.config.middleware import make_app
        make_app(load_config(args.config))
    except ImportError:

        def load_config(config):
            if False:
                i = 10
                return i + 15
            from ckan.lib.cli import CkanCommand
            cmd = CkanCommand(name=None)

            class Options(object):
                pass
            cmd.options = Options()
            cmd.options.config = config
            cmd._load_config()
            return
        load_config(args.config)
    if not plugin_loaded('activity'):
        print('Please add the `activity` plugin to your `ckan.plugins` setting')
        sys.exit(1)
    if not args.dataset:
        migrate_all_datasets()
        wipe_activity_detail(delete_activity_detail=args.delete)
    else:
        errors: Any = defaultdict(int)
        with PackageDictizeMonkeyPatch():
            migrate_dataset(args.dataset, errors)
        print_errors(errors)