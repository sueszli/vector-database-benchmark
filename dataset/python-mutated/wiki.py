import argparse
import datetime
from google.cloud import datastore

def path_to_key(datastore, path):
    if False:
        return 10
    '\n    Translates a file system path to a datastore key. The basename becomes the\n    key name and the extension becomes the kind.\n\n    Examples:\n        /file.ext -> key(ext, file)\n        /parent.ext/file.ext -> key(ext, parent, ext, file)\n    '
    key_parts = []
    path_parts = path.strip('/').split('/')
    for (n, x) in enumerate(path_parts):
        (name, ext) = x.rsplit('.', 1)
        key_parts.extend([ext, name])
    return datastore.key(*key_parts)

def save_page(ds, page, content):
    if False:
        i = 10
        return i + 15
    with ds.transaction():
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        current_key = path_to_key(ds, '{}.page/current.revision'.format(page))
        revision_key = path_to_key(ds, '{}.page/{}.revision'.format(page, now))
        if ds.get(revision_key):
            raise AssertionError('Revision %s already exists' % revision_key)
        current = ds.get(current_key)
        if current:
            revision = datastore.Entity(key=revision_key)
            revision.update(current)
            ds.put(revision)
        else:
            current = datastore.Entity(key=current_key)
        current['content'] = content
        ds.put(current)

def restore_revision(ds, page, revision):
    if False:
        i = 10
        return i + 15
    save_page(ds, page, revision['content'])

def list_pages(ds):
    if False:
        i = 10
        return i + 15
    return ds.query(kind='page').fetch()

def list_revisions(ds, page):
    if False:
        for i in range(10):
            print('nop')
    page_key = path_to_key(ds, '{}.page'.format(page))
    return ds.query(kind='revision', ancestor=page_key).fetch()

def main(project_id):
    if False:
        for i in range(10):
            print('nop')
    ds = datastore.Client(project_id)
    save_page(ds, 'page1', '1')
    save_page(ds, 'page1', '2')
    save_page(ds, 'page1', '3')
    print('Revisions for page1:')
    first_revision = None
    for revision in list_revisions(ds, 'page1'):
        if not first_revision:
            first_revision = revision
        print('{}: {}'.format(revision.key.name, revision['content']))
    print('restoring revision {}:'.format(first_revision.key.name))
    restore_revision(ds, 'page1', first_revision)
    print('Revisions for page1:')
    for revision in list_revisions(ds, 'page1'):
        print('{}: {}'.format(revision.key.name, revision['content']))
    print('Cleaning up')
    ds.delete_multi([path_to_key(ds, 'page1.page')])
    ds.delete_multi([x.key for x in list_revisions(ds, 'page1')])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrates wiki data model.')
    parser.add_argument('project_id', help='Your cloud project ID.')
    args = parser.parse_args()
    main(args.project_id)