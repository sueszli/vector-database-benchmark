"""Gets parent work, its disambiguation and id, composer, composer sort name
and work composition date
"""
import musicbrainzngs
from beets import ui
from beets.plugins import BeetsPlugin

def direct_parent_id(mb_workid, work_date=None):
    if False:
        i = 10
        return i + 15
    'Given a Musicbrainz work id, find the id one of the works the work is\n    part of and the first composition date it encounters.\n    '
    work_info = musicbrainzngs.get_work_by_id(mb_workid, includes=['work-rels', 'artist-rels'])
    if 'artist-relation-list' in work_info['work'] and work_date is None:
        for artist in work_info['work']['artist-relation-list']:
            if artist['type'] == 'composer':
                if 'end' in artist.keys():
                    work_date = artist['end']
    if 'work-relation-list' in work_info['work']:
        for direct_parent in work_info['work']['work-relation-list']:
            if direct_parent['type'] == 'parts' and direct_parent.get('direction') == 'backward':
                direct_id = direct_parent['work']['id']
                return (direct_id, work_date)
    return (None, work_date)

def work_parent_id(mb_workid):
    if False:
        for i in range(10):
            print('nop')
    'Find the parent work id and composition date of a work given its id.'
    work_date = None
    while True:
        (new_mb_workid, work_date) = direct_parent_id(mb_workid, work_date)
        if not new_mb_workid:
            return (mb_workid, work_date)
        mb_workid = new_mb_workid
    return (mb_workid, work_date)

def find_parentwork_info(mb_workid):
    if False:
        print('Hello World!')
    "Get the MusicBrainz information dict about a parent work, including\n    the artist relations, and the composition date for a work's parent work.\n    "
    (parent_id, work_date) = work_parent_id(mb_workid)
    work_info = musicbrainzngs.get_work_by_id(parent_id, includes=['artist-rels'])
    return (work_info, work_date)

class ParentWorkPlugin(BeetsPlugin):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.config.add({'auto': False, 'force': False})
        if self.config['auto']:
            self.import_stages = [self.imported]

    def commands(self):
        if False:
            i = 10
            return i + 15

        def func(lib, opts, args):
            if False:
                print('Hello World!')
            self.config.set_args(opts)
            force_parent = self.config['force'].get(bool)
            write = ui.should_write()
            for item in lib.items(ui.decargs(args)):
                changed = self.find_work(item, force_parent)
                if changed:
                    item.store()
                    if write:
                        item.try_write()
        command = ui.Subcommand('parentwork', help='fetch parent works, composers and dates')
        command.parser.add_option('-f', '--force', dest='force', action='store_true', default=None, help='re-fetch when parent work is already present')
        command.func = func
        return [command]

    def imported(self, session, task):
        if False:
            for i in range(10):
                print('nop')
        'Import hook for fetching parent works automatically.'
        force_parent = self.config['force'].get(bool)
        for item in task.imported_items():
            self.find_work(item, force_parent)
            item.store()

    def get_info(self, item, work_info):
        if False:
            i = 10
            return i + 15
        'Given the parent work info dict, fetch parent_composer,\n        parent_composer_sort, parentwork, parentwork_disambig, mb_workid and\n        composer_ids.\n        '
        parent_composer = []
        parent_composer_sort = []
        parentwork_info = {}
        composer_exists = False
        if 'artist-relation-list' in work_info['work']:
            for artist in work_info['work']['artist-relation-list']:
                if artist['type'] == 'composer':
                    composer_exists = True
                    parent_composer.append(artist['artist']['name'])
                    parent_composer_sort.append(artist['artist']['sort-name'])
                    if 'end' in artist.keys():
                        parentwork_info['parentwork_date'] = artist['end']
            parentwork_info['parent_composer'] = ', '.join(parent_composer)
            parentwork_info['parent_composer_sort'] = ', '.join(parent_composer_sort)
        if not composer_exists:
            self._log.debug('no composer for {}; add one at https://musicbrainz.org/work/{}', item, work_info['work']['id'])
        parentwork_info['parentwork'] = work_info['work']['title']
        parentwork_info['mb_parentworkid'] = work_info['work']['id']
        if 'disambiguation' in work_info['work']:
            parentwork_info['parentwork_disambig'] = work_info['work']['disambiguation']
        else:
            parentwork_info['parentwork_disambig'] = None
        return parentwork_info

    def find_work(self, item, force):
        if False:
            i = 10
            return i + 15
        'Finds the parent work of a recording and populates the tags\n        accordingly.\n\n        The parent work is found recursively, by finding the direct parent\n        repeatedly until there are no more links in the chain. We return the\n        final, topmost work in the chain.\n\n        Namely, the tags parentwork, parentwork_disambig, mb_parentworkid,\n        parent_composer, parent_composer_sort and work_date are populated.\n        '
        if not item.mb_workid:
            self._log.info('No work for {}, add one at https://musicbrainz.org/recording/{}', item, item.mb_trackid)
            return
        hasparent = hasattr(item, 'parentwork')
        work_changed = True
        if hasattr(item, 'parentwork_workid_current'):
            work_changed = item.parentwork_workid_current != item.mb_workid
        if force or not hasparent or work_changed:
            try:
                (work_info, work_date) = find_parentwork_info(item.mb_workid)
            except musicbrainzngs.musicbrainz.WebServiceError as e:
                self._log.debug('error fetching work: {}', e)
                return
            parent_info = self.get_info(item, work_info)
            parent_info['parentwork_workid_current'] = item.mb_workid
            if 'parent_composer' in parent_info:
                self._log.debug('Work fetched: {} - {}', parent_info['parentwork'], parent_info['parent_composer'])
            else:
                self._log.debug('Work fetched: {} - no parent composer', parent_info['parentwork'])
        elif hasparent:
            self._log.debug('{}: Work present, skipping', item)
            return
        for (key, value) in parent_info.items():
            if value:
                item[key] = value
        if work_date:
            item['work_date'] = work_date
        return ui.show_model_changes(item, fields=['parentwork', 'parentwork_disambig', 'mb_parentworkid', 'parent_composer', 'parent_composer_sort', 'work_date', 'parentwork_workid_current', 'parentwork_date'])