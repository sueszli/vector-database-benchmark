import sublime, sublime_plugin
import os
import re
import webbrowser
import itertools
import threading
from datetime import datetime, tzinfo, timedelta
import time
platform = sublime.platform()
ST3 = int(sublime.version()) >= 3000
if ST3:
    from .APlainTasksCommon import PlainTasksBase, PlainTasksFold, get_all_projects_and_separators
else:
    from APlainTasksCommon import PlainTasksBase, PlainTasksFold, get_all_projects_and_separators
    sublime_plugin.ViewEventListener = object
if not ST3 and platform == 'linux':
    import codecs as io
else:
    import io
NT = platform == 'windows'
if NT:
    import subprocess
if ST3:
    from datetime import timezone
else:

    class timezone(tzinfo):
        __slots__ = ('_offset', '_name')

        def __init__(self, offset, name=None):
            if False:
                print('Hello World!')
            if not isinstance(offset, timedelta):
                raise TypeError('offset must be a timedelta')
            self._offset = offset
            self._name = name

        def utcoffset(self, dt):
            if False:
                i = 10
                return i + 15
            return self._offset

        def tzname(self, dt):
            if False:
                return 10
            return self._name

        def dst(self, dt):
            if False:
                print('Hello World!')
            return timedelta(0)

def tznow():
    if False:
        while True:
            i = 10
    t = time.time()
    d = datetime.fromtimestamp(t)
    u = datetime.utcfromtimestamp(t)
    return d.replace(tzinfo=timezone(d - u))

def check_parentheses(date_format, regex_group, is_date=False):
    if False:
        print('Hello World!')
    if is_date:
        try:
            parentheses = regex_group if datetime.strptime(regex_group.strip(), date_format) else ''
        except ValueError:
            parentheses = ''
    else:
        try:
            parentheses = '' if datetime.strptime(regex_group.strip(), date_format) else regex_group
        except ValueError:
            parentheses = regex_group
    return parentheses

class PlainTasksNewCommand(PlainTasksBase):

    def runCommand(self, edit):
        if False:
            return 10
        regions = itertools.chain(*(reversed(self.view.lines(region)) for region in reversed(list(self.view.sel()))))
        header_to_task = self.view.settings().get('header_to_task', False)
        sels = self.view.sel()
        eol = None
        for (i, line) in enumerate(regions):
            line_contents = self.view.substr(line).rstrip()
            not_empty_line = re.match('^(\\s*)(\\S.*)$', self.view.substr(line))
            empty_line = re.match('^(\\s+)$', self.view.substr(line))
            current_scope = self.view.scope_name(line.a)
            eol = line.b
            if 'item' in current_scope:
                grps = not_empty_line.groups()
                line_contents = self.view.substr(line) + '\n' + grps[0] + self.open_tasks_bullet + self.tasks_bullet_space
            elif 'header' in current_scope and line_contents and (not header_to_task):
                grps = not_empty_line.groups()
                line_contents = self.view.substr(line) + '\n' + grps[0] + self.before_tasks_bullet_spaces + self.open_tasks_bullet + self.tasks_bullet_space
            elif 'separator' in current_scope:
                grps = not_empty_line.groups()
                line_contents = self.view.substr(line) + '\n' + grps[0] + self.before_tasks_bullet_spaces + self.open_tasks_bullet + self.tasks_bullet_space
            elif not ('header' and 'separator') in current_scope or header_to_task:
                eol = None
                if not_empty_line:
                    grps = not_empty_line.groups()
                    line_contents = (grps[0] if len(grps[0]) > 0 else self.before_tasks_bullet_spaces) + self.open_tasks_bullet + self.tasks_bullet_space + grps[1]
                elif empty_line:
                    grps = empty_line.groups()
                    line_contents = grps[0] + self.open_tasks_bullet + self.tasks_bullet_space
                else:
                    line_contents = self.before_tasks_bullet_spaces + self.open_tasks_bullet + self.tasks_bullet_space
            else:
                print('oops, need to improve PlainTasksNewCommand')
            if eol:
                sels.subtract(sels[~i])
                sels.add(sublime.Region(eol, eol))
            self.view.replace(edit, line, line_contents)
        new_selections = []
        for sel in list(self.view.sel()):
            eol = self.view.line(sel).b
            new_selections.append(sublime.Region(eol, eol))
        self.view.sel().clear()
        for sel in new_selections:
            self.view.sel().add(sel)
        PlainTasksStatsStatus.set_stats(self.view)
        self.view.run_command('plain_tasks_toggle_highlight_past_due')

class PlainTasksNewWithDateCommand(PlainTasksBase):

    def runCommand(self, edit):
        if False:
            for i in range(10):
                print('nop')
        self.view.run_command('plain_tasks_new')
        sels = list(self.view.sel())
        suffix = ' @created%s' % tznow().strftime(self.date_format)
        points = []
        for s in reversed(sels):
            if self.view.substr(sublime.Region(s.b - 2, s.b)) == '  ':
                point = s.b - 2
            else:
                point = s.b
            self.view.insert(edit, point, suffix)
            points.append(point)
        self.view.sel().clear()
        offset = len(suffix)
        for (i, sel) in enumerate(sels):
            self.view.sel().add(sublime.Region(points[~i] + i * offset, points[~i] + i * offset))

class PlainTasksCompleteCommand(PlainTasksBase):

    def runCommand(self, edit):
        if False:
            return 10
        original = [r for r in self.view.sel()]
        (done_line_end, now) = self.format_line_end(self.done_tag, tznow())
        offset = len(done_line_end)
        rom = '^(\\s*)(\\[\\s\\]|.)(\\s*.*)$'
        rdm = '\n            (?x)^(\\s*)(\\[x\\]|.)                           # 0,1 indent & bullet\n            (\\s*[^\\b]*?(?:[^\\@]|(?<!\\s)\\@|\\@(?=\\s))*?\\s*) #   2 very task\n            (?=\n              ((?:\\s@done|@project|@[wl]asted|$).*)   # 3 ending either w/ done or w/o it & no date\n              |                                       #   or\n              (?:[ \\t](\\([^()]*\\))\\s*([^@]*|(?:@project|@[wl]asted).*))?$ # 4 date & possible project tag after\n            )\n            '
        rcm = '^(\\s*)(\\[\\-\\]|.)(\\s*[^\\b]*?(?:[^\\@]|(?<!\\s)\\@|\\@(?=\\s))*?\\s*)(?=((?:\\s@cancelled|@project|@[wl]asted|$).*)|(?:[ \\t](\\([^()]*\\))\\s*([^@]*|(?:@project|@[wl]asted).*))?$)'
        started = '^\\s*[^\\b]*?\\s*@started(\\([\\d\\w,\\.:\\-\\/ @]*\\)).*$'
        toggle = '@toggle(\\([\\d\\w,\\.:\\-\\/ @]*\\))'
        regions = itertools.chain(*(reversed(self.view.lines(region)) for region in reversed(list(self.view.sel()))))
        for line in regions:
            line_contents = self.view.substr(line)
            open_matches = re.match(rom, line_contents, re.U)
            done_matches = re.match(rdm, line_contents, re.U)
            canc_matches = re.match(rcm, line_contents, re.U)
            started_matches = re.findall(started, line_contents, re.U)
            toggle_matches = re.findall(toggle, line_contents, re.U)
            done_line_end = done_line_end.rstrip()
            if line_contents.endswith('  '):
                done_line_end += '  '
                dblspc = '  '
            else:
                dblspc = ''
            current_scope = self.view.scope_name(line.a)
            if 'pending' in current_scope:
                grps = open_matches.groups()
                len_dle = self.view.insert(edit, line.end(), done_line_end)
                replacement = u'%s%s%s' % (grps[0], self.done_tasks_bullet, grps[2].rstrip())
                self.view.replace(edit, line, replacement)
                self.view.run_command('plain_tasks_calculate_time_for_task', {'started_matches': started_matches, 'toggle_matches': toggle_matches, 'now': now, 'eol': line.a + len(replacement) + len_dle})
            elif 'header' in current_scope:
                eol = self.view.insert(edit, line.end(), done_line_end)
                self.view.run_command('plain_tasks_calculate_time_for_task', {'started_matches': started_matches, 'toggle_matches': toggle_matches, 'now': now, 'eol': line.end() + eol})
                indent = re.match('^(\\s*)\\S', line_contents, re.U)
                self.view.insert(edit, line.begin() + len(indent.group(1)), '%s ' % self.done_tasks_bullet)
                self.view.run_command('plain_tasks_calculate_total_time_for_project', {'start': line.a})
            elif 'completed' in current_scope:
                grps = done_matches.groups()
                parentheses = check_parentheses(self.date_format, grps[4] or '')
                replacement = u'%s%s%s%s' % (grps[0], self.open_tasks_bullet, grps[2], parentheses)
                self.view.replace(edit, line, replacement.rstrip() + dblspc)
                offset = -offset
            elif 'cancelled' in current_scope:
                grps = canc_matches.groups()
                len_dle = self.view.insert(edit, line.end(), done_line_end)
                parentheses = check_parentheses(self.date_format, grps[4] or '')
                replacement = u'%s%s%s%s' % (grps[0], self.done_tasks_bullet, grps[2], parentheses)
                self.view.replace(edit, line, replacement.rstrip())
                offset = -offset
                self.view.run_command('plain_tasks_calculate_time_for_task', {'started_matches': started_matches, 'toggle_matches': toggle_matches, 'now': now, 'eol': line.a + len(replacement) + len_dle})
        self.view.sel().clear()
        for (ind, pt) in enumerate(original):
            ofs = ind * offset
            new_pt = sublime.Region(pt.a + ofs, pt.b + ofs)
            self.view.sel().add(new_pt)
        PlainTasksStatsStatus.set_stats(self.view)
        self.view.run_command('plain_tasks_toggle_highlight_past_due')

class PlainTasksInjectDueDateCommand(PlainTasksBase):

    def is_visible(self):
        if False:
            return 10
        return self.view.score_selector(0, 'text.todo') > 0

    def runCommand(self, edit):
        if False:
            return 10
        due_re = '^\\s*[^\\b]*?\\s*@due\\([\\d\\w,\\.:\\-\\/ @]*\\).*$'
        regions = itertools.chain(*(reversed(self.view.lines(region)) for region in reversed(list(self.view.sel()))))
        point = -1
        for line in regions:
            line_contents = self.view.substr(line)
            current_scope = self.view.scope_name(line.begin())
            due_matches = re.match(due_re, line_contents, re.U)
            if ('item' in current_scope or 'header' in current_scope) and (not due_matches):
                self.view.insert(edit, line.end(), ' @due()')
                point = line.end() + 6
        if point != -1:
            self.view.sel().clear()
            self.view.sel().add(point)

class PlainTasksSortByDueDateAndPriorityCommand(PlainTasksBase):

    class Task:
        pass

    def is_visible(self):
        if False:
            print('Hello World!')
        return self.view.score_selector(0, 'text.todo') > 0

    def runCommand(self, edit, descending=False):
        if False:
            for i in range(10):
                print('nop')
        due_re = '^\\s*[^\\b]*?\\s*@due\\(([\\d\\w,\\.:\\-\\/ @]*)\\).*$'
        critical_re = '^\\s*[^\\b]*?\\s*@critical\\b.*$'
        high_re = '^\\s*[^\\b]*?\\s*high\\b.*$'
        low_re = '^\\s*[^\\b]*?\\s*@low\\b.*$'
        regions = itertools.chain(*(reversed(self.view.lines(region)) for region in reversed(list(self.view.sel()))))
        for project in regions:
            project_scope = self.view.scope_name(project.begin())
            if not 'header' in project_scope:
                continue
            project_block = self.view.indented_region(project.end() + 1)
            if project_block.empty():
                continue
            tasks = []
            task = None
            pos = project_block.begin()
            first_task_pos = None
            first_task_indentation = None
            while pos < project_block.end():
                line = self.view.line(pos)
                line_contents = self.view.substr(line)
                scope = self.view.scope_name(pos)
                indentation = self.view.indentation_level(pos)
                if scope == 'text.todo notes.todo ' and task is None:
                    pass
                elif ('item' in scope or 'header' in scope) and (task is None or first_task_indentation == indentation):
                    task = self.Task()
                    task.region = line
                    if first_task_pos is None:
                        first_task_pos = pos
                        first_task_indentation = indentation
                    task.due = '99-01-01 00:00'
                    due_match = re.match(due_re, line_contents, re.U)
                    if due_match is not None:
                        task.due = due_match.groups()[0]
                    task.priority = '3normal'
                    critical_match = re.match(critical_re, line_contents, re.U)
                    high_match = re.match(high_re, line_contents, re.U)
                    low_match = re.match(low_re, line_contents, re.U)
                    if critical_match is not None:
                        task.priority = '1critical'
                    elif high_match is not None:
                        task.priority = '2high'
                    elif low_match is not None:
                        task.priority = '4low'
                    tasks.append(task)
                elif task is not None:
                    task.region = sublime.Region(task.region.begin(), line.end())
                pos = line.end() + 1
            tasks.sort(key=lambda t: (t.due, t.priority), reverse=descending)
            project_block = sublime.Region(first_task_pos, project_block.end())
            new_content = '\n'.join([self.view.substr(t.region).rstrip() for t in tasks]) + '\n'
            self.view.replace(edit, project_block, new_content)
        PlainTasksStatsStatus.set_stats(self.view)
        self.view.run_command('plain_tasks_toggle_highlight_past_due')

class PlainTasksCancelCommand(PlainTasksBase):

    def runCommand(self, edit):
        if False:
            while True:
                i = 10
        original = [r for r in self.view.sel()]
        (canc_line_end, now) = self.format_line_end(self.canc_tag, tznow())
        offset = len(canc_line_end)
        rom = '^(\\s*)(\\[\\s\\]|.)(\\s*.*)$'
        rdm = '^(\\s*)(\\[x\\]|.)(\\s*[^\\b]*?(?:[^\\@]|(?<!\\s)\\@|\\@(?=\\s))*?\\s*)(?=((?:\\s@done|@project|@[wl]asted|$).*)|(?:[ \\t](\\([^()]*\\))\\s*([^@]*|(?:@project|@[wl]asted).*))?$)'
        rcm = '^(\\s*)(\\[\\-\\]|.)(\\s*[^\\b]*?(?:[^\\@]|(?<!\\s)\\@|\\@(?=\\s))*?\\s*)(?=((?:\\s@cancelled|@project|@[wl]asted|$).*)|(?:[ \\t](\\([^()]*\\))\\s*([^@]*|(?:@project|@[wl]asted).*))?$)'
        started = '^\\s*[^\\b]*?\\s*@started(\\([\\d\\w,\\.:\\-\\/ @]*\\)).*$'
        toggle = '@toggle(\\([\\d\\w,\\.:\\-\\/ @]*\\))'
        regions = itertools.chain(*(reversed(self.view.lines(region)) for region in reversed(list(self.view.sel()))))
        for line in regions:
            line_contents = self.view.substr(line)
            open_matches = re.match(rom, line_contents, re.U)
            done_matches = re.match(rdm, line_contents, re.U)
            canc_matches = re.match(rcm, line_contents, re.U)
            started_matches = re.findall(started, line_contents, re.U)
            toggle_matches = re.findall(toggle, line_contents, re.U)
            canc_line_end = canc_line_end.rstrip()
            if line_contents.endswith('  '):
                canc_line_end += '  '
                dblspc = '  '
            else:
                dblspc = ''
            current_scope = self.view.scope_name(line.a)
            if 'pending' in current_scope:
                grps = open_matches.groups()
                len_cle = self.view.insert(edit, line.end(), canc_line_end)
                replacement = u'%s%s%s' % (grps[0], self.canc_tasks_bullet, grps[2].rstrip())
                self.view.replace(edit, line, replacement)
                self.view.run_command('plain_tasks_calculate_time_for_task', {'started_matches': started_matches, 'toggle_matches': toggle_matches, 'now': now, 'eol': line.a + len(replacement) + len_cle, 'tag': 'wasted'})
            elif 'header' in current_scope:
                eol = self.view.insert(edit, line.end(), canc_line_end)
                self.view.run_command('plain_tasks_calculate_time_for_task', {'started_matches': started_matches, 'toggle_matches': toggle_matches, 'now': now, 'eol': line.end() + eol, 'tag': 'wasted'})
                indent = re.match('^(\\s*)\\S', line_contents, re.U)
                self.view.insert(edit, line.begin() + len(indent.group(1)), '%s ' % self.canc_tasks_bullet)
                self.view.run_command('plain_tasks_calculate_total_time_for_project', {'start': line.a})
            elif 'completed' in current_scope:
                sublime.status_message('You cannot cancel what have been done, can you?')
            elif 'cancelled' in current_scope:
                grps = canc_matches.groups()
                parentheses = check_parentheses(self.date_format, grps[4] or '')
                replacement = u'%s%s%s%s' % (grps[0], self.open_tasks_bullet, grps[2], parentheses)
                self.view.replace(edit, line, replacement.rstrip() + dblspc)
                offset = -offset
        self.view.sel().clear()
        for (ind, pt) in enumerate(original):
            ofs = ind * offset
            new_pt = sublime.Region(pt.a + ofs, pt.b + ofs)
            self.view.sel().add(new_pt)
        PlainTasksStatsStatus.set_stats(self.view)
        self.view.run_command('plain_tasks_toggle_highlight_past_due')

class PlainTasksArchiveCommand(PlainTasksBase):

    def runCommand(self, edit, partial=False):
        if False:
            i = 10
            return i + 15
        rds = 'meta.item.todo.completed'
        rcs = 'meta.item.todo.cancelled'
        archive_pos = self.view.find(self.archive_name, 0, sublime.LITERAL)
        if partial:
            all_tasks = self.get_archivable_tasks_within_selections()
        else:
            all_tasks = self.get_all_archivable_tasks(archive_pos, rds, rcs)
        if not all_tasks:
            sublime.status_message('Nothing to archive')
        else:
            if archive_pos and archive_pos.a > 0:
                line = self.view.full_line(archive_pos).end()
            else:
                create_archive = u'\n\n＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿\n%s\n' % self.archive_name
                self.view.insert(edit, self.view.size(), create_archive)
                line = self.view.size()
            projects = get_all_projects_and_separators(self.view)
            for task in all_tasks:
                line_content = self.view.substr(task)
                match_task = re.match('^\\s*(\\[[x-]\\]|.)(\\s+.*$)', line_content, re.U)
                current_scope = self.view.scope_name(task.a)
                if rds in current_scope or rcs in current_scope:
                    pr = self.get_task_project(task, projects)
                    if self.project_postfix:
                        eol = u'{0}{1}{2}{3}\n'.format(self.before_tasks_bullet_spaces, line_content.strip(), u' @project(%s)' % pr if pr else '', '  ' if line_content.endswith('  ') else '')
                    else:
                        eol = u'{0}{1}{2}{3}\n'.format(self.before_tasks_bullet_spaces, match_task.group(1), u'%s%s:' % (self.tasks_bullet_space, pr) if pr else '', match_task.group(2))
                else:
                    eol = u'{0}{1}\n'.format(self.before_tasks_bullet_spaces * 2, line_content.lstrip())
                line += self.view.insert(edit, line, eol)
            for task in reversed(all_tasks):
                self.view.erase(edit, self.view.full_line(task))
            self.view.run_command('plain_tasks_sort_by_date')

    def get_task_project(self, task, projects):
        if False:
            while True:
                i = 10
        index = -1
        for (ind, pr) in enumerate(projects):
            if task < pr:
                if ind > 0:
                    index = ind - 1
                break
        if index == -1:
            return ''
        prog = re.compile('^\\n*(\\s*)(.+):(?=\\s|$)\\s*(\\@[^\\s]+(\\(.*?\\))?\\s*)*')
        hierarhProject = ''
        if index >= 0:
            depth = re.match('\\s*', self.view.substr(self.view.line(task))).group()
            while index >= 0:
                strProject = self.view.substr(projects[index])
                if prog.match(strProject):
                    spaces = prog.match(strProject).group(1)
                    if len(spaces) < len(depth):
                        hierarhProject = prog.match(strProject).group(2) + (' / ' + hierarhProject if hierarhProject else '')
                        depth = spaces
                        if len(depth) == 0:
                            break
                else:
                    sep = re.compile('(^\\s*)---.{3,5}---+$')
                    spaces = sep.match(strProject).group(1)
                    if len(spaces) < len(depth):
                        depth = spaces
                        if len(depth) == 0:
                            break
                index -= 1
        if not hierarhProject:
            return ''
        else:
            return hierarhProject

    def get_task_note(self, task, tasks):
        if False:
            while True:
                i = 10
        note_line = task.end() + 1
        while self.view.scope_name(note_line) == 'text.todo notes.todo ':
            note = self.view.line(note_line)
            if note not in tasks:
                tasks.append(note)
            note_line = self.view.line(note_line).end() + 1

    def get_all_archivable_tasks(self, archive_pos, rds, rcs):
        if False:
            i = 10
            return i + 15
        done_tasks = [i for i in self.view.find_by_selector(rds) if i.a < (archive_pos.a if archive_pos and archive_pos.a > 0 else self.view.size())]
        for i in done_tasks:
            self.get_task_note(i, done_tasks)
        canc_tasks = [i for i in self.view.find_by_selector(rcs) if i.a < (archive_pos.a if archive_pos and archive_pos.a > 0 else self.view.size())]
        for i in canc_tasks:
            self.get_task_note(i, canc_tasks)
        all_tasks = done_tasks + canc_tasks
        all_tasks.sort()
        return all_tasks

    def get_archivable_tasks_within_selections(self):
        if False:
            for i in range(10):
                print('nop')
        all_tasks = []
        for region in self.view.sel():
            for l in self.view.lines(region):
                line = self.view.line(l)
                if 'completed' in self.view.scope_name(line.a) or 'cancelled' in self.view.scope_name(line.a):
                    all_tasks.append(line)
                    self.get_task_note(line, all_tasks)
        return all_tasks

class PlainTasksNewTaskDocCommand(sublime_plugin.WindowCommand):

    def run(self):
        if False:
            print('Hello World!')
        view = self.window.new_file()
        view.settings().add_on_change('color_scheme', lambda : self.set_proper_scheme(view))
        view.set_syntax_file('Packages/PlainTasks/PlainTasks.sublime-syntax' if ST3 else 'Packages/PlainTasks/PlainTasks.tmLanguage')

    def set_proper_scheme(self, view):
        if False:
            print('Hello World!')
        if view.id() != sublime.active_window().active_view().id():
            return
        pts = sublime.load_settings('PlainTasks.sublime-settings')
        if view.settings().get('color_scheme') == pts.get('color_scheme'):
            return
        view.settings().set('color_scheme', pts.get('color_scheme'))

class PlainTasksOpenUrlCommand(sublime_plugin.TextCommand):
    URL_REGEX = '(?i)\\b((?:https?://|www\\d{0,3}[.]|[a-z0-9.\\-]+[.][a-z]{2,4}/)(?:[^\\s()<>]+|\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\))\n        +(?:\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\)|[^\\s`!()\\[\\]{};:\'".,<>?«»“”‘’]))'

    def run(self, edit):
        if False:
            return 10
        s = self.view.sel()[0]
        (start, end) = (s.a, s.b)
        if 'url' in self.view.scope_name(start):
            while self.view.substr(start) != '<':
                start -= 1
            while self.view.substr(end) != '>':
                end += 1
            rgn = sublime.Region(start + 1, end)
            self.view.sel().add(rgn)
            url = self.view.substr(rgn)
            if NT and all([ST3, ':' in url]):
                subprocess.Popen(['start', url], shell=True)
            else:
                webbrowser.open_new_tab(url)
        else:
            self.search_bare_weblink_and_open(start, end)

    def search_bare_weblink_and_open(self, start, end):
        if False:
            while True:
                i = 10
        view_size = self.view.size()
        stopSymbols = ['\t', ' ', '"', "'", '>', '<', ',']
        while start > 0 and (not self.view.substr(start - 1) in stopSymbols) and (self.view.classify(start) & sublime.CLASS_LINE_START == 0):
            start -= 1
        while end < view_size and (not self.view.substr(end) in stopSymbols) and (self.view.classify(end) & sublime.CLASS_LINE_END == 0):
            end += 1
        url = self.view.substr(sublime.Region(start, end))
        self.view.sel().add(sublime.Region(start, end))
        exp = re.search(self.URL_REGEX, url, re.X)
        if exp and exp.group(0):
            strUrl = exp.group(0)
            if strUrl.find('://') == -1:
                strUrl = 'http://' + strUrl
            webbrowser.open_new_tab(strUrl)
        else:
            sublime.status_message('Looks like there is nothing to open')

class PlainTasksOpenLinkCommand(sublime_plugin.TextCommand):
    LINK_PATTERN = re.compile('(?ixu)(?:^|[ \\t])\\.[\\\\/]\n            (?P<fn>\n            (?:[a-z]\\:[\\\\/])?      # special case for Windows full path\n            (?:[^\\\\/:">]+[\\\\/]?)+) # the very path (single filename/relative/full)\n            (?=[\\\\/:">])           # stop matching path\n                                   # options:\n            (>(?P<sym>\\w+))?(\\:(?P<line>\\d+))?(\\:(?P<col>\\d+))?(\\"(?P<text>[^\\n]*)\\")?\n        ')
    MD_LINK = re.compile('(?ixu)\\][ \\t]*\\(\\<?(?:file\\:///?)?\n            (?P<fn>.*?((\\\\\\))?.*?)*)\n              (?:\\>?[ \\t]*\n              \\"((\\:(?P<line>\\d+))?(\\:(?P<col>\\d+))?|(\\>(?P<sym>\\w+))?|(?P<text>[^\\n]*))\n              \\")?\n            \\)\n        ')
    WIKI_LINK = re.compile('(?ixu)\\[\\[(?:file(?:\\+(?:sys|emacs))?\\:)?(?:\\.[\\\\/])?\n            (?P<fn>.*?((\\\\\\])?.*?)*)\n              (?# options for orgmode link [[path::option]])\n              (?:\\:\\:(((?P<line>\\d+))?(\\:(?P<col>\\d+))?|(\\*(?P<sym>\\w+))?|(?P<text>.*?((\\\\\\])?.*?)*)))?\n            \\](?:\\[(.*?)\\])?\n            \\]\n              (?# options for NV [[path]] "option" — NV not support it, but PT should support so it wont break NV)\n              (?:[ \\t]*\n              \\"((\\:(?P<linen>\\d+))?(\\:(?P<coln>\\d+))?|(\\>(?P<symn>\\w+))?|(?P<textn>[^\\n]*))\n              \\")?\n        ')

    def _format_res(self, res):
        if False:
            for i in range(10):
                print('nop')
        if res[3] == 'f':
            return [res[0], 'line: %d column: %d' % (int(res[1]), int(res[2]))]
        elif res[3] == 'd':
            return [res[0], 'Add folder to project' if ST3 else 'Folders are supported only in Sublime 3']
        else:
            return [res[0], res[1]]

    def _on_panel_selection(self, selection, text=None, line=0):
        if False:
            i = 10
            return i + 15
        if selection < 0:
            self.panel_hidden = True
            return
        self.stop_thread = True
        self.thread.join()
        win = sublime.active_window()
        win.run_command('hide_overlay')
        res = self._current_res[selection]
        if not res[3]:
            return
        if not ST3 and res[3] == 'd':
            return sublime.status_message('Folders are supported only in Sublime 3')
        elif res[3] == 'd':
            data = win.project_data()
            if not data:
                data = {}
            if 'folders' not in data:
                data['folders'] = []
            data['folders'].append({'follow_symlinks': True, 'path': res[0]})
            win.set_project_data(data)
        else:
            self.opened_file = win.open_file('%s:%s:%s' % res[:3], sublime.ENCODED_POSITION)
            if text:
                sublime.set_timeout(lambda : self.find_text(self.opened_file, text, line), 300)

    def search_files(self, all_folders, fn, sym, line, col, text):
        if False:
            i = 10
            return i + 15
        'run in separate thread; worker'
        fn = fn.replace('/', os.sep)
        if os.path.isfile(fn):
            self._current_res.append((fn, line, col, 'f'))
        elif os.path.isdir(fn):
            self._current_res.append((fn, 0, 0, 'd'))
        seen_folders = []
        for folder in sorted(set(all_folders)):
            for (root, subdirs, _) in os.walk(folder):
                if self.stop_thread:
                    return
                if root in seen_folders:
                    continue
                else:
                    seen_folders.append(root)
                subdirs = [f for f in subdirs if os.path.join(root, f) not in seen_folders]
                tname = '%s at %s' % (fn, root)
                self.thread.name = tname if ST3 else tname.encode('utf8')
                name = os.path.normpath(os.path.abspath(os.path.join(root, fn)))
                if os.path.isfile(name):
                    item = (name, line, col, 'f')
                    if item not in self._current_res:
                        self._current_res.append(item)
                if os.path.isdir(name):
                    item = (name, 0, 0, 'd')
                    if item not in self._current_res:
                        self._current_res.append(item)
        self._current_res = self._current_res[1:]
        if not self._current_res:
            return sublime.error_message('File was not found\n\n\t%s' % fn)
        if len(self._current_res) == 1:
            sublime.set_timeout(lambda : self._on_panel_selection(0, text=text, line=line), 1)
        else:
            entries = [self._format_res(res) for res in self._current_res]
            sublime.set_timeout(lambda : self.window.show_quick_panel(entries, lambda i: self._on_panel_selection(i, text=text, line=line)), 1)

    def run(self, edit):
        if False:
            print('Hello World!')
        if hasattr(self, 'thread'):
            if self.thread.is_alive:
                self.stop_thread = True
                self.thread.join()
        point = self.view.sel()[0].begin()
        line = self.view.substr(self.view.line(point))
        (fn, sym, line, col, text) = self.parse_link(line)
        if not fn:
            sublime.status_message('Line does not contain a valid link to file')
            return
        self.window = win = sublime.active_window()
        self._current_res = [('Stop search', '', '', '')]
        self.items = 0
        self.panel_hidden = True
        if sym:
            for (name, _, pos) in win.lookup_symbol_in_index(sym):
                if name.endswith(fn):
                    (line, col) = pos
                    self._current_res.append((name, line, col, 'f'))
        all_folders = win.folders() + [os.path.dirname(v.file_name()) for v in win.views() if v.file_name()]
        self.stop_thread = False
        self.thread = threading.Thread(target=self.search_files, args=(all_folders, fn, sym, line, col, text))
        self.thread.setName('is starting')
        self.thread.start()
        self.progress_bar()

    def find_text(self, view, text, line):
        if False:
            print('Hello World!')
        result = view.find(text, view.sel()[0].a if line else 0, sublime.LITERAL)
        view.sel().clear()
        view.sel().add(result.a)
        view.set_viewport_position(view.text_to_layout(view.size()), False)
        view.show_at_center(result)

    def progress_bar(self, i=0, dir=1):
        if False:
            for i in range(10):
                print('nop')
        if not self.thread.is_alive():
            PlainTasksStatsStatus.set_stats(self.view)
            return
        if self._current_res and sublime.active_window().active_view().id() == self.view.id():
            items = len(self._current_res)
            if items != self.items:
                self.window.run_command('hide_overlay')
                self.items = items
            if self.panel_hidden:
                entries = [self._format_res(res) for res in self._current_res]
                self.window.show_quick_panel(entries, self._on_panel_selection)
                self.panel_hidden = False
        before = i % 8
        after = 7 - before
        if not after:
            dir = -1
        if not before:
            dir = 1
        i += dir
        self.view.set_status('PlainTasks', u'Please wait%s…%ssearching %s' % (' ' * before, ' ' * after, self.thread.name if ST3 else self.thread.name.decode('utf8')))
        sublime.set_timeout(lambda : self.progress_bar(i, dir), 100)
        return

    def parse_link(self, line):
        if False:
            return 10
        match_link = self.LINK_PATTERN.search(line)
        match_md = self.MD_LINK.search(line)
        match_wiki = self.WIKI_LINK.search(line)
        if match_link:
            (fn, sym, line, col, text) = match_link.group('fn', 'sym', 'line', 'col', 'text')
        elif match_md:
            (fn, sym, line, col, text) = match_md.group('fn', 'sym', 'line', 'col', 'text')
            fn = fn.replace('\\(', '(').replace('\\)', ')')
        elif match_wiki:
            fn = match_wiki.group('fn')
            sym = match_wiki.group('sym') or match_wiki.group('symn')
            line = match_wiki.group('line') or match_wiki.group('linen')
            col = match_wiki.group('col') or match_wiki.group('coln')
            text = match_wiki.group('text') or match_wiki.group('textn')
            fn = fn.replace('\\[', '[').replace('\\]', ']')
            if text:
                text = text.replace('\\[', '[').replace('\\]', ']')
        return (fn, sym, line or 0, col or 0, text)

class PlainTasksSortByDate(PlainTasksBase):

    def runCommand(self, edit):
        if False:
            print('Hello World!')
        if not re.search('(?su)%[Yy][-./ ]*%m[-./ ]*%d\\s*%H.*%M', self.date_format):
            return
        archive_pos = self.view.find(self.archive_name, 0, sublime.LITERAL)
        if archive_pos:
            have_date = '(^\\s*[^\\n]*?\\s\\@(?:done|cancelled)\\s*(\\([\\d\\w,\\.:\\-\\/ ]*\\))[^\\n]*$)'
            all_tasks_prefixed_date = []
            all_tasks = self.view.find_all(have_date, 0, u'\\2\\1', all_tasks_prefixed_date)
            tasks_prefixed_date = []
            tasks = []
            for (ind, task) in enumerate(all_tasks):
                if task.a > archive_pos.b:
                    tasks.append(task)
                    tasks_prefixed_date.append(all_tasks_prefixed_date[ind])
            notes = []
            for (ind, task) in enumerate(tasks):
                note_line = task.end() + 1
                while self.view.scope_name(note_line) == 'text.todo notes.todo ':
                    note = self.view.line(note_line)
                    notes.append(note)
                    tasks_prefixed_date[ind] += u'\n' + self.view.substr(note)
                    note_line = note.end() + 1
            to_remove = tasks + notes
            to_remove.sort()
            for i in reversed(to_remove):
                self.view.erase(edit, self.view.full_line(i))
            tasks_prefixed_date.sort(reverse=self.view.settings().get('new_on_top', True))
            eol = archive_pos.end()
            for a in tasks_prefixed_date:
                eol += self.view.insert(edit, eol, u'\n' + re.sub('^\\([\\d\\w,\\.:\\-\\/ ]*\\)([^\\b]*$)', u'\\1', a))
        else:
            sublime.status_message('Nothing to sort')

class PlainTasksRemoveBold(sublime_plugin.TextCommand):

    def run(self, edit):
        if False:
            while True:
                i = 10
        for s in reversed(list(self.view.sel())):
            (a, b) = (s.begin(), s.end())
            for r in (sublime.Region(b + 2, b), sublime.Region(a - 2, a)):
                self.view.erase(edit, r)

class PlainTasksStatsStatus(sublime_plugin.EventListener):

    def on_activated(self, view):
        if False:
            return 10
        if not view.score_selector(0, 'text.todo') > 0:
            return
        self.set_stats(view)

    def on_post_save(self, view):
        if False:
            for i in range(10):
                print('nop')
        self.on_activated(view)

    @staticmethod
    def set_stats(view):
        if False:
            while True:
                i = 10
        view.set_status('PlainTasks', PlainTasksStatsStatus.get_stats(view))

    @staticmethod
    def get_stats(view):
        if False:
            print('Hello World!')
        msgf = view.settings().get('stats_format', '$n/$a done ($percent%) $progress Last task @done $last')
        special_interest = re.findall('{{.*?}}', msgf)
        for i in special_interest:
            matches = view.find_all(i.strip('{}'))
            (pend, done, canc) = ([], [], [])
            for t in matches:
                t = view.line(t).a
                scope = view.scope_name(t)
                if 'pending' in scope and t not in pend:
                    pend.append(t)
                elif 'completed' in scope and t not in done:
                    done.append(t)
                elif 'cancelled' in scope and t not in canc:
                    canc.append(t)
            msgf = msgf.replace(i, '%d/%d/%d' % (len(pend), len(done), len(canc)))
        ignore_archive = view.settings().get('stats_ignore_archive', False)
        if ignore_archive:
            archive_pos = view.find(view.settings().get('archive_name', 'Archive:'), 0, sublime.LITERAL)
            pend = len([i for i in view.find_by_selector('meta.item.todo.pending') if i.a < (archive_pos.a if archive_pos and archive_pos.a > 0 else view.size())])
            done = len([i for i in view.find_by_selector('meta.item.todo.completed') if i.a < (archive_pos.a if archive_pos and archive_pos.a > 0 else view.size())])
            canc = len([i for i in view.find_by_selector('meta.item.todo.cancelled') if i.a < (archive_pos.a if archive_pos and archive_pos.a > 0 else view.size())])
        else:
            pend = len(view.find_by_selector('meta.item.todo.pending'))
            done = len(view.find_by_selector('meta.item.todo.completed'))
            canc = len(view.find_by_selector('meta.item.todo.cancelled'))
        allt = pend + done + canc
        percent = (done + canc) / float(allt) * 100 if allt else 0
        factor = int(round(percent / 10)) if percent < 90 else int(percent / 10)
        barfull = view.settings().get('bar_full', u'■')
        barempty = view.settings().get('bar_empty', u'□')
        progress = '%s%s' % (barfull * factor, barempty * (10 - factor)) if factor else ''
        tasks_dates = []
        view.find_all('(^\\s*[^\n]*?\\s\\@(?:done)\\s*(\\([\\d\\w,\\.:\\-\\/ ]*\\))[^\n]*$)', 0, '\\2', tasks_dates)
        date_format = view.settings().get('date_format', '(%y-%m-%d %H:%M)')
        tasks_dates = [check_parentheses(date_format, t, is_date=True) for t in tasks_dates]
        tasks_dates.sort(reverse=True)
        last = tasks_dates[0] if tasks_dates else '(UNKNOWN)'
        msg = msgf.replace('$o', str(pend)).replace('$d', str(done)).replace('$c', str(canc)).replace('$n', str(done + canc)).replace('$a', str(allt)).replace('$percent', str(int(percent))).replace('$progress', progress).replace('$last', last)
        return msg

class PlainTasksCopyStats(sublime_plugin.TextCommand):

    def is_enabled(self):
        if False:
            print('Hello World!')
        return self.view.score_selector(0, 'text.todo') > 0

    def run(self, edit):
        if False:
            while True:
                i = 10
        msg = self.view.get_status('PlainTasks')
        replacements = self.view.settings().get('replace_stats_chars', [])
        if replacements:
            for (o, r) in replacements:
                msg = msg.replace(o, r)
        sublime.set_clipboard(msg)

class PlainTasksArchiveOrgCommand(PlainTasksBase):

    def runCommand(self, edit):
        if False:
            return 10
        archive_filename = self.__createArchiveFilename()
        region = self.__findCurrentSubtree()
        if region.empty():
            sublime.error_message('Error:\n\nCould not find a tree to archive.')
            return
        success = self.__writeArchive(archive_filename, region)
        if success:
            self.view.erase(edit, region)
        return

    def __writeArchive(self, filename, region):
        if False:
            for i in range(10):
                print('nop')
        sublime.status_message(u'Archiving tree to {0}'.format(filename))
        try:
            with io.open(filename, 'a', encoding='utf8') as fh:
                data = self.view.substr(region)
                fh.write(u'--- ✄ -----------------------\n')
                fh.write(u'Archived {0}:\n'.format(tznow().strftime(self.date_format)))
                fh.write(u'{0}\n'.format(data))
            return True
        except Exception as e:
            sublime.error_message(u'Error:\n\nUnable to append to {0}\n{1}'.format(filename, str(e)))
            return False

    def __createArchiveFilename(self):
        if False:
            return 10
        (path_base, extension) = os.path.splitext(self.view.file_name())
        dir = os.path.dirname(path_base)
        base = os.path.basename(path_base)
        sep = os.sep
        try:
            archive_filename = self.archive_org_filemask.format(dir=dir, base=base, ext=extension, sep=sep)
        except:
            archive_filename = self.archive_org_default_filemask.format(dir=dir, base=base, ext=extension, sep=sep)
            sublime.error_message(u'Error:\n\nInvalid filemask:{0}\nUsing default: {1}'.format(self.archive_org_filemask, self.archive_org_default_filemask))
        return archive_filename

    def __findCurrentSubtree(self):
        if False:
            print('Hello World!')
        line = self.view.line(self.view.sel()[0].begin())
        region = self.view.indented_region(line.b + 2)
        if region.contains(line.b):
            return sublime.Region(-1, -1)
        if not region.empty():
            region = sublime.Region(line.a, region.b)
        return region

class PlainTasksFoldToTags(PlainTasksFold):
    TAG = '(?u)@\\w+'

    def run(self, edit):
        if False:
            i = 10
            return i + 15
        tag_sels = [s for s in list(self.view.sel()) if 'tag.todo' in self.view.scope_name(s.a)]
        if not tag_sels:
            sublime.status_message('Cursor(s) must be placed on tag(s)')
            return
        tags = self.extract_tags(tag_sels)
        tasks = [self.view.line(f) for f in self.view.find_all('[ \\t](%s)' % '|'.join(tags)) if 'pending' in self.view.scope_name(f.a)]
        if not tasks:
            sublime.status_message('Pending tasks with given tags are not found')
            print(tags, tag_sels)
            return
        self.exec_folding(self.add_projects_and_notes(tasks))

    def extract_tags(self, tag_sels):
        if False:
            print('Hello World!')
        tags = []
        for s in tag_sels:
            start = end = s.a
            limit = self.view.size()
            while all((self.view.substr(start) != c for c in '@ \n')):
                start -= 1
                if start == 0:
                    break
            while all((self.view.substr(end) != c for c in '( @\n')):
                end += 1
                if end == limit:
                    break
            match = re.match(self.TAG, self.view.substr(sublime.Region(start, end)))
            tag = match.group(0) if match else False
            if tag and tag not in tags:
                tags.append(tag)
        return tags

class PlainTasksAddGutterIconsForTags(sublime_plugin.EventListener):

    def on_activated(self, view):
        if False:
            print('Hello World!')
        if not view.score_selector(0, 'text.todo') > 0:
            return
        view.erase_regions('critical')
        view.erase_regions('high')
        view.erase_regions('low')
        view.erase_regions('today')
        icon_critical = view.settings().get('icon_critical', '')
        icon_high = view.settings().get('icon_high', '')
        icon_low = view.settings().get('icon_low', '')
        icon_today = view.settings().get('icon_today', '')
        if not any((icon_critical, icon_high, icon_low, icon_today)):
            return
        critical = 'string.other.tag.todo.critical'
        high = 'string.other.tag.todo.high'
        low = 'string.other.tag.todo.low'
        today = 'string.other.tag.todo.today'
        r_critical = view.find_by_selector(critical)
        r_high = view.find_by_selector(high)
        r_low = view.find_by_selector(low)
        r_today = view.find_by_selector(today)
        if not any((r_critical, r_high, r_low, r_today)):
            return
        view.add_regions('critical', r_critical, critical, icon_critical, sublime.HIDDEN)
        view.add_regions('high', r_high, high, icon_high, sublime.HIDDEN)
        view.add_regions('low', r_low, low, icon_low, sublime.HIDDEN)
        view.add_regions('today', r_today, today, icon_today, sublime.HIDDEN)

    def on_post_save(self, view):
        if False:
            i = 10
            return i + 15
        self.on_activated(view)

    def on_load(self, view):
        if False:
            while True:
                i = 10
        self.on_activated(view)

class PlainTasksHover(sublime_plugin.ViewEventListener):
    """Show popup with actions when hover over bullet"""
    msg = '<style>html {{{{background-color: color(var(--background) blenda(white 75%))}}}}body {{{{margin: .1em .3em}}}}p {{{{margin: .5em 0}}}}a {{{{text-decoration: none}}}}span.icon {{{{font-weight: bold; font-size: 1.3em}}}}#icon-done {{{{color: var(--greenish)}}}}#icon-cancel {{{{color: var(--redish)}}}}#icon-archive {{{{color: var(--bluish)}}}}#icon-outside {{{{color: var(--purplish)}}}}#done {{{{color: var(--greenish)}}}}#cancel {{{{color: var(--redish)}}}}#archive {{{{color: var(--bluish)}}}}#outside {{{{color: var(--purplish)}}}}</style><body>{actions}'
    complete = '<a href="complete\x0b{point}"><span class="icon" id="icon-done">✔</span> <span id="done">Toggle complete</span></a>'
    cancel = '<a href="cancel\x0b{point}"><span class="icon" id="icon-cancel">✘</span> <span id="cancel">Toggle cancel</span></a>'
    archive = '<a href="archive\x0b{point}"><span class="icon" id="icon-archive">📚</span> <span id="archive">Archive</span></a>'
    archivetofile = '<a href="tofile\x0b{point}"><span class="icon" id="icon-outside">📤</span> <span id="outside">Archive to file</span></a>'
    actions = {'text.todo meta.item.todo.pending': '<p>{complete}</p><p>{cancel}</p>'.format(complete=complete, cancel=cancel), 'text.todo meta.item.todo.completed': '<p>{archive}</p><p>{archivetofile}</p><p>{complete}</p>'.format(archive=archive, archivetofile=archivetofile, complete=complete), 'text.todo meta.item.todo.cancelled': '<p>{archive}</p><p>{archivetofile}</p><p>{complete}</p><p>{cancel}</p>'.format(archive=archive, archivetofile=archivetofile, complete=complete, cancel=cancel)}

    @classmethod
    def is_applicable(cls, settings):
        if False:
            for i in range(10):
                print('nop')
        return settings.get('syntax') == 'Packages/PlainTasks/PlainTasks.sublime-syntax'

    def on_hover(self, point, hover_zone):
        if False:
            i = 10
            return i + 15
        self.view.hide_popup()
        if hover_zone != sublime.HOVER_TEXT:
            return
        line = self.view.line(point)
        line_scope_name = self.view.scope_name(line.a).strip()
        if 'meta.item.todo' not in line_scope_name:
            return
        bullet = any(('bullet' in self.view.scope_name(p) for p in (point, point - 1)))
        if not bullet:
            return
        (width, height) = self.view.viewport_extent()
        self.view.show_popup(self.msg.format(actions=self.actions.get(line_scope_name)).format(point=point), 0, point or self.view.sel()[0].begin() or 1, width, height / 2, self.exec_action)

    def exec_action(self, msg):
        if False:
            for i in range(10):
                print('nop')
        (action, at) = msg.split('\x0b')
        case = {'complete': lambda : self.view.run_command('plain_tasks_complete'), 'cancel': lambda : self.view.run_command('plain_tasks_cancel'), 'archive': lambda : self.view.run_command('plain_tasks_archive', {'partial': True}), 'tofile': lambda : self.view.run_command('plain_tasks_org_archive')}
        self.view.sel().clear()
        self.view.sel().add(sublime.Region(int(at)))
        case[action]()
        self.view.hide_popup()

class PlainTasksGotoTag(sublime_plugin.TextCommand):

    def run(self, edit):
        if False:
            print('Hello World!')
        self.initial_viewport = self.view.viewport_position()
        self.initial_sels = list(self.view.sel())
        self.tags = sorted([r for r in self.view.find_by_selector('meta.tag.todo') if not any((s in self.view.scope_name(r.a) for s in ('completed', 'cancelled')))] + self.view.find_by_selector('string.other.tag.todo.critical') + self.view.find_by_selector('string.other.tag.todo.high') + self.view.find_by_selector('string.other.tag.todo.low') + self.view.find_by_selector('string.other.tag.todo.today'))
        window = self.view.window() or sublime.active_window()
        items = [[self.view.substr(t), u'{0}: {1}'.format(self.view.rowcol(t.a)[0], self.view.substr(self.view.line(t)).strip())] for t in self.tags]
        if ST3:
            from bisect import bisect_left
            closest_index = bisect_left([r.a for r in self.tags], self.view.layout_to_text(self.initial_viewport))
            llen = len(self.tags)
            selected_index = closest_index if closest_index < llen else llen - 1
            window.show_quick_panel(items, self.on_done, 0, selected_index, self.on_highlighted)
        else:
            window.show_quick_panel(items, self.on_done)

    def on_done(self, index):
        if False:
            return 10
        if index < 0:
            self.view.sel().clear()
            self.view.sel().add_all(self.initial_sels)
            self.view.set_viewport_position(self.initial_viewport)
            return
        self.view.sel().clear()
        self.view.sel().add(sublime.Region(self.tags[index].a))
        self.view.show_at_center(self.tags[index])

    def on_highlighted(self, index):
        if False:
            return 10
        self.view.sel().clear()
        self.view.sel().add(self.tags[index])
        self.view.show(self.tags[index], True)