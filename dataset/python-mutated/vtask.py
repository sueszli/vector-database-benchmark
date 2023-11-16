"""Full terminal interface for TaskWarrior (task)."""
import tasklib
from visidata import vd, launchExternalEditorValue, Sheet, ColumnItem, date, vlen, CellColorizer, Column, run
from visidata import *
vd.options.disp_date_fmt = '%Y-%m-%d %H:%M'
vd.option('color_task_changed', 'reverse yellow', 'color when vtask is changed')

def editTask(task):
    if False:
        for i in range(10):
            print('nop')
    taskdesc = ''
    for k in 'description project status'.split():
        taskdesc += '%s: %s\n' % (k, task[k])
    for note in task['annotations']:
        taskdesc += '\n---\n'
        taskdesc += note.description
    taskdesc += '\n---\n'
    ret = launchExternalEditorValue(taskdesc)
    newnotes = ret.split('\n---\n')
    task['annotations'] = newnotes[1:]
    for line in newnotes[0].splitlines():
        (k, v) = line.split(': ', maxsplit=1)
        task[k] = v

class TodoSheet(Sheet):
    rowtype = 'tasks'
    columns = [ColumnItem('id', type=int, width=4), ColumnItem('project'), ColumnItem('description'), ColumnItem('status'), ColumnItem('urgency', type=float, fmtstr='{:.01f}'), ColumnItem('start', type=date), ColumnItem('due', type=date), ColumnItem('wait', type=date, width=0), ColumnItem('scheduled', type=date, width=0), ColumnItem('until', type=date, width=0), ColumnItem('entry', type=date, width=0), ColumnItem('modified', type=date, width=0), ColumnItem('completed', type=date, width=0), Column('tags', getter=lambda c, r: ' '.join(r['tags']), setter=lambda c, r: r['tags'].tags.split(' ')), ColumnItem('annotations', type=vlen)]
    nKeys = 1
    colorizers = Sheet.colorizers + [CellColorizer(8, 'color_task_changed', lambda s, c, r, v: r and c and isChanged(r, c.name))]

    def newRow(self, **kwargs):
        if False:
            print('Hello World!')
        return tasklib.Task(self.tw, **kwargs)

    def reload(self):
        if False:
            print('Hello World!')
        self.tw = tasklib.TaskWarrior(data_location=str(self.source), create=True)
        self.rows = list(self.tw.tasks.pending())
        self.orderBy(None, self.column('urgency'), reverse=True)

def isChanged(r, key):
    if False:
        return 10
    return r._data.get(key, None) != r._original_data.get(key, None)

class TaskAnnotationsSheet(Sheet):
    rowtype = 'notes'
    columns = [ColumnItem('entry', type=date), ColumnItem('description')]

    def reload(self):
        if False:
            while True:
                i = 10
        self.rows = self.source['annotations']
TodoSheet.addCommand('^O', 'edit-notes', 'editTask(cursorRow)')
TodoSheet.addCommand('a', 'add-task', 't=newRow(description=input("new task: ")); rows.insert(cursorRowIndex+1, t); t.save(); cursorDown()')
TodoSheet.addCommand('d', 'complete-task', 'cursorRow.done(); cursorRow.refresh()')
TodoSheet.addCommand('gd', 'complete-tasks', 'for r in selectedRows: r.done() or r.refresh()')
TodoSheet.addCommand('zd', 'delete-task', 'cursorRow.delete(); cursorRow.refresh()')
TodoSheet.addCommand('gzd', 'delete-tasks', 'for r in selectedRows: r.delete() or r.refresh()')
TodoSheet.addCommand('z^R', 'refresh-tasks', 'cursorRow.refresh()')
TodoSheet.addCommand('z^S', 'save-task', 'cursorRow.save()')
TodoSheet.addCommand('^S', 'save-modified-tasks', 'list(r.save() for r in rows if r.modified)')
TodoSheet.addCommand(' ', 'start-task', 'cursorRow.stop() if cursorRow["start"] else cursorRow.start()')
TodoSheet.addCommand(ENTER, '', 'vd.push(TaskAnnotationsSheet("cursorRow.description", source=cursorRow))')
TaskAnnotationsSheet.addCommand('a', 'add-task-note', 'source.add_annotation(input("note: ")); reload()')
TaskAnnotationsSheet.addCommand('d', 'delete-task-note', 'source.remove_annotation(cursorRow); reload()')

def main_vtask():
    if False:
        while True:
            i = 10
    run(TodoSheet('todos', source=Path('~/.task')))
vd.addGlobals(globals())