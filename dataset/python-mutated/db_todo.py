from h2o_wave import main, app, Q, ui, connect, WaveDB, expando_to_dict

class TodoItem:

    def __init__(self, id, label, done):
        if False:
            i = 10
            return i + 15
        self.id = id
        self.label = label
        self.done = done

async def setup_db() -> WaveDB:
    db = connect()['todo']
    (_, err) = await db.exec_atomic('\n        create table if not exists todo (\n            id integer primary key,\n            user text not null,\n            label text not null,\n            done integer not null default 0\n        )\n        ')
    if err:
        raise RuntimeError(f'Failed setting up database: {err}')
    return db

@app('/demo')
async def serve(q: Q):
    if q.app.db is None:
        q.app.db = await setup_db()
    if q.args.new_todo:
        await new_todo(q)
    elif q.args.add_todo:
        await add_todo(q)
    else:
        await show_todos(q)

async def show_todos(q: Q):
    db: WaveDB = q.app.db
    updates = []
    for (key, done) in expando_to_dict(q.args).items():
        if key.startswith('todo_'):
            (_, id) = key.split('_', 1)
            updates.append(('update todo set done=? where id=?', 1 if done else 0, int(id)))
    if len(updates):
        (_, err) = await db.exec_many(*updates)
        if err:
            raise RuntimeError(f'Failed updating todos: {err}')
    (rows, err) = await db.exec('select id, label, done from todo where user=?', q.auth.subject)
    if err:
        raise RuntimeError(f'Failed fetching todos: {err}')
    todos = [TodoItem(id, label, done) for (id, label, done) in rows]
    done = [ui.checkbox(name=f'todo_{todo.id}', label=todo.label, value=True, trigger=True) for todo in todos if todo.done]
    not_done = [ui.checkbox(name=f'todo_{todo.id}', label=todo.label, trigger=True) for todo in todos if not todo.done]
    q.page['form'] = ui.form_card(box='1 1 4 10', items=[ui.text_l('To Do'), ui.button(name='new_todo', label='Add To Do...', primary=True), *not_done, *([ui.separator('Done')] if len(done) else []), *done])
    await q.page.save()

async def add_todo(q: Q):
    db: WaveDB = q.app.db
    (_, err) = await db.exec('insert into todo (user, label) values (? , ?)', q.auth.subject, q.args.label or 'Untitled')
    if err:
        raise RuntimeError(f'Failed inserting todo: {err}')
    await show_todos(q)

async def new_todo(q: Q):
    q.page['form'] = ui.form_card(box='1 1 4 10', items=[ui.text_l('Add To Do'), ui.textbox(name='label', label='What needs to be done?', multiline=True), ui.buttons([ui.button(name='add_todo', label='Add', primary=True), ui.button(name='show_todos', label='Back')])])
    await q.page.save()