"""myapp.py

Usage::

   # The worker service reacts to messages by executing tasks.
   (window1)$ python myapp.py worker -l INFO

   # The beat service sends messages at scheduled intervals.
   (window2)$ python myapp.py beat -l INFO

   # XXX To diagnose problems use -l debug:
   (window2)$ python myapp.py beat -l debug

   # XXX XXX To diagnose calculated runtimes use C_REMDEBUG envvar:
   (window2) $ C_REMDEBUG=1 python myapp.py beat -l debug


You can also specify the app to use with the `celery` command,
using the `-A` / `--app` option::

    $ celery -A myapp worker -l INFO

With the `-A myproj` argument the program will search for an app
instance in the module ``myproj``.  You can also specify an explicit
name using the fully qualified form::

    $ celery -A myapp:app worker -l INFO

"""
from celery import Celery
app = Celery('myapp', broker='amqp://guest@localhost//')
app.conf.timezone = 'UTC'

@app.task
def say(what):
    if False:
        return 10
    print(what)

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    if False:
        print('Hello World!')
    sender.add_periodic_task(10.0, say.s('hello'), name='add every 10')
if __name__ == '__main__':
    app.start()