from celery import Celery
app = Celery()

@app.task
def plugh():
    if False:
        i = 10
        return i + 15
    'This task is in a different module!'