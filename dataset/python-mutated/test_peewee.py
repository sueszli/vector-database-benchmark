from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['peewee'])
def test_peewee(selenium):
    if False:
        print('Hello World!')
    import os
    from peewee import CharField, IntegerField, Model, SqliteDatabase
    db = SqliteDatabase(os.path.join('/tmp', 'database.db'))

    class Person(Model):
        name = CharField()
        age = IntegerField()

        class Meta:
            database = db
    with db:
        db.create_tables([Person])
        person = Person.create(name='John Doe', age=25)
        people = Person.select()
        assert person in people
        person.age = 30
        person.save()
        person.delete_instance()
        assert person not in Person.select()