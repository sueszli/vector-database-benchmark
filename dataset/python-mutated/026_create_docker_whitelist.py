import peewee as pw
SCHEMA_VERSION = 26

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15

    @migrator.create_model
    class DockerWhitelist(pw.Model):
        repository = pw.CharField(primary_key=True)

        class Meta:
            db_table = 'dockerwhitelist'
    migrator.sql("INSERT INTO dockerwhitelist(repository) VALUES ('golemfactory')")

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15
    migrator.remove_model('dockerwhitelist')