from posthog.async_migrations.definition import AsyncMigrationDefinition, AsyncMigrationOperation

def raise_exception_fn(_):
    if False:
        while True:
            i = 10
    raise Exception('Test rollback Exception')

class Migration(AsyncMigrationDefinition):
    description = "Another example async migration that's less realistic and used in tests."
    operations = [AsyncMigrationOperation(fn=lambda _: None), AsyncMigrationOperation(fn=lambda _: None, rollback_fn=raise_exception_fn), AsyncMigrationOperation(fn=lambda _: None)]