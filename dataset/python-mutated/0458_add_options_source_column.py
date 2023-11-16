from enum import Enum
from django.db import migrations, models
from sentry.new_migrations.migrations import CheckedMigration

class UpdateChannel(Enum):
    UNKNOWN = 'unknown'
    APPLICATION = 'application'
    ADMIN = 'admin'
    AUTOMATOR = 'automator'
    CLI = 'cli'
    KILLSWITCH = 'killswitch'

    @classmethod
    def choices(cls):
        if False:
            i = 10
            return i + 15
        return [(i.name, i.value) for i in cls]

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0457_sentry_monitorcheckin_date_added_index')]
    operations = [migrations.SeparateDatabaseAndState(database_operations=[migrations.RunSQL('\n                    ALTER TABLE "sentry_option" ADD COLUMN "last_updated_by" VARCHAR(16) NOT NULL DEFAULT \'unknown\';\n                    ', reverse_sql='\n                    ALTER TABLE "sentry_option" DROP COLUMN "last_updated_by";\n                    ', hints={'tables': ['sentry_option']}), migrations.RunSQL('\n                    ALTER TABLE "sentry_controloption" ADD COLUMN "last_updated_by" VARCHAR(16) NOT NULL DEFAULT \'unknown\';\n                    ', reverse_sql='\n                    ALTER TABLE "sentry_controloption" DROP COLUMN "last_updated_by";\n                    ', hints={'tables': ['sentry_controloption']})], state_operations=[migrations.AddField(model_name='option', name='last_updated_by', field=models.CharField(default=UpdateChannel.UNKNOWN.value, max_length=16, choices=UpdateChannel.choices())), migrations.AddField(model_name='controloption', name='last_updated_by', field=models.CharField(default=UpdateChannel.UNKNOWN.value, max_length=16, choices=UpdateChannel.choices()))])]