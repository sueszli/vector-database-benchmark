from django.contrib.auth.management.commands.createsuperuser import Command as DjangoCommand


class Command(DjangoCommand):
    help = "Performs any pending database migrations and upgrades"

    def handle(self, **options):
        from sentry.runner import call_command

        call_command("sentry.runner.commands.createuser.createuser", superuser=True)
