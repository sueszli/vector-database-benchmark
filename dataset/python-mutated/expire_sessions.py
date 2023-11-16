from importlib import import_module
from django.utils import timezone
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User
from django.contrib.sessions.models import Session
from django.core.exceptions import ObjectDoesNotExist

class Command(BaseCommand):
    """Expire Django auth sessions for a user/all users"""
    help = 'Expire Django auth sessions. Will expire all auth sessions if --user option is not supplied.'

    def add_arguments(self, parser):
        if False:
            return 10
        parser.add_argument('--user', dest='user', type=str)

    def handle(self, *args, **options):
        if False:
            return 10
        try:
            user = User.objects.get(username=options['user']) if options['user'] else None
        except ObjectDoesNotExist:
            raise CommandError('The user does not exist.')
        start = timezone.now()
        sessions = Session.objects.filter(expire_date__gte=start).iterator()
        for session in sessions:
            user_id = session.get_decoded().get('_auth_user_id')
            if user is None or (user_id and user.id == int(user_id)):
                session = import_module(settings.SESSION_ENGINE).SessionStore(session.session_key)
                session.flush()