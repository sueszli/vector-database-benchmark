import typing
import pytest
from django.db.utils import IntegrityError
from organisations.invites.exceptions import InviteLinksDisabledError
from organisations.invites.models import Invite, InviteLink
from organisations.models import Organisation
if typing.TYPE_CHECKING:
    from django.core.mail import EmailMessage
    from pytest_django.fixtures import SettingsWrapper

@pytest.mark.django_db
def test_cannot_create_invite_link_if_disabled(settings: 'SettingsWrapper') -> None:
    if False:
        while True:
            i = 10
    settings.DISABLE_INVITE_LINKS = True
    with pytest.raises(InviteLinksDisabledError):
        InviteLink.objects.create()

@pytest.mark.django_db
def test_save_invalid_invite__dont_send(mailoutbox: 'list[EmailMessage]') -> None:
    if False:
        print('Hello World!')
    email = 'unknown@test.com'
    organisation = Organisation.objects.create(name='ssg')
    invite = Invite(email=email, organisation=organisation)
    invite.save()
    invalid_invite = Invite(email=email, organisation=organisation)
    with pytest.raises(IntegrityError):
        invalid_invite.save()
    assert len(mailoutbox) == 1