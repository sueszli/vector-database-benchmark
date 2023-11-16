from django.conf import settings
from environments.models import Environment
from organisations.models import Organisation, OrganisationRole, Subscription
from projects.models import Project
from users.models import FFAdminUser
PASSWORD = 'str0ngp4ssw0rd!'

def delete_user_and_its_organisations(user_email: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    user: FFAdminUser | None = FFAdminUser.objects.filter(email=user_email).first()
    if user:
        user.organisations.all().delete()
        user.delete()

def teardown() -> None:
    if False:
        i = 10
        return i + 15
    delete_user_and_its_organisations(user_email=settings.E2E_SIGNUP_USER)
    delete_user_and_its_organisations(user_email=settings.E2E_USER)
    delete_user_and_its_organisations(user_email=settings.E2E_CHANGE_EMAIL_USER)

def seed_data() -> None:
    if False:
        print('Hello World!')
    organisation: Organisation = Organisation.objects.create(name='Bullet Train Ltd')
    org_admin: FFAdminUser = FFAdminUser.objects.create_user(email=settings.E2E_USER, password=PASSWORD, username=settings.E2E_USER)
    org_admin.add_organisation(organisation, OrganisationRole.ADMIN)
    project: Project = Project.objects.create(name='My Test Project', organisation=organisation)
    Environment.objects.create(name='Development', project=project)
    Environment.objects.create(name='Production', project=project)
    Subscription.objects.filter(organisation__in=org_admin.organisations.all()).update(max_seats=2)