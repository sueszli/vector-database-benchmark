from sentry.silo import SiloMode
from sentry.tasks.base import instrumented_task
from sentry.utils.safe import safe_execute

@instrumented_task(name='sentry.tasks.user_report', silo_mode=SiloMode.REGION)
def user_report(report, project_id):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create and send a UserReport.\n\n    :param report: Serialized `UserReport` object from the DB\n    :param project_id: The user's project's ID\n    "
    from sentry.mail import mail_adapter
    from sentry.models.project import Project
    project = Project.objects.get_from_cache(id=project_id)
    safe_execute(mail_adapter.handle_user_report, report=report, project=project)