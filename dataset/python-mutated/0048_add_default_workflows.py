from django.db import migrations
from django.db.models import Count, Q
from wagtail.models import Page as RealPage

def ancestor_of_q(page):
    if False:
        print('Hello World!')
    paths = [page.path[0:pos] for pos in range(0, len(page.path) + 1, page.steplen)[1:]]
    q = Q(path__in=paths)
    return q

def create_default_workflows(apps, schema_editor):
    if False:
        return 10
    ContentType = apps.get_model('contenttypes.ContentType')
    Workflow = apps.get_model('wagtailcore.Workflow')
    GroupApprovalTask = apps.get_model('wagtailcore.GroupApprovalTask')
    GroupPagePermission = apps.get_model('wagtailcore.GroupPagePermission')
    WorkflowPage = apps.get_model('wagtailcore.WorkflowPage')
    WorkflowTask = apps.get_model('wagtailcore.WorkflowTask')
    Page = apps.get_model('wagtailcore.Page')
    Group = apps.get_model('auth.Group')
    Page.steplen = RealPage.steplen
    (group_approval_content_type, __) = ContentType.objects.get_or_create(model='groupapprovaltask', app_label='wagtailcore')
    publish_permissions = GroupPagePermission.objects.filter(permission_type='publish')
    for permission in publish_permissions:
        page = permission.page
        page = Page.objects.get(pk=page.pk)
        ancestors = Page.objects.filter(ancestor_of_q(page))
        ancestor_permissions = publish_permissions.filter(page__in=ancestors)
        groups = Group.objects.filter(Q(page_permissions__in=ancestor_permissions) | Q(page_permissions__pk=permission.pk)).distinct()
        task = GroupApprovalTask.objects.filter(groups__id__in=groups.all()).annotate(count=Count('groups')).filter(count=groups.count()).filter(active=True).first()
        if not task:
            group_names = ' '.join([group.name for group in groups])
            task = GroupApprovalTask.objects.create(name=group_names + ' approval', content_type=group_approval_content_type, active=True)
            task.groups.set(groups)
        workflow = Workflow.objects.annotate(task_number=Count('workflow_tasks')).filter(task_number=1).filter(workflow_tasks__task=task).filter(active=True).first()
        if not workflow:
            workflow = Workflow.objects.create(name=task.name, active=True)
            WorkflowTask.objects.create(workflow=workflow, task=task, sort_order=0)
        if not WorkflowPage.objects.filter(workflow=workflow, page=page).exists():
            WorkflowPage.objects.create(workflow=workflow, page=page)

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0047_add_workflow_models')]
    operations = [migrations.RunPython(create_default_workflows, migrations.RunPython.noop)]