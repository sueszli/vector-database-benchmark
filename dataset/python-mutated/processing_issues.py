from collections import defaultdict
from django.db.models import Count, Max
from sentry.api.serializers import serialize
from sentry.models.processingissue import ProcessingIssue
from sentry.models.reprocessingreport import ReprocessingReport

def get_processing_issues(user, projects, include_detailed_issues=False):
    if False:
        while True:
            i = 10
    "\n    Given a list of projects, returns a list containing stats about processing\n    issues for those projects\n    :param include_detailed_issues: Include specific details on each processing\n    issue\n    :return: A list of dicts, with each dict containing keys:\n        - 'hasIssues': Whether the project has any processing issues\n        - 'numIssues': How many processing issues the project has\n        - 'lastSeen': The date a processing issue was last seen\n        - 'resolveableIssues': How many Raw Events have no remaining issues and\n        can be resolved automatically\n        - 'hasMoreResolveableIssues': Whether there are any Raw Events that\n        have no remaining issues and can be resolved automatically\n        'issuesProcessing': How many ReprocessingReports exist for this Project\n        'project': Slug for the project\n\n    "
    project_agg_results = {result['project']: result for result in ProcessingIssue.objects.filter(project__in=projects).values('project').annotate(num_issues=Count('id'), last_seen=Max('datetime'))}
    project_reprocessing_issues = {result['project']: result['reprocessing_issues'] for result in ReprocessingReport.objects.filter(project__in=projects).values('project').annotate(reprocessing_issues=Count('id'))}
    resolved_qs = ProcessingIssue.objects.find_resolved_queryset([p.id for p in projects])
    project_resolveable = {result['project']: result['count'] for result in resolved_qs.values('project').annotate(count=Count('id'))}
    project_issues = defaultdict(list)
    if include_detailed_issues:
        for proc_issue in ProcessingIssue.objects.with_num_events().filter(project__in=projects).order_by('type', 'datetime'):
            project_issues[proc_issue.project_id].append(proc_issue)
    project_results = []
    for project in projects:
        agg_results = project_agg_results.get(project.id, {})
        num_issues = agg_results.get('num_issues', 0)
        last_seen = agg_results.get('last_seen')
        data = {'hasIssues': num_issues > 0, 'numIssues': num_issues, 'lastSeen': last_seen and serialize(last_seen) or None, 'resolveableIssues': project_resolveable.get(project.id, 0), 'hasMoreResolveableIssues': False, 'issuesProcessing': project_reprocessing_issues.get(project.id, 0), 'project': project.slug}
        if include_detailed_issues:
            issues = project_issues[project.id]
            data['issues'] = [serialize(issue, user) for issue in issues]
        project_results.append(data)
    return project_results