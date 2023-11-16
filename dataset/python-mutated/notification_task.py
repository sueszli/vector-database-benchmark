import json
import uuid
from plane.db.models import IssueMention, IssueSubscriber, Project, User, IssueAssignee, Issue, Notification, IssueComment, IssueActivity
from celery import shared_task
from bs4 import BeautifulSoup

def update_mentions_for_issue(issue, project, new_mentions, removed_mention):
    if False:
        i = 10
        return i + 15
    aggregated_issue_mentions = []
    for mention_id in new_mentions:
        aggregated_issue_mentions.append(IssueMention(mention_id=mention_id, issue=issue, project=project, workspace_id=project.workspace_id))
    IssueMention.objects.bulk_create(aggregated_issue_mentions, batch_size=100)
    IssueMention.objects.filter(issue=issue, mention__in=removed_mention).delete()

def get_new_mentions(requested_instance, current_instance):
    if False:
        while True:
            i = 10
    mentions_older = extract_mentions(current_instance)
    mentions_newer = extract_mentions(requested_instance)
    new_mentions = [mention for mention in mentions_newer if mention not in mentions_older]
    return new_mentions

def get_removed_mentions(requested_instance, current_instance):
    if False:
        return 10
    mentions_older = extract_mentions(current_instance)
    mentions_newer = extract_mentions(requested_instance)
    removed_mentions = [mention for mention in mentions_older if mention not in mentions_newer]
    return removed_mentions

def extract_mentions_as_subscribers(project_id, issue_id, mentions):
    if False:
        return 10
    bulk_mention_subscribers = []
    for mention_id in mentions:
        if not IssueSubscriber.objects.filter(issue_id=issue_id, subscriber_id=mention_id, project_id=project_id).exists() and (not IssueAssignee.objects.filter(project_id=project_id, issue_id=issue_id, assignee_id=mention_id).exists()) and (not Issue.objects.filter(project_id=project_id, pk=issue_id, created_by_id=mention_id).exists()):
            project = Project.objects.get(pk=project_id)
            bulk_mention_subscribers.append(IssueSubscriber(workspace_id=project.workspace_id, project_id=project_id, issue_id=issue_id, subscriber_id=mention_id))
    return bulk_mention_subscribers

def extract_mentions(issue_instance):
    if False:
        while True:
            i = 10
    try:
        mentions = []
        data = json.loads(issue_instance)
        html = data.get('description_html')
        soup = BeautifulSoup(html, 'html.parser')
        mention_tags = soup.find_all('mention-component', attrs={'target': 'users'})
        mentions = [mention_tag['id'] for mention_tag in mention_tags]
        return list(set(mentions))
    except Exception as e:
        return []

def extract_comment_mentions(comment_value):
    if False:
        print('Hello World!')
    try:
        mentions = []
        soup = BeautifulSoup(comment_value, 'html.parser')
        mentions_tags = soup.find_all('mention-component', attrs={'target': 'users'})
        for mention_tag in mentions_tags:
            mentions.append(mention_tag['id'])
        return list(set(mentions))
    except Exception as e:
        return []

def get_new_comment_mentions(new_value, old_value):
    if False:
        for i in range(10):
            print('nop')
    mentions_newer = extract_comment_mentions(new_value)
    if old_value is None:
        return mentions_newer
    mentions_older = extract_comment_mentions(old_value)
    new_mentions = [mention for mention in mentions_newer if mention not in mentions_older]
    return new_mentions

def createMentionNotification(project, notification_comment, issue, actor_id, mention_id, issue_id, activity):
    if False:
        print('Hello World!')
    return Notification(workspace=project.workspace, sender='in_app:issue_activities:mentioned', triggered_by_id=actor_id, receiver_id=mention_id, entity_identifier=issue_id, entity_name='issue', project=project, message=notification_comment, data={'issue': {'id': str(issue_id), 'name': str(issue.name), 'identifier': str(issue.project.identifier), 'sequence_id': issue.sequence_id, 'state_name': issue.state.name, 'state_group': issue.state.group}, 'issue_activity': {'id': str(activity.get('id')), 'verb': str(activity.get('verb')), 'field': str(activity.get('field')), 'actor': str(activity.get('actor_id')), 'new_value': str(activity.get('new_value')), 'old_value': str(activity.get('old_value'))}})

@shared_task
def notifications(type, issue_id, project_id, actor_id, subscriber, issue_activities_created, requested_data, current_instance):
    if False:
        print('Hello World!')
    issue_activities_created = json.loads(issue_activities_created) if issue_activities_created is not None else None
    if type not in ['cycle.activity.created', 'cycle.activity.deleted', 'module.activity.created', 'module.activity.deleted', 'issue_reaction.activity.created', 'issue_reaction.activity.deleted', 'comment_reaction.activity.created', 'comment_reaction.activity.deleted', 'issue_vote.activity.created', 'issue_vote.activity.deleted', 'issue_draft.activity.created', 'issue_draft.activity.updated', 'issue_draft.activity.deleted']:
        bulk_notifications = []
        '\n        Mention Tasks\n        1. Perform Diffing and Extract the mentions, that mention notification needs to be sent\n        2. From the latest set of mentions, extract the users which are not a subscribers & make them subscribers\n        '
        new_mentions = get_new_mentions(requested_instance=requested_data, current_instance=current_instance)
        removed_mention = get_removed_mentions(requested_instance=requested_data, current_instance=current_instance)
        comment_mentions = []
        all_comment_mentions = []
        requested_mentions = extract_mentions(issue_instance=requested_data)
        mention_subscribers = extract_mentions_as_subscribers(project_id=project_id, issue_id=issue_id, mentions=requested_mentions)
        for issue_activity in issue_activities_created:
            issue_comment = issue_activity.get('issue_comment')
            issue_comment_new_value = issue_activity.get('new_value')
            issue_comment_old_value = issue_activity.get('old_value')
            if issue_comment is not None:
                all_comment_mentions = all_comment_mentions + extract_comment_mentions(issue_comment_new_value)
                new_comment_mentions = get_new_comment_mentions(old_value=issue_comment_old_value, new_value=issue_comment_new_value)
                comment_mentions = comment_mentions + new_comment_mentions
        comment_mention_subscribers = extract_mentions_as_subscribers(project_id=project_id, issue_id=issue_id, mentions=all_comment_mentions)
        '\n        We will not send subscription activity notification to the below mentioned user sets\n        - Those who have been newly mentioned in the issue description, we will send mention notification to them.\n        - When the activity is a comment_created and there exist a mention in the comment, then we have to send the "mention_in_comment" notification\n        - When the activity is a comment_updated and there exist a mention change, then also we have to send the "mention_in_comment" notification\n        '
        issue_assignees = list(IssueAssignee.objects.filter(project_id=project_id, issue_id=issue_id).exclude(assignee_id__in=list(new_mentions + comment_mentions)).values_list('assignee', flat=True))
        issue_subscribers = list(IssueSubscriber.objects.filter(project_id=project_id, issue_id=issue_id).exclude(subscriber_id__in=list(new_mentions + comment_mentions + [actor_id])).values_list('subscriber', flat=True))
        issue = Issue.objects.filter(pk=issue_id).first()
        if issue.created_by_id is not None and str(issue.created_by_id) != str(actor_id):
            issue_subscribers = issue_subscribers + [issue.created_by_id]
        if subscriber:
            try:
                if str(issue.created_by_id) != str(actor_id) and uuid.UUID(actor_id) not in issue_assignees:
                    _ = IssueSubscriber.objects.get_or_create(project_id=project_id, issue_id=issue_id, subscriber_id=actor_id)
            except Exception as e:
                pass
        project = Project.objects.get(pk=project_id)
        issue_subscribers = list(set(issue_subscribers + issue_assignees) - {uuid.UUID(actor_id)})
        for subscriber in issue_subscribers:
            if subscriber in issue_subscribers:
                sender = 'in_app:issue_activities:subscribed'
            if issue.created_by_id is not None and subscriber == issue.created_by_id:
                sender = 'in_app:issue_activities:created'
            if subscriber in issue_assignees:
                sender = 'in_app:issue_activities:assigned'
            for issue_activity in issue_activities_created:
                issue_comment = issue_activity.get('issue_comment')
                if issue_comment is not None:
                    issue_comment = IssueComment.objects.get(id=issue_comment, issue_id=issue_id, project_id=project_id, workspace_id=project.workspace_id)
                bulk_notifications.append(Notification(workspace=project.workspace, sender=sender, triggered_by_id=actor_id, receiver_id=subscriber, entity_identifier=issue_id, entity_name='issue', project=project, title=issue_activity.get('comment'), data={'issue': {'id': str(issue_id), 'name': str(issue.name), 'identifier': str(issue.project.identifier), 'sequence_id': issue.sequence_id, 'state_name': issue.state.name, 'state_group': issue.state.group}, 'issue_activity': {'id': str(issue_activity.get('id')), 'verb': str(issue_activity.get('verb')), 'field': str(issue_activity.get('field')), 'actor': str(issue_activity.get('actor_id')), 'new_value': str(issue_activity.get('new_value')), 'old_value': str(issue_activity.get('old_value')), 'issue_comment': str(issue_comment.comment_stripped if issue_activity.get('issue_comment') is not None else '')}}))
        IssueSubscriber.objects.bulk_create(mention_subscribers + comment_mention_subscribers, batch_size=100)
        last_activity = IssueActivity.objects.filter(issue_id=issue_id).order_by('-created_at').first()
        actor = User.objects.get(pk=actor_id)
        for mention_id in comment_mentions:
            if mention_id != actor_id:
                for issue_activity in issue_activities_created:
                    notification = createMentionNotification(project=project, issue=issue, notification_comment=f'{actor.display_name} has mentioned you in a comment in issue {issue.name}', actor_id=actor_id, mention_id=mention_id, issue_id=issue_id, activity=issue_activity)
                    bulk_notifications.append(notification)
        for mention_id in new_mentions:
            if mention_id != actor_id:
                if last_activity is not None and last_activity.field == 'description' and (actor_id == str(last_activity.actor_id)):
                    bulk_notifications.append(Notification(workspace=project.workspace, sender='in_app:issue_activities:mentioned', triggered_by_id=actor_id, receiver_id=mention_id, entity_identifier=issue_id, entity_name='issue', project=project, message=f'You have been mentioned in the issue {issue.name}', data={'issue': {'id': str(issue_id), 'name': str(issue.name), 'identifier': str(issue.project.identifier), 'sequence_id': issue.sequence_id, 'state_name': issue.state.name, 'state_group': issue.state.group}, 'issue_activity': {'id': str(last_activity.id), 'verb': str(last_activity.verb), 'field': str(last_activity.field), 'actor': str(last_activity.actor_id), 'new_value': str(last_activity.new_value), 'old_value': str(last_activity.old_value)}}))
                else:
                    for issue_activity in issue_activities_created:
                        notification = createMentionNotification(project=project, issue=issue, notification_comment=f'You have been mentioned in the issue {issue.name}', actor_id=actor_id, mention_id=mention_id, issue_id=issue_id, activity=issue_activity)
                        bulk_notifications.append(notification)
        update_mentions_for_issue(issue=issue, project=project, new_mentions=new_mentions, removed_mention=removed_mention)
        Notification.objects.bulk_create(bulk_notifications, batch_size=100)