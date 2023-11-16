import json
import re
import uuid
import boto3
from core.constants import STRING
from django.contrib.contenttypes.models import ContentType
from django.core.management import call_command
from django.core.serializers.json import DjangoJSONEncoder
from flag_engine.segments.constants import ALL_RULE
from moto import mock_s3
from environments.models import Environment, EnvironmentAPIKey, Webhook
from features.feature_types import MULTIVARIATE
from features.models import Feature, FeatureSegment, FeatureState
from features.multivariate.models import MultivariateFeatureOption
from features.workflows.core.models import ChangeRequest
from import_export.export import S3OrganisationExporter, export_environments, export_features, export_metadata, export_organisation, export_projects
from integrations.amplitude.models import AmplitudeConfiguration
from integrations.datadog.models import DataDogConfiguration
from integrations.heap.models import HeapConfiguration
from integrations.mixpanel.models import MixpanelConfiguration
from integrations.new_relic.models import NewRelicConfiguration
from integrations.rudderstack.models import RudderstackConfiguration
from integrations.segment.models import SegmentConfiguration
from integrations.slack.models import SlackConfiguration, SlackEnvironment
from integrations.webhook.models import WebhookConfiguration
from metadata.models import Metadata, MetadataField, MetadataModelField, MetadataModelFieldRequirement
from organisations.invites.models import InviteLink
from organisations.models import Organisation, OrganisationWebhook
from projects.models import Project
from projects.tags.models import Tag
from segments.models import EQUAL, Condition, Segment, SegmentRule

def test_export_organisation(db):
    if False:
        i = 10
        return i + 15
    organisation_name = 'test org'
    organisation = Organisation.objects.create(name=organisation_name)
    InviteLink.objects.create(organisation=organisation)
    OrganisationWebhook.objects.create(organisation=organisation, url='https://test.webhooks.com/')
    export = export_organisation(organisation.id)
    assert export

def test_export_project(organisation):
    if False:
        for i in range(10):
            print('nop')
    project_name = 'test project'
    project = Project.objects.create(organisation=organisation, name=project_name)
    segment = Segment.objects.create(project=project, name='test segment')
    segment_rule = SegmentRule.objects.create(segment=segment, type=ALL_RULE)
    Condition(rule=segment_rule, operator=EQUAL, property='foo', value='bar')
    Tag.objects.create(label='tag', project=project, color='#000000')
    DataDogConfiguration.objects.create(project=project, api_key='api-key')
    NewRelicConfiguration.objects.create(project=project, api_key='api-key')
    SlackConfiguration.objects.create(project=project, api_token='api-token')
    export = export_projects(organisation.id)
    assert export

def test_export_environments(project):
    if False:
        return 10
    environment_name = 'test environment'
    environment = Environment.objects.create(project=project, name=environment_name)
    EnvironmentAPIKey.objects.create(environment=environment)
    Webhook.objects.create(environment=environment, url='https://test.webhook.com')
    AmplitudeConfiguration.objects.create(environment=environment, api_key='api-key')
    HeapConfiguration.objects.create(environment=environment, api_key='api-key')
    MixpanelConfiguration.objects.create(environment=environment, api_key='api-key')
    SegmentConfiguration.objects.create(environment=environment, api_key='api-key')
    RudderstackConfiguration.objects.create(environment=environment, api_key='api-key')
    WebhookConfiguration.objects.create(environment=environment, url='https://test.webhook.com')
    slack_project_config = SlackConfiguration.objects.create(project=project, api_token='api-token')
    SlackEnvironment.objects.create(environment=environment, slack_configuration=slack_project_config, channel_id='channel-id')
    export = export_environments(project.organisation_id)
    assert export

def test_export_metadata(environment, organisation, settings):
    if False:
        return 10
    environment_type = ContentType.objects.get_for_model(environment)
    metadata_field = MetadataField.objects.create(name='test_field', type='int', organisation=organisation)
    environment_metadata_field = MetadataModelField.objects.create(field=metadata_field, content_type=environment_type)
    required_for_project = MetadataModelFieldRequirement.objects.create(model_field=environment_metadata_field, content_object=environment.project)
    environment_metadata = Metadata.objects.create(object_id=environment.id, content_type=environment_type, model_field=environment_metadata_field, field_value='some_data')
    exported_environment = export_environments(environment.project.organisation_id)
    exported_metadata = export_metadata(organisation.id)
    data = exported_environment + exported_metadata
    metadata_field.delete()
    environment.hard_delete()
    file_path = f'/tmp/{uuid.uuid4()}.json'
    with open(file_path, 'a+') as f:
        f.write(json.dumps(data, cls=DjangoJSONEncoder))
        f.seek(0)
        call_command('loaddata', f.name, format='json')
    assert MetadataField.objects.filter(uuid=metadata_field.uuid)
    metadata_model_field = MetadataModelField.objects.get(uuid=environment_metadata_field.uuid)
    requrired_for_project = MetadataModelFieldRequirement.objects.get(uuid=required_for_project.uuid)
    assert metadata_model_field == requrired_for_project.model_field
    assert requrired_for_project.content_type.model == 'project'
    metadata = Metadata.objects.get(uuid=environment_metadata.uuid)
    loaded_environment = Environment.objects.get(api_key=environment.api_key)
    assert metadata.content_object == loaded_environment

def test_export_features(project, environment, segment, admin_user):
    if False:
        for i in range(10):
            print('nop')
    standard_feature = Feature.objects.create(project=project, name='standard_feature')
    standard_feature.owners.add(admin_user)
    mv_feature = Feature.objects.create(project=project, name='mv_feature', type=MULTIVARIATE)
    MultivariateFeatureOption.objects.create(feature=mv_feature, default_percentage_allocation=10, type=STRING, string_value='foo')
    feature_segment = FeatureSegment.objects.create(feature=standard_feature, segment=segment, environment=environment)
    FeatureState.objects.create(feature=standard_feature, feature_segment=feature_segment, environment=environment)
    cr = ChangeRequest.objects.create(environment=environment, title='Test CR', user=admin_user)
    FeatureState.objects.create(feature=standard_feature, environment=environment, version=2, change_request=cr)
    export = export_features(organisation_id=project.organisation_id)
    assert export
    json_export = json.dumps(export, cls=DjangoJSONEncoder)
    assert 'owners' not in json_export
    assert 'workflows_core.changerequest' not in json_export
    assert not re.findall('\\"change_request\\": \\[\\"[a-z0-9\\-]{36}\\"\\]', json_export)

@mock_s3
def test_organisation_exporter_export_to_s3(organisation):
    if False:
        while True:
            i = 10
    bucket_name = 'test-bucket'
    file_key = 'organisation-exports/org-1.json'
    s3_resource = boto3.resource('s3', region_name='eu-west-2')
    s3_resource.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': 'eu-west-2'})
    s3_client = boto3.client('s3')
    exporter = S3OrganisationExporter(s3_client=s3_client)
    exporter.export_to_s3(organisation.id, bucket_name, file_key)
    retrieved_object = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    assert retrieved_object.get('ContentLength', 0) > 0