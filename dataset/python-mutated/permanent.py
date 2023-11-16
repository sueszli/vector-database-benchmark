from sentry.features.base import FeatureHandlerStrategy, OrganizationFeature, ProjectFeature
from sentry.features.manager import FeatureManager

def register_permanent_features(manager: FeatureManager):
    if False:
        i = 10
        return i + 15
    '\n    These flags are permanent.\n\n    Organization Features that are part of sentry.io subscription plans\n    Features here should ideally be enabled sentry/conf/server.py so that\n    self-hosted and single-tenant are aligned with sentry.io. Features here should\n    also be listed in SubscriptionPlanFeatureHandler in getsentry so that sentry.io\n    behaves correctly.\n    '
    permanent_organization_features = ['organizations:advanced-search', 'organizations:app-store-connect-multiple', 'organizations:change-alerts', 'organizations:commit-context', 'organizations:codecov-integration', 'organizations:crash-rate-alerts', 'organizations:custom-symbol-sources', 'organizations:dashboards-basic', 'organizations:dashboards-edit', 'organizations:data-forwarding', 'organizations:discover-basic', 'organizations:discover-query', 'organizations:dynamic-sampling', 'organizations:event-attachments', 'organizations:global-views', 'organizations:incidents', 'organizations:integrations-alert-rule', 'organizations:integrations-chat-unfurl', 'organizations:integrations-codeowners', 'organizations:integrations-event-hooks', 'organizations:integrations-enterprise-alert-rule', 'organizations:integrations-enterprise-incident-management', 'organizations:integrations-incident-management', 'organizations:integrations-issue-basic', 'organizations:integrations-issue-sync', 'organizations:integrations-ticket-rules', 'organizations:performance-view', 'organizations:profiling-view', 'organizations:relay', 'organizations:session-replay', 'organizations:sso-basic', 'organizations:sso-saml2', 'organizations:team-insights', 'organizations:team-roles', 'organizations:on-demand-metrics-prefill', 'organizations:custom-metrics']
    permanent_project_features = ['projects:data-forwarding', 'projects:rate-limits', 'projects:custom-inbound-filters', 'projects:discard-groups', 'projects:servicehooks']
    for org_feature in permanent_organization_features:
        manager.add(org_feature, OrganizationFeature, FeatureHandlerStrategy.INTERNAL)
    for project_feature in permanent_project_features:
        manager.add(project_feature, ProjectFeature, FeatureHandlerStrategy.INTERNAL)