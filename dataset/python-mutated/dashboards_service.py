import proto
from google.monitoring.dashboard_v1.types import dashboard as gmd_dashboard
__protobuf__ = proto.module(package='google.monitoring.dashboard.v1', manifest={'CreateDashboardRequest', 'ListDashboardsRequest', 'ListDashboardsResponse', 'GetDashboardRequest', 'DeleteDashboardRequest', 'UpdateDashboardRequest'})

class CreateDashboardRequest(proto.Message):
    """The ``CreateDashboard`` request.

    Attributes:
        parent (str):
            Required. The project on which to execute the request. The
            format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]

            The ``[PROJECT_ID_OR_NUMBER]`` must match the dashboard
            resource name.
        dashboard (~.gmd_dashboard.Dashboard):
            Required. The initial dashboard
            specification.
    """
    parent = proto.Field(proto.STRING, number=1)
    dashboard = proto.Field(proto.MESSAGE, number=2, message=gmd_dashboard.Dashboard)

class ListDashboardsRequest(proto.Message):
    """The ``ListDashboards`` request.

    Attributes:
        parent (str):
            Required. The scope of the dashboards to list. The format
            is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        page_size (int):
            A positive number that is the maximum number
            of results to return. If unspecified, a default
            of 1000 is used.
        page_token (str):
            If this field is not empty then it must contain the
            ``nextPageToken`` value returned by a previous call to this
            method. Using this field causes the method to return
            additional results from the previous method call.
    """
    parent = proto.Field(proto.STRING, number=1)
    page_size = proto.Field(proto.INT32, number=2)
    page_token = proto.Field(proto.STRING, number=3)

class ListDashboardsResponse(proto.Message):
    """The ``ListDashboards`` request.

    Attributes:
        dashboards (Sequence[~.gmd_dashboard.Dashboard]):
            The list of requested dashboards.
        next_page_token (str):
            If there are more results than have been returned, then this
            field is set to a non-empty value. To see the additional
            results, use that value as ``page_token`` in the next call
            to this method.
    """

    @property
    def raw_page(self):
        if False:
            print('Hello World!')
        return self
    dashboards = proto.RepeatedField(proto.MESSAGE, number=1, message=gmd_dashboard.Dashboard)
    next_page_token = proto.Field(proto.STRING, number=2)

class GetDashboardRequest(proto.Message):
    """The ``GetDashboard`` request.

    Attributes:
        name (str):
            Required. The resource name of the Dashboard. The format is
            one of:

            -  ``dashboards/[DASHBOARD_ID]`` (for system dashboards)
            -  ``projects/[PROJECT_ID_OR_NUMBER]/dashboards/[DASHBOARD_ID]``
               (for custom dashboards).
    """
    name = proto.Field(proto.STRING, number=1)

class DeleteDashboardRequest(proto.Message):
    """The ``DeleteDashboard`` request.

    Attributes:
        name (str):
            Required. The resource name of the Dashboard. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/dashboards/[DASHBOARD_ID]
    """
    name = proto.Field(proto.STRING, number=1)

class UpdateDashboardRequest(proto.Message):
    """The ``UpdateDashboard`` request.

    Attributes:
        dashboard (~.gmd_dashboard.Dashboard):
            Required. The dashboard that will replace the
            existing dashboard.
    """
    dashboard = proto.Field(proto.MESSAGE, number=1, message=gmd_dashboard.Dashboard)
__all__ = tuple(sorted(__protobuf__.manifest))