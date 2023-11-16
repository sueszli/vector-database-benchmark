# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-mgmt-costmanagement
# USAGE
    python scheduled_actioncreate_or_updateprivate.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = CostManagementClient(
        credential=DefaultAzureCredential(),
    )

    response = client.scheduled_actions.create_or_update(
        name="monthlyCostByResource",
        scheduled_action={
            "kind": "Email",
            "properties": {
                "displayName": "Monthly Cost By Resource",
                "notification": {"subject": "Cost by resource this month", "to": ["user@gmail.com", "team@gmail.com"]},
                "schedule": {
                    "daysOfWeek": ["Monday"],
                    "endDate": "2021-06-19T22:21:51.1287144Z",
                    "frequency": "Monthly",
                    "hourOfDay": 10,
                    "startDate": "2020-06-19T22:21:51.1287144Z",
                    "weeksOfMonth": ["First", "Third"],
                },
                "status": "Enabled",
                "viewId": "/providers/Microsoft.CostManagement/views/swaggerExample",
            },
        },
    )
    print(response)


# x-ms-original-file: specification/cost-management/resource-manager/Microsoft.CostManagement/stable/2022-10-01/examples/scheduledActions/scheduledAction-createOrUpdate-private.json
if __name__ == "__main__":
    main()
