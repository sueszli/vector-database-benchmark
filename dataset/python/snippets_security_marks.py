#!/usr/bin/env python
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Demos for working with security marks."""


def add_to_asset(asset_name):
    """Add new security marks to an asset."""
    # [START securitycenter_add_security_marks]
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2

    # Create a new client.
    client = securitycenter.SecurityCenterClient()

    # 'asset_name' is the resource path for an asset that exists in SCC.
    # Specify the value of 'asset_name' in one of the following formats:
    #   f"organizations/{org_id}/assets/{asset_id}"
    #   f"projects/{project_id}/assets/{asset_id}"
    #   f"folders/{folder_id}/assets/{asset_id}"
    # asset_name = organizations/123123342/assets/12312321
    marks_name = f"{asset_name}/securityMarks"

    # Notice the suffix after "marks." in the field mask matches the keys
    # in marks.
    field_mask = field_mask_pb2.FieldMask(paths=["marks.key_a", "marks.key_b"])
    marks = {"key_a": "value_a", "key_b": "value_b"}

    updated_marks = client.update_security_marks(
        request={
            "security_marks": {"name": marks_name, "marks": marks},
            "update_mask": field_mask,
        }
    )
    print(updated_marks)
    # [END securitycenter_add_security_marks]
    return updated_marks, marks


def clear_from_asset(asset_name):
    """Removes security marks from an asset."""
    # Make sure they are there first
    add_to_asset(asset_name)
    # [START securitycenter_delete_security_marks]
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2

    # Create a new client.
    client = securitycenter.SecurityCenterClient()

    # 'asset_name' is the resource path for an asset that exists in SCC.
    # Specify the value of 'asset_name' in one of the following formats:
    #   f"organizations/{org_id}/assets/{asset_id}"
    #   f"projects/{project_id}/assets/{asset_id}"
    #   f"folders/{folder_id}/assets/{asset_id}"
    # asset_name = organizations/123123342/assets/12312321
    marks_name = f"{asset_name}/securityMarks"

    field_mask = field_mask_pb2.FieldMask(paths=["marks.key_a", "marks.key_b"])

    updated_marks = client.update_security_marks(
        request={
            "security_marks": {
                "name": marks_name
                # Note, no marks specified, so the specified values in
                # the fields masks will be deleted.
            },
            "update_mask": field_mask,
        }
    )
    print(updated_marks)
    # [END securitycenter_delete_security_marks]
    return updated_marks


def delete_and_update_marks(asset_name):
    """Updates and deletes security marks from an asset in the same call."""
    # Make sure they are there first
    add_to_asset(asset_name)
    # [START securitycenter_add_delete_security_marks]
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2

    client = securitycenter.SecurityCenterClient()
    # 'asset_name' is the resource path for an asset that exists in SCC.
    # Specify the value of 'asset_name' in one of the following formats:
    #   f"organizations/{org_id}/assets/{asset_id}"
    #   f"projects/{project_id}/assets/{asset_id}"
    #   f"folders/{folder_id}/assets/{asset_id}"
    # asset_name = organizations/123123342/assets/12312321
    marks_name = f"{asset_name}/securityMarks"

    field_mask = field_mask_pb2.FieldMask(paths=["marks.key_a", "marks.key_b"])
    marks = {"key_a": "new_value_for_a"}

    updated_marks = client.update_security_marks(
        request={
            "security_marks": {"name": marks_name, "marks": marks},
            "update_mask": field_mask,
        }
    )
    print(updated_marks)
    # [END securitycenter_add_delete_security_marks]
    return updated_marks


def add_to_finding(finding_name):
    """Adds security marks to a finding."""
    # [START securitycenter_add_finding_security_marks]
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2

    client = securitycenter.SecurityCenterClient()
    # 'finding_name' is the resource path for a finding that exists in SCC.
    # Specify the value of 'asset_name' in one of the following formats:
    #   f"organizations/{org_id}/assets/{asset_id}"
    #   f"projects/{project_id}/assets/{asset_id}"
    #   f"folders/{folder_id}/assets/{asset_id}"
    # finding_name = "organizations/1112/sources/1234/findings/findingid"
    finding_marks_name = f"{finding_name}/securityMarks"

    # Notice the suffix after "marks." in the field mask matches the keys
    # in marks.
    field_mask = field_mask_pb2.FieldMask(
        paths=["marks.finding_key_a", "marks.finding_key_b"]
    )
    marks = {"finding_key_a": "value_a", "finding_key_b": "value_b"}

    updated_marks = client.update_security_marks(
        request={
            "security_marks": {"name": finding_marks_name, "marks": marks},
            "update_mask": field_mask,
        }
    )
    # [END securitycenter_add_finding_security_marks]
    return updated_marks, marks


def list_assets_with_query_marks(organization_id, asset_name):
    """Lists assets with a filter on security marks."""
    add_to_asset(asset_name)
    i = -1
    # [START securitycenter_list_assets_with_security_marks]
    from google.cloud import securitycenter

    client = securitycenter.SecurityCenterClient()

    # 'parent' must be in one of the following formats:
    #   "organizations/{organization_id}"
    #   "projects/{project_id}"
    #   "folders/{folder_id}"
    parent = f"organizations/{organization_id}"

    marks_filter = 'security_marks.marks.key_a = "value_a"'
    # Call the API and print results.
    asset_iterator = client.list_assets(
        request={"parent": parent, "filter": marks_filter}
    )
    for i, asset_result in enumerate(asset_iterator):
        print(i, asset_result)
    # [END securitycenter_list_assets_with_security_marks]
    return i


def list_findings_with_query_marks(source_name, finding_name):
    """Lists findings with a filter on security marks."""
    # ensure marks are set on finding.
    add_to_finding(finding_name)
    i = -1
    # [START securitycenter_list_findings_with_security_marks]
    from google.cloud import securitycenter

    client = securitycenter.SecurityCenterClient()

    # 'source_name' is the resource path for a source that has been
    # created previously (you can use list_sources to find a specific one).
    # Its format is:
    # source_name = "{parent}/sources/{source_id}"
    # 'parent' must be in one of the following formats:
    #   "organizations/{organization_id}"
    #   "projects/{project_id}"
    #   "folders/{folder_id}"
    # source_name = "organizations/111122222444/sources/1234"
    marks_filter = 'NOT security_marks.marks.finding_key_a="value_a"'

    # Call the API and print results.
    finding_iterator = client.list_findings(
        request={"parent": source_name, "filter": marks_filter}
    )
    for i, finding_result in enumerate(finding_iterator):
        print(i, finding_result)
    # [END securitycenter_list_findings_with_security_marks]
    # one finding should have been updated with keys, and one should be
    # untouched.
    return i
