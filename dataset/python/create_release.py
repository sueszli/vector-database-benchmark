#!/usr/bin/env python
# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create a release using Github API"""
import os

import requests


def create_release():
    """Create a release from the Git Tag"""

    tag = os.getenv("GIT_TAG")
    access_token = os.getenv("GH_TOKEN")

    if not tag:
        raise Exception("Unable to retrieve GIT_TAG environment variable")

    url = "https://api.github.com/repos/streamlit/streamlit/releases"
    header = {"Authorization": f"token {access_token}"}

    # Get the latest release tag to compare against
    response = requests.get(f"{url}/latest", headers=header)
    previous_tag_name = None
    if response.status_code == 200:
        previous_tag_name = response.json()["tag_name"]
    else:
        raise Exception(f"Unable get the latest release: {response.text}")

    # Generate the automated release notes
    payload = {"tag_name": tag, "previous_tag_name": previous_tag_name}
    response = requests.post(f"{url}/generate-notes", json=payload, headers=header)
    body = None
    if response.status_code == 200:
        body = response.json()["body"]
    else:
        raise Exception(f"Unable generate the latest release notes: {response.text}")

    # Create the release with the generated release notes
    payload = {"tag_name": tag, "name": tag, "body": body}
    response = requests.post(url, json=payload, headers=header)

    if response.status_code == 201:
        print(f"Successfully created Release {tag}")
    else:
        raise Exception(f"Unable to create release, HTTP response: {response.text}")


def main():
    create_release()


if __name__ == "__main__":
    main()
