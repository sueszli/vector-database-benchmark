# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START translate_v3_delete_glossary]
from google.cloud import translate_v3 as translate


def delete_glossary(
    project_id: str = "YOUR_PROJECT_ID",
    glossary_id: str = "YOUR_GLOSSARY_ID",
    timeout: int = 180,
) -> translate.Glossary:
    """Delete a specific glossary based on the glossary ID.

    Args:
        project_id: The ID of the GCP project that owns the glossary.
        glossary_id: The ID of the glossary to delete.
        timeout: The timeout for this request.

    Returns:
        The glossary that was deleted.
    """
    client = translate.TranslationServiceClient()

    name = client.glossary_path(project_id, "us-central1", glossary_id)

    operation = client.delete_glossary(name=name)
    result = operation.result(timeout)
    print(f"Deleted: {result.name}")

    return result


# [END translate_v3_delete_glossary]
