# Copyright 2019 Google LLC
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

# [START translate_v3_batch_translate_text_with_glossary_and_model]
from google.cloud import translate


def batch_translate_text_with_glossary_and_model(
    input_uri: str,
    output_uri: str,
    project_id: str,
    model_id: str,
    glossary_id: str,
) -> translate.TranslateTextResponse:
    """Batch translate text with Glossary and Translation model.
    Args:
        input_uri: The input text to be translated.
        output_uri: The output text to be translated.
        project_id: The ID of the GCP project that owns the model.
        model_id: The ID of the model
        glossary_id: The ID of the glossary

    Returns:
        The translated text.
    """

    client = translate.TranslationServiceClient()

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    location = "us-central1"

    target_language_codes = ["ja"]
    gcs_source = {"input_uri": input_uri}

    # Optional. Can be "text/plain" or "text/html".
    mime_type = "text/plain"
    input_configs_element = {"gcs_source": gcs_source, "mime_type": mime_type}
    input_configs = [input_configs_element]
    gcs_destination = {"output_uri_prefix": output_uri}
    output_config = {"gcs_destination": gcs_destination}
    parent = f"projects/{project_id}/locations/{location}"
    model_path = "projects/{}/locations/{}/models/{}".format(
        project_id, "us-central1", model_id
    )
    models = {"ja": model_path}

    glossary_path = client.glossary_path(
        project_id, "us-central1", glossary_id  # The location of the glossary
    )

    glossary_config = translate.TranslateTextGlossaryConfig(glossary=glossary_path)
    glossaries = {"ja": glossary_config}  # target lang as key

    operation = client.batch_translate_text(
        request={
            "parent": parent,
            "source_language_code": "en",
            "target_language_codes": target_language_codes,
            "input_configs": input_configs,
            "output_config": output_config,
            "models": models,
            "glossaries": glossaries,
        }
    )

    print("Waiting for operation to complete...")
    response = operation.result()

    # Display the translation for each input text provided
    print(f"Total Characters: {response.total_characters}")
    print(f"Translated Characters: {response.translated_characters}")

    return response


# [END translate_v3_batch_translate_text_with_glossary_and_model]
