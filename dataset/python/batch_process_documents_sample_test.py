# Copyright 2020 Google LLC
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
#

# flake8: noqa

import os
from uuid import uuid4

from documentai.snippets import batch_process_documents_sample

location = "us"
project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
processor_id = "90484cfdedb024f6"
processor_version_id = "pretrained-form-parser-v1.0-2020-09-23"
gcs_input_uri = "gs://cloud-samples-data/documentai/invoice.pdf"
gcs_input_prefix = "gs://cloud-samples-data/documentai/workflows/"
input_mime_type = "application/pdf"
gcs_output_uri = f"gs://document-ai-python/{uuid4()}/"
field_mask = "text,pages.pageNumber"


def test_batch_process_documents(capsys):
    batch_process_documents_sample.batch_process_documents(
        project_id=project_id,
        location=location,
        processor_id=processor_id,
        gcs_input_uri=gcs_input_uri,
        input_mime_type=input_mime_type,
        gcs_output_uri=gcs_output_uri,
        field_mask=field_mask,
    )
    out, _ = capsys.readouterr()

    assert "operation" in out
    assert "Fetching" in out
    assert "text:" in out


def test_batch_process_documents_processor_version(capsys):
    batch_process_documents_sample.batch_process_documents(
        project_id=project_id,
        location=location,
        processor_id=processor_id,
        processor_version_id=processor_version_id,
        gcs_input_uri=gcs_input_uri,
        input_mime_type=input_mime_type,
        gcs_output_uri=gcs_output_uri,
        field_mask=field_mask,
    )
    out, _ = capsys.readouterr()

    assert "operation" in out
    assert "Fetching" in out
    assert "text:" in out


def test_batch_process_documents_gcs_prefix(capsys):
    batch_process_documents_sample.batch_process_documents(
        project_id=project_id,
        location=location,
        processor_id=processor_id,
        gcs_input_prefix=gcs_input_prefix,
        gcs_output_uri=gcs_output_uri,
        field_mask=field_mask,
    )
    out, _ = capsys.readouterr()

    assert "operation" in out
    assert "Fetching" in out
    assert "text:" in out
