# Copyright 2023 Google LLC
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


# [START contentwarehouse_search_documents]

from google.cloud import contentwarehouse

# TODO(developer): Uncomment these variables before running the sample.
# project_number = 'YOUR_PROJECT_NUMBER'
# location = 'YOUR_PROJECT_LOCATION' # Format is 'us' or 'eu'
# document_query_text = 'YOUR_DOCUMENT_QUERY'
# user_id = 'user:YOUR_SERVICE_ACCOUNT_ID' # Format is "user:xxxx@example.com"


def search_documents_sample(
    project_number: str, location: str, document_query_text: str, user_id: str
) -> None:
    # Create a client
    client = contentwarehouse.DocumentServiceClient()

    # The full resource name of the location, e.g.:
    # projects/{project_number}/locations/{location}
    parent = client.common_location_path(project=project_number, location=location)

    # File Type Filter
    # Options: DOCUMENT, FOLDER
    file_type_filter = contentwarehouse.FileTypeFilter(
        file_type=contentwarehouse.FileTypeFilter.FileType.DOCUMENT
    )

    # Document Text Query
    document_query = contentwarehouse.DocumentQuery(
        query=document_query_text,
        file_type_filter=file_type_filter,
    )

    # Histogram Query
    histogram_query = contentwarehouse.HistogramQuery(
        histogram_query='count("DocumentSchemaId")'
    )

    request_metadata = contentwarehouse.RequestMetadata(
        user_info=contentwarehouse.UserInfo(id=user_id)
    )

    # Define request
    request = contentwarehouse.SearchDocumentsRequest(
        parent=parent,
        request_metadata=request_metadata,
        document_query=document_query,
        histogram_queries=[histogram_query],
    )

    # Make the request
    response = client.search_documents(request=request)

    # Print search results
    for matching_document in response.matching_documents:
        document = matching_document.document
        # Display name - schema display name.
        # Name.
        # Create date.
        # Snippet - keywords are highlighted with <b> & </b>.
        print(
            f"{document.display_name} - {document.document_schema_name}\n"
            f"{document.name}\n"
            f"{document.create_time}\n"
            f"{matching_document.search_text_snippet}\n"
        )

    # Print histogram
    for histogram_query_result in response.histogram_query_results:
        print(
            f"Histogram Query: {histogram_query_result.histogram_query}\n"
            f"| {'Schema':<70} | {'Count':<15} |"
        )
        for key, value in histogram_query_result.histogram.items():
            print(f"| {key:<70} | {value:<15} |")


# [END contentwarehouse_search_documents]
