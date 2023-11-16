from google.cloud import contentwarehouse

def search_documents_sample(project_number: str, location: str, document_query_text: str, user_id: str) -> None:
    if False:
        while True:
            i = 10
    client = contentwarehouse.DocumentServiceClient()
    parent = client.common_location_path(project=project_number, location=location)
    file_type_filter = contentwarehouse.FileTypeFilter(file_type=contentwarehouse.FileTypeFilter.FileType.DOCUMENT)
    document_query = contentwarehouse.DocumentQuery(query=document_query_text, file_type_filter=file_type_filter)
    histogram_query = contentwarehouse.HistogramQuery(histogram_query='count("DocumentSchemaId")')
    request_metadata = contentwarehouse.RequestMetadata(user_info=contentwarehouse.UserInfo(id=user_id))
    request = contentwarehouse.SearchDocumentsRequest(parent=parent, request_metadata=request_metadata, document_query=document_query, histogram_queries=[histogram_query])
    response = client.search_documents(request=request)
    for matching_document in response.matching_documents:
        document = matching_document.document
        print(f'{document.display_name} - {document.document_schema_name}\n{document.name}\n{document.create_time}\n{matching_document.search_text_snippet}\n')
    for histogram_query_result in response.histogram_query_results:
        print(f"Histogram Query: {histogram_query_result.histogram_query}\n| {'Schema':<70} | {'Count':<15} |")
        for (key, value) in histogram_query_result.histogram.items():
            print(f'| {key:<70} | {value:<15} |')