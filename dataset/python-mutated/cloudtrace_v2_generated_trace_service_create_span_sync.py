from google.cloud import trace_v2

def sample_create_span():
    if False:
        for i in range(10):
            print('nop')
    client = trace_v2.TraceServiceClient()
    request = trace_v2.Span(name='name_value', span_id='span_id_value')
    response = client.create_span(request=request)
    print(response)