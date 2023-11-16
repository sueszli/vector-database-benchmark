from google.cloud import trace_v2

def sample_batch_write_spans():
    if False:
        print('Hello World!')
    client = trace_v2.TraceServiceClient()
    spans = trace_v2.Span()
    spans.name = 'name_value'
    spans.span_id = 'span_id_value'
    request = trace_v2.BatchWriteSpansRequest(name='name_value', spans=spans)
    client.batch_write_spans(request=request)