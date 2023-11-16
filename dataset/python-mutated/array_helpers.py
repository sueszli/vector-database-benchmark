def concat(model, blobs_in, blob_out, **kwargs):
    if False:
        i = 10
        return i + 15
    'Depth Concat.'
    if kwargs.get('order') and kwargs.get('axis'):
        kwargs.pop('order')
    return model.net.Concat(blobs_in, [blob_out, '_' + blob_out + '_concat_dims'], **kwargs)[0]

def depth_concat(model, blobs_in, blob_out, **kwargs):
    if False:
        print('Hello World!')
    'The old depth concat function - we should move to use concat.'
    print('DepthConcat is deprecated. use Concat instead.')
    return concat(blobs_in, blob_out, **kwargs)