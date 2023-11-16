def _mean(x):
    if False:
        while True:
            i = 10
    if len(x) == 0:
        raise ValueError('x must have positive length')
    return float(sum(x)) / len(x)

def _argmin(x):
    if False:
        while True:
            i = 10
    return sorted(enumerate(x), key=lambda t: t[1])[0][0]

def _argmax(x):
    if False:
        return 10
    return sorted(enumerate(x), key=lambda t: t[1], reverse=True)[0][0]

def _df_anno(xanchor, yanchor, x, y):
    if False:
        print('Hello World!')
    'Default annotation parameters'
    return dict(xanchor=xanchor, yanchor=yanchor, x=x, y=y, showarrow=False)

def _add_inside_to_position(pos):
    if False:
        for i in range(10):
            print('nop')
    if not ('inside' in pos or 'outside' in pos):
        pos.add('inside')
    return pos

def _prepare_position(position, prepend_inside=False):
    if False:
        while True:
            i = 10
    if position is None:
        position = 'top right'
    pos_str = position
    position = set(position.split(' '))
    if prepend_inside:
        position = _add_inside_to_position(position)
    return (position, pos_str)

def annotation_params_for_line(shape_type, shape_args, position):
    if False:
        return 10
    x0 = shape_args['x0']
    x1 = shape_args['x1']
    y0 = shape_args['y0']
    y1 = shape_args['y1']
    X = [x0, x1]
    Y = [y0, y1]
    R = 'right'
    T = 'top'
    L = 'left'
    C = 'center'
    B = 'bottom'
    M = 'middle'
    aY = max(Y)
    iY = min(Y)
    eY = _mean(Y)
    aaY = _argmax(Y)
    aiY = _argmin(Y)
    aX = max(X)
    iX = min(X)
    eX = _mean(X)
    aaX = _argmax(X)
    aiX = _argmin(X)
    (position, pos_str) = _prepare_position(position)
    if shape_type == 'vline':
        if position == set(['top', 'left']):
            return _df_anno(R, T, X[aaY], aY)
        if position == set(['top', 'right']):
            return _df_anno(L, T, X[aaY], aY)
        if position == set(['top']):
            return _df_anno(C, B, X[aaY], aY)
        if position == set(['bottom', 'left']):
            return _df_anno(R, B, X[aiY], iY)
        if position == set(['bottom', 'right']):
            return _df_anno(L, B, X[aiY], iY)
        if position == set(['bottom']):
            return _df_anno(C, T, X[aiY], iY)
        if position == set(['left']):
            return _df_anno(R, M, eX, eY)
        if position == set(['right']):
            return _df_anno(L, M, eX, eY)
    elif shape_type == 'hline':
        if position == set(['top', 'left']):
            return _df_anno(L, B, iX, Y[aiX])
        if position == set(['top', 'right']):
            return _df_anno(R, B, aX, Y[aaX])
        if position == set(['top']):
            return _df_anno(C, B, eX, eY)
        if position == set(['bottom', 'left']):
            return _df_anno(L, T, iX, Y[aiX])
        if position == set(['bottom', 'right']):
            return _df_anno(R, T, aX, Y[aaX])
        if position == set(['bottom']):
            return _df_anno(C, T, eX, eY)
        if position == set(['left']):
            return _df_anno(R, M, iX, Y[aiX])
        if position == set(['right']):
            return _df_anno(L, M, aX, Y[aaX])
    raise ValueError('Invalid annotation position "%s"' % (pos_str,))

def annotation_params_for_rect(shape_type, shape_args, position):
    if False:
        while True:
            i = 10
    x0 = shape_args['x0']
    x1 = shape_args['x1']
    y0 = shape_args['y0']
    y1 = shape_args['y1']
    (position, pos_str) = _prepare_position(position, prepend_inside=True)
    if position == set(['inside', 'top', 'left']):
        return _df_anno('left', 'top', min([x0, x1]), max([y0, y1]))
    if position == set(['inside', 'top', 'right']):
        return _df_anno('right', 'top', max([x0, x1]), max([y0, y1]))
    if position == set(['inside', 'top']):
        return _df_anno('center', 'top', _mean([x0, x1]), max([y0, y1]))
    if position == set(['inside', 'bottom', 'left']):
        return _df_anno('left', 'bottom', min([x0, x1]), min([y0, y1]))
    if position == set(['inside', 'bottom', 'right']):
        return _df_anno('right', 'bottom', max([x0, x1]), min([y0, y1]))
    if position == set(['inside', 'bottom']):
        return _df_anno('center', 'bottom', _mean([x0, x1]), min([y0, y1]))
    if position == set(['inside', 'left']):
        return _df_anno('left', 'middle', min([x0, x1]), _mean([y0, y1]))
    if position == set(['inside', 'right']):
        return _df_anno('right', 'middle', max([x0, x1]), _mean([y0, y1]))
    if position == set(['inside']):
        return _df_anno('center', 'middle', _mean([x0, x1]), _mean([y0, y1]))
    if position == set(['outside', 'top', 'left']):
        return _df_anno('right' if shape_type == 'vrect' else 'left', 'bottom' if shape_type == 'hrect' else 'top', min([x0, x1]), max([y0, y1]))
    if position == set(['outside', 'top', 'right']):
        return _df_anno('left' if shape_type == 'vrect' else 'right', 'bottom' if shape_type == 'hrect' else 'top', max([x0, x1]), max([y0, y1]))
    if position == set(['outside', 'top']):
        return _df_anno('center', 'bottom', _mean([x0, x1]), max([y0, y1]))
    if position == set(['outside', 'bottom', 'left']):
        return _df_anno('right' if shape_type == 'vrect' else 'left', 'top' if shape_type == 'hrect' else 'bottom', min([x0, x1]), min([y0, y1]))
    if position == set(['outside', 'bottom', 'right']):
        return _df_anno('left' if shape_type == 'vrect' else 'right', 'top' if shape_type == 'hrect' else 'bottom', max([x0, x1]), min([y0, y1]))
    if position == set(['outside', 'bottom']):
        return _df_anno('center', 'top', _mean([x0, x1]), min([y0, y1]))
    if position == set(['outside', 'left']):
        return _df_anno('right', 'middle', min([x0, x1]), _mean([y0, y1]))
    if position == set(['outside', 'right']):
        return _df_anno('left', 'middle', max([x0, x1]), _mean([y0, y1]))
    raise ValueError('Invalid annotation position %s' % (pos_str,))

def axis_spanning_shape_annotation(annotation, shape_type, shape_args, kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    annotation: a go.layout.Annotation object, a dict describing an annotation, or None\n    shape_type: one of 'vline', 'hline', 'vrect', 'hrect' and determines how the\n                x, y, xanchor, and yanchor values are set.\n    shape_args: the parameters used to draw the shape, which are used to place the annotation\n    kwargs:     a dictionary that was the kwargs of a\n                _process_multiple_axis_spanning_shapes spanning shapes call. Items in this\n                dict whose keys start with 'annotation_' will be extracted and the keys with\n                the 'annotation_' part stripped off will be used to assign properties of the\n                new annotation.\n\n    Property precedence:\n    The annotation's x, y, xanchor, and yanchor properties are set based on the\n    shape_type argument. Each property already specified in the annotation or\n    through kwargs will be left as is (not replaced by the value computed using\n    shape_type). Note that the xref and yref properties will in general get\n    overwritten if the result of this function is passed to an add_annotation\n    called with the row and col parameters specified.\n\n    Returns an annotation populated with fields based on the\n    annotation_position, annotation_ prefixed kwargs or the original annotation\n    passed in to this function.\n    "
    prefix = 'annotation_'
    len_prefix = len(prefix)
    annotation_keys = list(filter(lambda k: k.startswith(prefix), kwargs.keys()))
    if annotation is None and len(annotation_keys) == 0:
        return None
    if annotation is None:
        annotation = dict()
    for k in annotation_keys:
        if k == 'annotation_position':
            continue
        subk = k[len_prefix:]
        annotation[subk] = kwargs[k]
    annotation_position = None
    if 'annotation_position' in kwargs.keys():
        annotation_position = kwargs['annotation_position']
    if shape_type.endswith('line'):
        shape_dict = annotation_params_for_line(shape_type, shape_args, annotation_position)
    elif shape_type.endswith('rect'):
        shape_dict = annotation_params_for_rect(shape_type, shape_args, annotation_position)
    for k in shape_dict.keys():
        if k not in annotation or annotation[k] is None:
            annotation[k] = shape_dict[k]
    return annotation

def split_dict_by_key_prefix(d, prefix):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns two dictionaries, one containing all the items whose keys do not\n    start with a prefix and another containing all the items whose keys do start\n    with the prefix. Note that the prefix is not removed from the keys.\n    '
    no_prefix = dict()
    with_prefix = dict()
    for k in d.keys():
        if k.startswith(prefix):
            with_prefix[k] = d[k]
        else:
            no_prefix[k] = d[k]
    return (no_prefix, with_prefix)