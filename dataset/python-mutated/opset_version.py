def support(opset_versions):
    if False:
        return 10
    'Detect lowest supported version of the target converter\n\n    A simple wrap function for convert functions to detect lowest number of\n    supported opset version. For example, the target ONNX operater is added\n    from 6 and updated on 8, add this function as decorator like the below.\n\n    >>> @support((6, 8))\n    ... def own_converter(func, opset_version, *args):\n    ...     print(opset_version)\n    >>>\n    >>> own_converter(None, 6)\n    6\n    >>> own_converter(None, 7)\n    6\n    >>> own_converter(None, 8)\n    8\n    >>> own_converter(None, 9)\n    8\n    >>> own_converter(None, 5)\n    RuntimeError: ONNX-Chainer cannot convert ...(snip)\n\n    Arguments:\n        opset_versions (tuple): Tuple of opset versions.\n\n    '

    def _wrapper(func):
        if False:
            print('Hello World!')

        def _func_with_lower_opset_version(*args, **kwargs):
            if False:
                while True:
                    i = 10
            if opset_versions is None:
                return func(*args, **kwargs)
            opset_version = args[1]
            for opver in sorted(opset_versions, reverse=True):
                if opver <= opset_version:
                    break
            if opver > opset_version:
                func_name = args[0].__class__.__name__
                raise RuntimeError('ONNX-Chainer cannot convert `{}` of Chainer with ONNX opset_version {}'.format(func_name, opset_version))
            opset_version = opver
            return func(args[0], opset_version, *args[2:], **kwargs)
        return _func_with_lower_opset_version
    return _wrapper