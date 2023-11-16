from featuretools.utils.gen_utils import import_or_raise

def check_graphviz():
    if False:
        print('Hello World!')
    GRAPHVIZ_ERR_MSG = 'Please install graphviz to plot.' + ' (See https://featuretools.alteryx.com/en/stable/install.html#installing-graphviz for' + ' details)'
    graphviz = import_or_raise('graphviz', GRAPHVIZ_ERR_MSG)
    try:
        graphviz.Digraph().pipe(format='svg')
    except graphviz.backend.ExecutableNotFound:
        raise RuntimeError('To plot entity sets, a graphviz backend is required.\n' + 'Install the backend using one of the following commands:\n' + '  Mac OS: brew install graphviz\n' + '  Linux (Ubuntu): $ sudo apt install graphviz\n' + '  Windows (conda): conda install -c conda-forge python-graphviz\n' + '  Windows (pip): pip install graphviz\n' + '  Windows (EXE required if graphviz was installed via pip): https://graphviz.org/download/#windows' + '  For more details visit: https://featuretools.alteryx.com/en/stable/install.html#installing-graphviz')
    return graphviz

def get_graphviz_format(graphviz, to_file):
    if False:
        while True:
            i = 10
    if to_file:
        to_file = str(to_file)
        split_path = to_file.split('.')
        if len(split_path) < 2:
            raise ValueError("Please use a file extension like '.pdf'" + ' so that the format can be inferred')
        format_ = split_path[-1]
        valid_formats = graphviz.FORMATS
        if format_ not in valid_formats:
            raise ValueError('Unknown format. Make sure your format is' + ' amongst the following: %s' % valid_formats)
    else:
        format_ = None
    return format_

def save_graph(graph, to_file, format_):
    if False:
        i = 10
        return i + 15
    offset = len(format_) + 1
    output_path = to_file[:-offset]
    graph.render(output_path, cleanup=True)