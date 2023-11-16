"""
*****
LaTeX
*****

Export NetworkX graphs in LaTeX format using the TikZ library within TeX/LaTeX.
Usually, you will want the drawing to appear in a figure environment so
you use ``to_latex(G, caption="A caption")``. If you want the raw
drawing commands without a figure environment use :func:`to_latex_raw`.
And if you want to write to a file instead of just returning the latex
code as a string, use ``write_latex(G, "filename.tex", caption="A caption")``.

To construct a figure with subfigures for each graph to be shown, provide
``to_latex`` or ``write_latex`` a list of graphs, a list of subcaptions,
and a number of rows of subfigures inside the figure.

To be able to refer to the figures or subfigures in latex using ``\\\\ref``,
the keyword ``latex_label`` is available for figures and `sub_labels` for
a list of labels, one for each subfigure.

We intend to eventually provide an interface to the TikZ Graph
features which include e.g. layout algorithms.

Let us know via github what you'd like to see available, or better yet
give us some code to do it, or even better make a github pull request
to add the feature.

The TikZ approach
=================
Drawing options can be stored on the graph as node/edge attributes, or
can be provided as dicts keyed by node/edge to a string of the options
for that node/edge. Similarly a label can be shown for each node/edge
by specifying the labels as graph node/edge attributes or by providing
a dict keyed by node/edge to the text to be written for that node/edge.

Options for the tikzpicture environment (e.g. "[scale=2]") can be provided
via a keyword argument. Similarly default node and edge options can be
provided through keywords arguments. The default node options are applied
to the single TikZ "path" that draws all nodes (and no edges). The default edge
options are applied to a TikZ "scope" which contains a path for each edge.

Examples
========
>>> G = nx.path_graph(3)
>>> nx.write_latex(G, "just_my_figure.tex", as_document=True)
>>> nx.write_latex(G, "my_figure.tex", caption="A path graph", latex_label="fig1")
>>> latex_code = nx.to_latex(G)  # a string rather than a file

You can change many features of the nodes and edges.

>>> G = nx.path_graph(4, create_using=nx.DiGraph)
>>> pos = {n: (n, n) for n in G}  # nodes set on a line

>>> G.nodes[0]["style"] = "blue"
>>> G.nodes[2]["style"] = "line width=3,draw"
>>> G.nodes[3]["label"] = "Stop"
>>> G.edges[(0, 1)]["label"] = "1st Step"
>>> G.edges[(0, 1)]["label_opts"] = "near start"
>>> G.edges[(1, 2)]["style"] = "line width=3"
>>> G.edges[(1, 2)]["label"] = "2nd Step"
>>> G.edges[(2, 3)]["style"] = "green"
>>> G.edges[(2, 3)]["label"] = "3rd Step"
>>> G.edges[(2, 3)]["label_opts"] = "near end"

>>> nx.write_latex(G, "latex_graph.tex", pos=pos, as_document=True)

Then compile the LaTeX using something like ``pdflatex latex_graph.tex``
and view the pdf file created: ``latex_graph.pdf``.

If you want **subfigures** each containing one graph, you can input a list of graphs.

>>> H1 = nx.path_graph(4)
>>> H2 = nx.complete_graph(4)
>>> H3 = nx.path_graph(8)
>>> H4 = nx.complete_graph(8)
>>> graphs = [H1, H2, H3, H4]
>>> caps = ["Path 4", "Complete graph 4", "Path 8", "Complete graph 8"]
>>> lbls = ["fig2a", "fig2b", "fig2c", "fig2d"]
>>> nx.write_latex(graphs, "subfigs.tex", n_rows=2, sub_captions=caps, sub_labels=lbls)
>>> latex_code = nx.to_latex(graphs, n_rows=2, sub_captions=caps, sub_labels=lbls)

>>> node_color = {0: "red", 1: "orange", 2: "blue", 3: "gray!90"}
>>> edge_width = {e: "line width=1.5" for e in H3.edges}
>>> pos = nx.circular_layout(H3)
>>> latex_code = nx.to_latex(H3, pos, node_options=node_color, edge_options=edge_width)
>>> print(latex_code)
\\documentclass{report}
\\usepackage{tikz}
\\usepackage{subcaption}
<BLANKLINE>
\\begin{document}
\\begin{figure}
  \\begin{tikzpicture}
      \\draw
        (1.0, 0.0) node[red] (0){0}
        (0.707, 0.707) node[orange] (1){1}
        (-0.0, 1.0) node[blue] (2){2}
        (-0.707, 0.707) node[gray!90] (3){3}
        (-1.0, -0.0) node (4){4}
        (-0.707, -0.707) node (5){5}
        (0.0, -1.0) node (6){6}
        (0.707, -0.707) node (7){7};
      \\begin{scope}[-]
        \\draw[line width=1.5] (0) to (1);
        \\draw[line width=1.5] (1) to (2);
        \\draw[line width=1.5] (2) to (3);
        \\draw[line width=1.5] (3) to (4);
        \\draw[line width=1.5] (4) to (5);
        \\draw[line width=1.5] (5) to (6);
        \\draw[line width=1.5] (6) to (7);
      \\end{scope}
    \\end{tikzpicture}
\\end{figure}
\\end{document}

Notes
-----
If you want to change the preamble/postamble of the figure/document/subfigure
environment, use the keyword arguments: `figure_wrapper`, `document_wrapper`,
`subfigure_wrapper`. The default values are stored in private variables
e.g. ``nx.nx_layout._DOCUMENT_WRAPPER``

References
----------
TikZ:          https://tikz.dev/

TikZ options details:   https://tikz.dev/tikz-actions
"""
import numbers
import os
import networkx as nx
__all__ = ['to_latex_raw', 'to_latex', 'write_latex']

@nx.utils.not_implemented_for('multigraph')
def to_latex_raw(G, pos='pos', tikz_options='', default_node_options='', node_options='node_options', node_label='label', default_edge_options='', edge_options='edge_options', edge_label='label', edge_label_options='edge_label_options'):
    if False:
        for i in range(10):
            print('nop')
    'Return a string of the LaTeX/TikZ code to draw `G`\n\n    This function produces just the code for the tikzpicture\n    without any enclosing environment.\n\n    Parameters\n    ==========\n    G : NetworkX graph\n        The NetworkX graph to be drawn\n    pos : string or dict (default "pos")\n        The name of the node attribute on `G` that holds the position of each node.\n        Positions can be sequences of length 2 with numbers for (x,y) coordinates.\n        They can also be strings to denote positions in TikZ style, such as (x, y)\n        or (angle:radius).\n        If a dict, it should be keyed by node to a position.\n        If an empty dict, a circular layout is computed by TikZ.\n    tikz_options : string\n        The tikzpicture options description defining the options for the picture.\n        Often large scale options like `[scale=2]`.\n    default_node_options : string\n        The draw options for a path of nodes. Individual node options override these.\n    node_options : string or dict\n        The name of the node attribute on `G` that holds the options for each node.\n        Or a dict keyed by node to a string holding the options for that node.\n    node_label : string or dict\n        The name of the node attribute on `G` that holds the node label (text)\n        displayed for each node. If the attribute is "" or not present, the node\n        itself is drawn as a string. LaTeX processing such as ``"$A_1$"`` is allowed.\n        Or a dict keyed by node to a string holding the label for that node.\n    default_edge_options : string\n        The options for the scope drawing all edges. The default is "[-]" for\n        undirected graphs and "[->]" for directed graphs.\n    edge_options : string or dict\n        The name of the edge attribute on `G` that holds the options for each edge.\n        If the edge is a self-loop and ``"loop" not in edge_options`` the option\n        "loop," is added to the options for the self-loop edge. Hence you can\n        use "[loop above]" explicitly, but the default is "[loop]".\n        Or a dict keyed by edge to a string holding the options for that edge.\n    edge_label : string or dict\n        The name of the edge attribute on `G` that holds the edge label (text)\n        displayed for each edge. If the attribute is "" or not present, no edge\n        label is drawn.\n        Or a dict keyed by edge to a string holding the label for that edge.\n    edge_label_options : string or dict\n        The name of the edge attribute on `G` that holds the label options for\n        each edge. For example, "[sloped,above,blue]". The default is no options.\n        Or a dict keyed by edge to a string holding the label options for that edge.\n\n    Returns\n    =======\n    latex_code : string\n       The text string which draws the desired graph(s) when compiled by LaTeX.\n\n    See Also\n    ========\n    to_latex\n    write_latex\n    '
    i4 = '\n    '
    i8 = '\n        '
    if not isinstance(pos, dict):
        pos = nx.get_node_attributes(G, pos)
    if not pos:
        pos = {n: f'({round(360.0 * i / len(G), 3)}:2)' for (i, n) in enumerate(G)}
    for node in G:
        if node not in pos:
            raise nx.NetworkXError(f'node {node} has no specified pos {pos}')
        posnode = pos[node]
        if not isinstance(posnode, str):
            try:
                (posx, posy) = posnode
                pos[node] = f'({round(posx, 3)}, {round(posy, 3)})'
            except (TypeError, ValueError):
                msg = f'position pos[{node}] is not 2-tuple or a string: {posnode}'
                raise nx.NetworkXError(msg)
    if not isinstance(node_options, dict):
        node_options = nx.get_node_attributes(G, node_options)
    if not isinstance(node_label, dict):
        node_label = nx.get_node_attributes(G, node_label)
    if not isinstance(edge_options, dict):
        edge_options = nx.get_edge_attributes(G, edge_options)
    if not isinstance(edge_label, dict):
        edge_label = nx.get_edge_attributes(G, edge_label)
    if not isinstance(edge_label_options, dict):
        edge_label_options = nx.get_edge_attributes(G, edge_label_options)
    topts = '' if tikz_options == '' else f"[{tikz_options.strip('[]')}]"
    defn = '' if default_node_options == '' else f"[{default_node_options.strip('[]')}]"
    linestyle = f"{('->' if G.is_directed() else '-')}"
    if default_edge_options == '':
        defe = '[' + linestyle + ']'
    elif '-' in default_edge_options:
        defe = default_edge_options
    else:
        defe = f"[{linestyle},{default_edge_options.strip('[]')}]"
    result = '  \\begin{tikzpicture}' + topts
    result += i4 + '  \\draw' + defn
    for n in G:
        nopts = f"[{node_options[n].strip('[]')}]" if n in node_options else ''
        ntext = f'{{{node_label[n]}}}' if n in node_label else f'{{{n}}}'
        result += i8 + f'{pos[n]} node{nopts} ({n}){ntext}'
    result += ';\n'
    result += '      \\begin{scope}' + defe
    for edge in G.edges:
        (u, v) = edge[:2]
        e_opts = f'{edge_options[edge]}'.strip('[]') if edge in edge_options else ''
        if u == v and 'loop' not in e_opts:
            e_opts = 'loop,' + e_opts
        e_opts = f'[{e_opts}]' if e_opts != '' else ''
        els = edge_label_options[edge] if edge in edge_label_options else ''
        els = f"[{els.strip('[]')}]"
        e_label = f' node{els} {{{edge_label[edge]}}}' if edge in edge_label else ''
        result += i8 + f'\\draw{e_opts} ({u}) to{e_label} ({v});'
    result += '\n      \\end{scope}\n    \\end{tikzpicture}\n'
    return result
_DOC_WRAPPER_TIKZ = '\\documentclass{{report}}\n\\usepackage{{tikz}}\n\\usepackage{{subcaption}}\n\n\\begin{{document}}\n{content}\n\\end{{document}}'
_FIG_WRAPPER = '\\begin{{figure}}\n{content}{caption}{label}\n\\end{{figure}}'
_SUBFIG_WRAPPER = '  \\begin{{subfigure}}{{{size}\\textwidth}}\n{content}{caption}{label}\n  \\end{{subfigure}}'

def to_latex(Gbunch, pos='pos', tikz_options='', default_node_options='', node_options='node_options', node_label='node_label', default_edge_options='', edge_options='edge_options', edge_label='edge_label', edge_label_options='edge_label_options', caption='', latex_label='', sub_captions=None, sub_labels=None, n_rows=1, as_document=True, document_wrapper=_DOC_WRAPPER_TIKZ, figure_wrapper=_FIG_WRAPPER, subfigure_wrapper=_SUBFIG_WRAPPER):
    if False:
        print('Hello World!')
    'Return latex code to draw the graph(s) in `Gbunch`\n\n    The TikZ drawing utility in LaTeX is used to draw the graph(s).\n    If `Gbunch` is a graph, it is drawn in a figure environment.\n    If `Gbunch` is an iterable of graphs, each is drawn in a subfigure environment\n    within a single figure environment.\n\n    If `as_document` is True, the figure is wrapped inside a document environment\n    so that the resulting string is ready to be compiled by LaTeX. Otherwise,\n    the string is ready for inclusion in a larger tex document using ``\\include``\n    or ``\\input`` statements.\n\n    Parameters\n    ==========\n    Gbunch : NetworkX graph or iterable of NetworkX graphs\n        The NetworkX graph to be drawn or an iterable of graphs\n        to be drawn inside subfigures of a single figure.\n    pos : string or list of strings\n        The name of the node attribute on `G` that holds the position of each node.\n        Positions can be sequences of length 2 with numbers for (x,y) coordinates.\n        They can also be strings to denote positions in TikZ style, such as (x, y)\n        or (angle:radius).\n        If a dict, it should be keyed by node to a position.\n        If an empty dict, a circular layout is computed by TikZ.\n        If you are drawing many graphs in subfigures, use a list of position dicts.\n    tikz_options : string\n        The tikzpicture options description defining the options for the picture.\n        Often large scale options like `[scale=2]`.\n    default_node_options : string\n        The draw options for a path of nodes. Individual node options override these.\n    node_options : string or dict\n        The name of the node attribute on `G` that holds the options for each node.\n        Or a dict keyed by node to a string holding the options for that node.\n    node_label : string or dict\n        The name of the node attribute on `G` that holds the node label (text)\n        displayed for each node. If the attribute is "" or not present, the node\n        itself is drawn as a string. LaTeX processing such as ``"$A_1$"`` is allowed.\n        Or a dict keyed by node to a string holding the label for that node.\n    default_edge_options : string\n        The options for the scope drawing all edges. The default is "[-]" for\n        undirected graphs and "[->]" for directed graphs.\n    edge_options : string or dict\n        The name of the edge attribute on `G` that holds the options for each edge.\n        If the edge is a self-loop and ``"loop" not in edge_options`` the option\n        "loop," is added to the options for the self-loop edge. Hence you can\n        use "[loop above]" explicitly, but the default is "[loop]".\n        Or a dict keyed by edge to a string holding the options for that edge.\n    edge_label : string or dict\n        The name of the edge attribute on `G` that holds the edge label (text)\n        displayed for each edge. If the attribute is "" or not present, no edge\n        label is drawn.\n        Or a dict keyed by edge to a string holding the label for that edge.\n    edge_label_options : string or dict\n        The name of the edge attribute on `G` that holds the label options for\n        each edge. For example, "[sloped,above,blue]". The default is no options.\n        Or a dict keyed by edge to a string holding the label options for that edge.\n    caption : string\n        The caption string for the figure environment\n    latex_label : string\n        The latex label used for the figure for easy referral from the main text\n    sub_captions : list of strings\n        The sub_caption string for each subfigure in the figure\n    sub_latex_labels : list of strings\n        The latex label for each subfigure in the figure\n    n_rows : int\n        The number of rows of subfigures to arrange for multiple graphs\n    as_document : bool\n        Whether to wrap the latex code in a document environment for compiling\n    document_wrapper : formatted text string with variable ``content``.\n        This text is called to evaluate the content embedded in a document\n        environment with a preamble setting up TikZ.\n    figure_wrapper : formatted text string\n        This text is evaluated with variables ``content``, ``caption`` and ``label``.\n        It wraps the content and if a caption is provided, adds the latex code for\n        that caption, and if a label is provided, adds the latex code for a label.\n    subfigure_wrapper : formatted text string\n        This text evaluate variables ``size``, ``content``, ``caption`` and ``label``.\n        It wraps the content and if a caption is provided, adds the latex code for\n        that caption, and if a label is provided, adds the latex code for a label.\n        The size is the vertical size of each row of subfigures as a fraction.\n\n    Returns\n    =======\n    latex_code : string\n        The text string which draws the desired graph(s) when compiled by LaTeX.\n\n    See Also\n    ========\n    write_latex\n    to_latex_raw\n    '
    if hasattr(Gbunch, 'adj'):
        raw = to_latex_raw(Gbunch, pos, tikz_options, default_node_options, node_options, node_label, default_edge_options, edge_options, edge_label, edge_label_options)
    else:
        sbf = subfigure_wrapper
        size = 1 / n_rows
        N = len(Gbunch)
        if isinstance(pos, str | dict):
            pos = [pos] * N
        if sub_captions is None:
            sub_captions = [''] * N
        if sub_labels is None:
            sub_labels = [''] * N
        if not len(Gbunch) == len(pos) == len(sub_captions) == len(sub_labels):
            raise nx.NetworkXError('length of Gbunch, sub_captions and sub_figures must agree')
        raw = ''
        for (G, pos, subcap, sublbl) in zip(Gbunch, pos, sub_captions, sub_labels):
            subraw = to_latex_raw(G, pos, tikz_options, default_node_options, node_options, node_label, default_edge_options, edge_options, edge_label, edge_label_options)
            cap = f'    \\caption{{{subcap}}}' if subcap else ''
            lbl = f'\\label{{{sublbl}}}' if sublbl else ''
            raw += sbf.format(size=size, content=subraw, caption=cap, label=lbl)
            raw += '\n'
    raw = raw[:-1]
    cap = f'\n  \\caption{{{caption}}}' if caption else ''
    lbl = f'\\label{{{latex_label}}}' if latex_label else ''
    fig = figure_wrapper.format(content=raw, caption=cap, label=lbl)
    if as_document:
        return document_wrapper.format(content=fig)
    return fig

@nx.utils.open_file(1, mode='w')
def write_latex(Gbunch, path, **options):
    if False:
        print('Hello World!')
    'Write the latex code to draw the graph(s) onto `path`.\n\n    This convenience function creates the latex drawing code as a string\n    and writes that to a file ready to be compiled when `as_document` is True\n    or ready to be ``import`` ed or ``include`` ed into your main LaTeX document.\n\n    The `path` argument can be a string filename or a file handle to write to.\n\n    Parameters\n    ----------\n    Gbunch : NetworkX graph or iterable of NetworkX graphs\n        If Gbunch is a graph, it is drawn in a figure environment.\n        If Gbunch is an iterable of graphs, each is drawn in a subfigure\n        environment within a single figure environment.\n    path : filename\n        Filename or file handle to write to\n    options : dict\n        By default, TikZ is used with options: (others are ignored)::\n\n            pos : string or dict or list\n                The name of the node attribute on `G` that holds the position of each node.\n                Positions can be sequences of length 2 with numbers for (x,y) coordinates.\n                They can also be strings to denote positions in TikZ style, such as (x, y)\n                or (angle:radius).\n                If a dict, it should be keyed by node to a position.\n                If an empty dict, a circular layout is computed by TikZ.\n                If you are drawing many graphs in subfigures, use a list of position dicts.\n            tikz_options : string\n                The tikzpicture options description defining the options for the picture.\n                Often large scale options like `[scale=2]`.\n            default_node_options : string\n                The draw options for a path of nodes. Individual node options override these.\n            node_options : string or dict\n                The name of the node attribute on `G` that holds the options for each node.\n                Or a dict keyed by node to a string holding the options for that node.\n            node_label : string or dict\n                The name of the node attribute on `G` that holds the node label (text)\n                displayed for each node. If the attribute is "" or not present, the node\n                itself is drawn as a string. LaTeX processing such as ``"$A_1$"`` is allowed.\n                Or a dict keyed by node to a string holding the label for that node.\n            default_edge_options : string\n                The options for the scope drawing all edges. The default is "[-]" for\n                undirected graphs and "[->]" for directed graphs.\n            edge_options : string or dict\n                The name of the edge attribute on `G` that holds the options for each edge.\n                If the edge is a self-loop and ``"loop" not in edge_options`` the option\n                "loop," is added to the options for the self-loop edge. Hence you can\n                use "[loop above]" explicitly, but the default is "[loop]".\n                Or a dict keyed by edge to a string holding the options for that edge.\n            edge_label : string or dict\n                The name of the edge attribute on `G` that holds the edge label (text)\n                displayed for each edge. If the attribute is "" or not present, no edge\n                label is drawn.\n                Or a dict keyed by edge to a string holding the label for that edge.\n            edge_label_options : string or dict\n                The name of the edge attribute on `G` that holds the label options for\n                each edge. For example, "[sloped,above,blue]". The default is no options.\n                Or a dict keyed by edge to a string holding the label options for that edge.\n            caption : string\n                The caption string for the figure environment\n            latex_label : string\n                The latex label used for the figure for easy referral from the main text\n            sub_captions : list of strings\n                The sub_caption string for each subfigure in the figure\n            sub_latex_labels : list of strings\n                The latex label for each subfigure in the figure\n            n_rows : int\n                The number of rows of subfigures to arrange for multiple graphs\n            as_document : bool\n                Whether to wrap the latex code in a document environment for compiling\n            document_wrapper : formatted text string with variable ``content``.\n                This text is called to evaluate the content embedded in a document\n                environment with a preamble setting up the TikZ syntax.\n            figure_wrapper : formatted text string\n                This text is evaluated with variables ``content``, ``caption`` and ``label``.\n                It wraps the content and if a caption is provided, adds the latex code for\n                that caption, and if a label is provided, adds the latex code for a label.\n            subfigure_wrapper : formatted text string\n                This text evaluate variables ``size``, ``content``, ``caption`` and ``label``.\n                It wraps the content and if a caption is provided, adds the latex code for\n                that caption, and if a label is provided, adds the latex code for a label.\n                The size is the vertical size of each row of subfigures as a fraction.\n\n    See Also\n    ========\n    to_latex\n    '
    path.write(to_latex(Gbunch, **options))