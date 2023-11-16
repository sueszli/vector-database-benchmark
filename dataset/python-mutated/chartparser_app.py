"""
A graphical tool for exploring chart parsing.

Chart parsing is a flexible parsing algorithm that uses a data
structure called a "chart" to record hypotheses about syntactic
constituents.  Each hypothesis is represented by a single "edge" on
the chart.  A set of "chart rules" determine when new edges can be
added to the chart.  This set of rules controls the overall behavior
of the parser (e.g. whether it parses top-down or bottom-up).

The chart parsing tool demonstrates the process of parsing a single
sentence, with a given grammar and lexicon.  Its display is divided
into three sections: the bottom section displays the chart; the middle
section displays the sentence; and the top section displays the
partial syntax tree corresponding to the selected edge.  Buttons along
the bottom of the window are used to control the execution of the
algorithm.

The chart parsing tool allows for flexible control of the parsing
algorithm.  At each step of the algorithm, you can select which rule
or strategy you wish to apply.  This allows you to experiment with
mixing different strategies (e.g. top-down and bottom-up).  You can
exercise fine-grained control over the algorithm by selecting which
edge you wish to apply a rule to.
"""
import os.path
import pickle
from tkinter import Button, Canvas, Checkbutton, Frame, IntVar, Label, Menu, Scrollbar, Tk, Toplevel
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from tkinter.messagebox import showerror, showinfo
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, ColorizedList, EntryDialog, MutableOptionMenu, ShowText, SymbolWidget
from nltk.grammar import CFG, Nonterminal
from nltk.parse.chart import BottomUpPredictCombineRule, BottomUpPredictRule, Chart, LeafEdge, LeafInitRule, SingleEdgeFundamentalRule, SteppingChartParser, TopDownInitRule, TopDownPredictRule, TreeEdge
from nltk.tree import Tree
from nltk.util import in_idle

class EdgeList(ColorizedList):
    ARROW = SymbolWidget.SYMBOLS['rightarrow']

    def _init_colortags(self, textwidget, options):
        if False:
            while True:
                i = 10
        textwidget.tag_config('terminal', foreground='#006000')
        textwidget.tag_config('arrow', font='symbol', underline='0')
        textwidget.tag_config('dot', foreground='#000000')
        textwidget.tag_config('nonterminal', foreground='blue', font=('helvetica', -12, 'bold'))

    def _item_repr(self, item):
        if False:
            while True:
                i = 10
        contents = []
        contents.append(('%s\t' % item.lhs(), 'nonterminal'))
        contents.append((self.ARROW, 'arrow'))
        for (i, elt) in enumerate(item.rhs()):
            if i == item.dot():
                contents.append((' *', 'dot'))
            if isinstance(elt, Nonterminal):
                contents.append((' %s' % elt.symbol(), 'nonterminal'))
            else:
                contents.append((' %r' % elt, 'terminal'))
        if item.is_complete():
            contents.append((' *', 'dot'))
        return contents

class ChartMatrixView:
    """
    A view of a chart that displays the contents of the corresponding matrix.
    """

    def __init__(self, parent, chart, toplevel=True, title='Chart Matrix', show_numedges=False):
        if False:
            for i in range(10):
                print('nop')
        self._chart = chart
        self._cells = []
        self._marks = []
        self._selected_cell = None
        if toplevel:
            self._root = Toplevel(parent)
            self._root.title(title)
            self._root.bind('<Control-q>', self.destroy)
            self._init_quit(self._root)
        else:
            self._root = Frame(parent)
        self._init_matrix(self._root)
        self._init_list(self._root)
        if show_numedges:
            self._init_numedges(self._root)
        else:
            self._numedges_label = None
        self._callbacks = {}
        self._num_edges = 0
        self.draw()

    def _init_quit(self, root):
        if False:
            print('Hello World!')
        quit = Button(root, text='Quit', command=self.destroy)
        quit.pack(side='bottom', expand=0, fill='none')

    def _init_matrix(self, root):
        if False:
            i = 10
            return i + 15
        cframe = Frame(root, border=2, relief='sunken')
        cframe.pack(expand=0, fill='none', padx=1, pady=3, side='top')
        self._canvas = Canvas(cframe, width=200, height=200, background='white')
        self._canvas.pack(expand=0, fill='none')

    def _init_numedges(self, root):
        if False:
            while True:
                i = 10
        self._numedges_label = Label(root, text='0 edges')
        self._numedges_label.pack(expand=0, fill='none', side='top')

    def _init_list(self, root):
        if False:
            print('Hello World!')
        self._list = EdgeList(root, [], width=20, height=5)
        self._list.pack(side='top', expand=1, fill='both', pady=3)

        def cb(edge, self=self):
            if False:
                return 10
            self._fire_callbacks('select', edge)
        self._list.add_callback('select', cb)
        self._list.focus()

    def destroy(self, *e):
        if False:
            return 10
        if self._root is None:
            return
        try:
            self._root.destroy()
        except:
            pass
        self._root = None

    def set_chart(self, chart):
        if False:
            i = 10
            return i + 15
        if chart is not self._chart:
            self._chart = chart
            self._num_edges = 0
            self.draw()

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        if self._root is None:
            return
        N = len(self._cells)
        cell_edges = [[0 for i in range(N)] for j in range(N)]
        for edge in self._chart:
            cell_edges[edge.start()][edge.end()] += 1
        for i in range(N):
            for j in range(i, N):
                if cell_edges[i][j] == 0:
                    color = 'gray20'
                else:
                    color = '#00{:02x}{:02x}'.format(min(255, 50 + 128 * cell_edges[i][j] / 10), max(0, 128 - 128 * cell_edges[i][j] / 10))
                cell_tag = self._cells[i][j]
                self._canvas.itemconfig(cell_tag, fill=color)
                if (i, j) == self._selected_cell:
                    self._canvas.itemconfig(cell_tag, outline='#00ffff', width=3)
                    self._canvas.tag_raise(cell_tag)
                else:
                    self._canvas.itemconfig(cell_tag, outline='black', width=1)
        edges = list(self._chart.select(span=self._selected_cell))
        self._list.set(edges)
        self._num_edges = self._chart.num_edges()
        if self._numedges_label is not None:
            self._numedges_label['text'] = '%d edges' % self._num_edges

    def activate(self):
        if False:
            for i in range(10):
                print('nop')
        self._canvas.itemconfig('inactivebox', state='hidden')
        self.update()

    def inactivate(self):
        if False:
            print('Hello World!')
        self._canvas.itemconfig('inactivebox', state='normal')
        self.update()

    def add_callback(self, event, func):
        if False:
            print('Hello World!')
        self._callbacks.setdefault(event, {})[func] = 1

    def remove_callback(self, event, func=None):
        if False:
            for i in range(10):
                print('nop')
        if func is None:
            del self._callbacks[event]
        else:
            try:
                del self._callbacks[event][func]
            except:
                pass

    def _fire_callbacks(self, event, *args):
        if False:
            while True:
                i = 10
        if event not in self._callbacks:
            return
        for cb_func in list(self._callbacks[event].keys()):
            cb_func(*args)

    def select_cell(self, i, j):
        if False:
            return 10
        if self._root is None:
            return
        if (i, j) == self._selected_cell and self._chart.num_edges() == self._num_edges:
            return
        self._selected_cell = (i, j)
        self.update()
        self._fire_callbacks('select_cell', i, j)

    def deselect_cell(self):
        if False:
            for i in range(10):
                print('nop')
        if self._root is None:
            return
        self._selected_cell = None
        self._list.set([])
        self.update()

    def _click_cell(self, i, j):
        if False:
            return 10
        if self._selected_cell == (i, j):
            self.deselect_cell()
        else:
            self.select_cell(i, j)

    def view_edge(self, edge):
        if False:
            i = 10
            return i + 15
        self.select_cell(*edge.span())
        self._list.view(edge)

    def mark_edge(self, edge):
        if False:
            return 10
        if self._root is None:
            return
        self.select_cell(*edge.span())
        self._list.mark(edge)

    def unmark_edge(self, edge=None):
        if False:
            return 10
        if self._root is None:
            return
        self._list.unmark(edge)

    def markonly_edge(self, edge):
        if False:
            while True:
                i = 10
        if self._root is None:
            return
        self.select_cell(*edge.span())
        self._list.markonly(edge)

    def draw(self):
        if False:
            for i in range(10):
                print('nop')
        if self._root is None:
            return
        LEFT_MARGIN = BOT_MARGIN = 15
        TOP_MARGIN = 5
        c = self._canvas
        c.delete('all')
        N = self._chart.num_leaves() + 1
        dx = (int(c['width']) - LEFT_MARGIN) / N
        dy = (int(c['height']) - TOP_MARGIN - BOT_MARGIN) / N
        c.delete('all')
        for i in range(N):
            c.create_text(LEFT_MARGIN - 2, i * dy + dy / 2 + TOP_MARGIN, text=repr(i), anchor='e')
            c.create_text(i * dx + dx / 2 + LEFT_MARGIN, N * dy + TOP_MARGIN + 1, text=repr(i), anchor='n')
            c.create_line(LEFT_MARGIN, dy * (i + 1) + TOP_MARGIN, dx * N + LEFT_MARGIN, dy * (i + 1) + TOP_MARGIN, dash='.')
            c.create_line(dx * i + LEFT_MARGIN, TOP_MARGIN, dx * i + LEFT_MARGIN, dy * N + TOP_MARGIN, dash='.')
        c.create_rectangle(LEFT_MARGIN, TOP_MARGIN, LEFT_MARGIN + dx * N, dy * N + TOP_MARGIN, width=2)
        self._cells = [[None for i in range(N)] for j in range(N)]
        for i in range(N):
            for j in range(i, N):
                t = c.create_rectangle(j * dx + LEFT_MARGIN, i * dy + TOP_MARGIN, (j + 1) * dx + LEFT_MARGIN, (i + 1) * dy + TOP_MARGIN, fill='gray20')
                self._cells[i][j] = t

                def cb(event, self=self, i=i, j=j):
                    if False:
                        return 10
                    self._click_cell(i, j)
                c.tag_bind(t, '<Button-1>', cb)
        (xmax, ymax) = (int(c['width']), int(c['height']))
        t = c.create_rectangle(-100, -100, xmax + 100, ymax + 100, fill='gray50', state='hidden', tag='inactivebox')
        c.tag_lower(t)
        self.update()

    def pack(self, *args, **kwargs):
        if False:
            return 10
        self._root.pack(*args, **kwargs)

class ChartResultsView:

    def __init__(self, parent, chart, grammar, toplevel=True):
        if False:
            print('Hello World!')
        self._chart = chart
        self._grammar = grammar
        self._trees = []
        self._y = 10
        self._treewidgets = []
        self._selection = None
        self._selectbox = None
        if toplevel:
            self._root = Toplevel(parent)
            self._root.title('Chart Parser Application: Results')
            self._root.bind('<Control-q>', self.destroy)
        else:
            self._root = Frame(parent)
        if toplevel:
            buttons = Frame(self._root)
            buttons.pack(side='bottom', expand=0, fill='x')
            Button(buttons, text='Quit', command=self.destroy).pack(side='right')
            Button(buttons, text='Print All', command=self.print_all).pack(side='left')
            Button(buttons, text='Print Selection', command=self.print_selection).pack(side='left')
        self._cframe = CanvasFrame(self._root, closeenough=20)
        self._cframe.pack(side='top', expand=1, fill='both')
        self.update()

    def update(self, edge=None):
        if False:
            for i in range(10):
                print('nop')
        if self._root is None:
            return
        if edge is not None:
            if edge.lhs() != self._grammar.start():
                return
            if edge.span() != (0, self._chart.num_leaves()):
                return
        for parse in self._chart.parses(self._grammar.start()):
            if parse not in self._trees:
                self._add(parse)

    def _add(self, parse):
        if False:
            for i in range(10):
                print('nop')
        self._trees.append(parse)
        c = self._cframe.canvas()
        treewidget = tree_to_treesegment(c, parse)
        self._treewidgets.append(treewidget)
        self._cframe.add_widget(treewidget, 10, self._y)
        treewidget.bind_click(self._click)
        self._y = treewidget.bbox()[3] + 10

    def _click(self, widget):
        if False:
            i = 10
            return i + 15
        c = self._cframe.canvas()
        if self._selection is not None:
            c.delete(self._selectbox)
        self._selection = widget
        (x1, y1, x2, y2) = widget.bbox()
        self._selectbox = c.create_rectangle(x1, y1, x2, y2, width=2, outline='#088')

    def _color(self, treewidget, color):
        if False:
            print('Hello World!')
        treewidget.label()['color'] = color
        for child in treewidget.subtrees():
            if isinstance(child, TreeSegmentWidget):
                self._color(child, color)
            else:
                child['color'] = color

    def print_all(self, *e):
        if False:
            i = 10
            return i + 15
        if self._root is None:
            return
        self._cframe.print_to_file()

    def print_selection(self, *e):
        if False:
            return 10
        if self._root is None:
            return
        if self._selection is None:
            showerror('Print Error', 'No tree selected')
        else:
            c = self._cframe.canvas()
            for widget in self._treewidgets:
                if widget is not self._selection:
                    self._cframe.destroy_widget(widget)
            c.delete(self._selectbox)
            (x1, y1, x2, y2) = self._selection.bbox()
            self._selection.move(10 - x1, 10 - y1)
            c['scrollregion'] = f'0 0 {x2 - x1 + 20} {y2 - y1 + 20}'
            self._cframe.print_to_file()
            self._treewidgets = [self._selection]
            self.clear()
            self.update()

    def clear(self):
        if False:
            print('Hello World!')
        if self._root is None:
            return
        for treewidget in self._treewidgets:
            self._cframe.destroy_widget(treewidget)
        self._trees = []
        self._treewidgets = []
        if self._selection is not None:
            self._cframe.canvas().delete(self._selectbox)
        self._selection = None
        self._y = 10

    def set_chart(self, chart):
        if False:
            i = 10
            return i + 15
        self.clear()
        self._chart = chart
        self.update()

    def set_grammar(self, grammar):
        if False:
            for i in range(10):
                print('nop')
        self.clear()
        self._grammar = grammar
        self.update()

    def destroy(self, *e):
        if False:
            for i in range(10):
                print('nop')
        if self._root is None:
            return
        try:
            self._root.destroy()
        except:
            pass
        self._root = None

    def pack(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._root.pack(*args, **kwargs)

class ChartComparer:
    """

    :ivar _root: The root window

    :ivar _charts: A dictionary mapping names to charts.  When
        charts are loaded, they are added to this dictionary.

    :ivar _left_chart: The left ``Chart``.
    :ivar _left_name: The name ``_left_chart`` (derived from filename)
    :ivar _left_matrix: The ``ChartMatrixView`` for ``_left_chart``
    :ivar _left_selector: The drop-down ``MutableOptionsMenu`` used
          to select ``_left_chart``.

    :ivar _right_chart: The right ``Chart``.
    :ivar _right_name: The name ``_right_chart`` (derived from filename)
    :ivar _right_matrix: The ``ChartMatrixView`` for ``_right_chart``
    :ivar _right_selector: The drop-down ``MutableOptionsMenu`` used
          to select ``_right_chart``.

    :ivar _out_chart: The out ``Chart``.
    :ivar _out_name: The name ``_out_chart`` (derived from filename)
    :ivar _out_matrix: The ``ChartMatrixView`` for ``_out_chart``
    :ivar _out_label: The label for ``_out_chart``.

    :ivar _op_label: A Label containing the most recent operation.
    """
    _OPSYMBOL = {'-': '-', 'and': SymbolWidget.SYMBOLS['intersection'], 'or': SymbolWidget.SYMBOLS['union']}

    def __init__(self, *chart_filenames):
        if False:
            for i in range(10):
                print('nop')
        faketok = [''] * 8
        self._emptychart = Chart(faketok)
        self._left_name = 'None'
        self._right_name = 'None'
        self._left_chart = self._emptychart
        self._right_chart = self._emptychart
        self._charts = {'None': self._emptychart}
        self._out_chart = self._emptychart
        self._operator = None
        self._root = Tk()
        self._root.title('Chart Comparison')
        self._root.bind('<Control-q>', self.destroy)
        self._root.bind('<Control-x>', self.destroy)
        self._init_menubar(self._root)
        self._init_chartviews(self._root)
        self._init_divider(self._root)
        self._init_buttons(self._root)
        self._init_bindings(self._root)
        for filename in chart_filenames:
            self.load_chart(filename)

    def destroy(self, *e):
        if False:
            for i in range(10):
                print('nop')
        if self._root is None:
            return
        try:
            self._root.destroy()
        except:
            pass
        self._root = None

    def mainloop(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return
        self._root.mainloop(*args, **kwargs)

    def _init_menubar(self, root):
        if False:
            i = 10
            return i + 15
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label='Load Chart', accelerator='Ctrl-o', underline=0, command=self.load_chart_dialog)
        filemenu.add_command(label='Save Output', accelerator='Ctrl-s', underline=0, command=self.save_chart_dialog)
        filemenu.add_separator()
        filemenu.add_command(label='Exit', underline=1, command=self.destroy, accelerator='Ctrl-x')
        menubar.add_cascade(label='File', underline=0, menu=filemenu)
        opmenu = Menu(menubar, tearoff=0)
        opmenu.add_command(label='Intersection', command=self._intersection, accelerator='+')
        opmenu.add_command(label='Union', command=self._union, accelerator='*')
        opmenu.add_command(label='Difference', command=self._difference, accelerator='-')
        opmenu.add_separator()
        opmenu.add_command(label='Swap Charts', command=self._swapcharts)
        menubar.add_cascade(label='Compare', underline=0, menu=opmenu)
        self._root.config(menu=menubar)

    def _init_divider(self, root):
        if False:
            for i in range(10):
                print('nop')
        divider = Frame(root, border=2, relief='sunken')
        divider.pack(side='top', fill='x', ipady=2)

    def _init_chartviews(self, root):
        if False:
            print('Hello World!')
        opfont = ('symbol', -36)
        eqfont = ('helvetica', -36)
        frame = Frame(root, background='#c0c0c0')
        frame.pack(side='top', expand=1, fill='both')
        cv1_frame = Frame(frame, border=3, relief='groove')
        cv1_frame.pack(side='left', padx=8, pady=7, expand=1, fill='both')
        self._left_selector = MutableOptionMenu(cv1_frame, list(self._charts.keys()), command=self._select_left)
        self._left_selector.pack(side='top', pady=5, fill='x')
        self._left_matrix = ChartMatrixView(cv1_frame, self._emptychart, toplevel=False, show_numedges=True)
        self._left_matrix.pack(side='bottom', padx=5, pady=5, expand=1, fill='both')
        self._left_matrix.add_callback('select', self.select_edge)
        self._left_matrix.add_callback('select_cell', self.select_cell)
        self._left_matrix.inactivate()
        self._op_label = Label(frame, text=' ', width=3, background='#c0c0c0', font=opfont)
        self._op_label.pack(side='left', padx=5, pady=5)
        cv2_frame = Frame(frame, border=3, relief='groove')
        cv2_frame.pack(side='left', padx=8, pady=7, expand=1, fill='both')
        self._right_selector = MutableOptionMenu(cv2_frame, list(self._charts.keys()), command=self._select_right)
        self._right_selector.pack(side='top', pady=5, fill='x')
        self._right_matrix = ChartMatrixView(cv2_frame, self._emptychart, toplevel=False, show_numedges=True)
        self._right_matrix.pack(side='bottom', padx=5, pady=5, expand=1, fill='both')
        self._right_matrix.add_callback('select', self.select_edge)
        self._right_matrix.add_callback('select_cell', self.select_cell)
        self._right_matrix.inactivate()
        Label(frame, text='=', width=3, background='#c0c0c0', font=eqfont).pack(side='left', padx=5, pady=5)
        out_frame = Frame(frame, border=3, relief='groove')
        out_frame.pack(side='left', padx=8, pady=7, expand=1, fill='both')
        self._out_label = Label(out_frame, text='Output')
        self._out_label.pack(side='top', pady=9)
        self._out_matrix = ChartMatrixView(out_frame, self._emptychart, toplevel=False, show_numedges=True)
        self._out_matrix.pack(side='bottom', padx=5, pady=5, expand=1, fill='both')
        self._out_matrix.add_callback('select', self.select_edge)
        self._out_matrix.add_callback('select_cell', self.select_cell)
        self._out_matrix.inactivate()

    def _init_buttons(self, root):
        if False:
            while True:
                i = 10
        buttons = Frame(root)
        buttons.pack(side='bottom', pady=5, fill='x', expand=0)
        Button(buttons, text='Intersection', command=self._intersection).pack(side='left')
        Button(buttons, text='Union', command=self._union).pack(side='left')
        Button(buttons, text='Difference', command=self._difference).pack(side='left')
        Frame(buttons, width=20).pack(side='left')
        Button(buttons, text='Swap Charts', command=self._swapcharts).pack(side='left')
        Button(buttons, text='Detach Output', command=self._detach_out).pack(side='right')

    def _init_bindings(self, root):
        if False:
            i = 10
            return i + 15
        root.bind('<Control-o>', self.load_chart_dialog)

    def _select_left(self, name):
        if False:
            return 10
        self._left_name = name
        self._left_chart = self._charts[name]
        self._left_matrix.set_chart(self._left_chart)
        if name == 'None':
            self._left_matrix.inactivate()
        self._apply_op()

    def _select_right(self, name):
        if False:
            for i in range(10):
                print('nop')
        self._right_name = name
        self._right_chart = self._charts[name]
        self._right_matrix.set_chart(self._right_chart)
        if name == 'None':
            self._right_matrix.inactivate()
        self._apply_op()

    def _apply_op(self):
        if False:
            print('Hello World!')
        if self._operator == '-':
            self._difference()
        elif self._operator == 'or':
            self._union()
        elif self._operator == 'and':
            self._intersection()
    CHART_FILE_TYPES = [('Pickle file', '.pickle'), ('All files', '*')]

    def save_chart_dialog(self, *args):
        if False:
            for i in range(10):
                print('nop')
        filename = asksaveasfilename(filetypes=self.CHART_FILE_TYPES, defaultextension='.pickle')
        if not filename:
            return
        try:
            with open(filename, 'wb') as outfile:
                pickle.dump(self._out_chart, outfile)
        except Exception as e:
            showerror('Error Saving Chart', f'Unable to open file: {filename!r}\n{e}')

    def load_chart_dialog(self, *args):
        if False:
            while True:
                i = 10
        filename = askopenfilename(filetypes=self.CHART_FILE_TYPES, defaultextension='.pickle')
        if not filename:
            return
        try:
            self.load_chart(filename)
        except Exception as e:
            showerror('Error Loading Chart', f'Unable to open file: {filename!r}\n{e}')

    def load_chart(self, filename):
        if False:
            for i in range(10):
                print('nop')
        with open(filename, 'rb') as infile:
            chart = pickle.load(infile)
        name = os.path.basename(filename)
        if name.endswith('.pickle'):
            name = name[:-7]
        if name.endswith('.chart'):
            name = name[:-6]
        self._charts[name] = chart
        self._left_selector.add(name)
        self._right_selector.add(name)
        if self._left_chart is self._emptychart:
            self._left_selector.set(name)
        elif self._right_chart is self._emptychart:
            self._right_selector.set(name)

    def _update_chartviews(self):
        if False:
            return 10
        self._left_matrix.update()
        self._right_matrix.update()
        self._out_matrix.update()

    def select_edge(self, edge):
        if False:
            print('Hello World!')
        if edge in self._left_chart:
            self._left_matrix.markonly_edge(edge)
        else:
            self._left_matrix.unmark_edge()
        if edge in self._right_chart:
            self._right_matrix.markonly_edge(edge)
        else:
            self._right_matrix.unmark_edge()
        if edge in self._out_chart:
            self._out_matrix.markonly_edge(edge)
        else:
            self._out_matrix.unmark_edge()

    def select_cell(self, i, j):
        if False:
            for i in range(10):
                print('nop')
        self._left_matrix.select_cell(i, j)
        self._right_matrix.select_cell(i, j)
        self._out_matrix.select_cell(i, j)

    def _difference(self):
        if False:
            while True:
                i = 10
        if not self._checkcompat():
            return
        out_chart = Chart(self._left_chart.tokens())
        for edge in self._left_chart:
            if edge not in self._right_chart:
                out_chart.insert(edge, [])
        self._update('-', out_chart)

    def _intersection(self):
        if False:
            print('Hello World!')
        if not self._checkcompat():
            return
        out_chart = Chart(self._left_chart.tokens())
        for edge in self._left_chart:
            if edge in self._right_chart:
                out_chart.insert(edge, [])
        self._update('and', out_chart)

    def _union(self):
        if False:
            return 10
        if not self._checkcompat():
            return
        out_chart = Chart(self._left_chart.tokens())
        for edge in self._left_chart:
            out_chart.insert(edge, [])
        for edge in self._right_chart:
            out_chart.insert(edge, [])
        self._update('or', out_chart)

    def _swapcharts(self):
        if False:
            i = 10
            return i + 15
        (left, right) = (self._left_name, self._right_name)
        self._left_selector.set(right)
        self._right_selector.set(left)

    def _checkcompat(self):
        if False:
            for i in range(10):
                print('nop')
        if self._left_chart.tokens() != self._right_chart.tokens() or self._left_chart.property_names() != self._right_chart.property_names() or self._left_chart == self._emptychart or (self._right_chart == self._emptychart):
            self._out_chart = self._emptychart
            self._out_matrix.set_chart(self._out_chart)
            self._out_matrix.inactivate()
            self._out_label['text'] = 'Output'
            return False
        else:
            return True

    def _update(self, operator, out_chart):
        if False:
            i = 10
            return i + 15
        self._operator = operator
        self._op_label['text'] = self._OPSYMBOL[operator]
        self._out_chart = out_chart
        self._out_matrix.set_chart(out_chart)
        self._out_label['text'] = '{} {} {}'.format(self._left_name, self._operator, self._right_name)

    def _clear_out_chart(self):
        if False:
            while True:
                i = 10
        self._out_chart = self._emptychart
        self._out_matrix.set_chart(self._out_chart)
        self._op_label['text'] = ' '
        self._out_matrix.inactivate()

    def _detach_out(self):
        if False:
            i = 10
            return i + 15
        ChartMatrixView(self._root, self._out_chart, title=self._out_label['text'])

class ChartView:
    """
    A component for viewing charts.  This is used by ``ChartParserApp`` to
    allow students to interactively experiment with various chart
    parsing techniques.  It is also used by ``Chart.draw()``.

    :ivar _chart: The chart that we are giving a view of.  This chart
       may be modified; after it is modified, you should call
       ``update``.
    :ivar _sentence: The list of tokens that the chart spans.

    :ivar _root: The root window.
    :ivar _chart_canvas: The canvas we're using to display the chart
        itself.
    :ivar _tree_canvas: The canvas we're using to display the tree
        that each edge spans.  May be None, if we're not displaying
        trees.
    :ivar _sentence_canvas: The canvas we're using to display the sentence
        text.  May be None, if we're not displaying the sentence text.
    :ivar _edgetags: A dictionary mapping from edges to the tags of
        the canvas elements (lines, etc) used to display that edge.
        The values of this dictionary have the form
        ``(linetag, rhstag1, dottag, rhstag2, lhstag)``.
    :ivar _treetags: A list of all the tags that make up the tree;
        used to erase the tree (without erasing the loclines).
    :ivar _chart_height: The height of the chart canvas.
    :ivar _sentence_height: The height of the sentence canvas.
    :ivar _tree_height: The height of the tree

    :ivar _text_height: The height of a text string (in the normal
        font).

    :ivar _edgelevels: A list of edges at each level of the chart (the
        top level is the 0th element).  This list is used to remember
        where edges should be drawn; and to make sure that no edges
        are overlapping on the chart view.

    :ivar _unitsize: Pixel size of one unit (from the location).  This
       is determined by the span of the chart's location, and the
       width of the chart display canvas.

    :ivar _fontsize: The current font size

    :ivar _marks: A dictionary from edges to marks.  Marks are
        strings, specifying colors (e.g. 'green').
    """
    _LEAF_SPACING = 10
    _MARGIN = 10
    _TREE_LEVEL_SIZE = 12
    _CHART_LEVEL_SIZE = 40

    def __init__(self, chart, root=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new ``Chart`` display.\n        '
        draw_tree = kw.get('draw_tree', 0)
        draw_sentence = kw.get('draw_sentence', 1)
        self._fontsize = kw.get('fontsize', -12)
        self._chart = chart
        self._callbacks = {}
        self._edgelevels = []
        self._edgetags = {}
        self._marks = {}
        self._treetoks = []
        self._treetoks_edge = None
        self._treetoks_index = 0
        self._tree_tags = []
        self._compact = 0
        if root is None:
            top = Tk()
            top.title('Chart View')

            def destroy1(e, top=top):
                if False:
                    print('Hello World!')
                top.destroy()

            def destroy2(top=top):
                if False:
                    print('Hello World!')
                top.destroy()
            top.bind('q', destroy1)
            b = Button(top, text='Done', command=destroy2)
            b.pack(side='bottom')
            self._root = top
        else:
            self._root = root
        self._init_fonts(root)
        (self._chart_sb, self._chart_canvas) = self._sb_canvas(self._root)
        self._chart_canvas['height'] = 300
        self._chart_canvas['closeenough'] = 15
        if draw_sentence:
            cframe = Frame(self._root, relief='sunk', border=2)
            cframe.pack(fill='both', side='bottom')
            self._sentence_canvas = Canvas(cframe, height=50)
            self._sentence_canvas['background'] = '#e0e0e0'
            self._sentence_canvas.pack(fill='both')
        else:
            self._sentence_canvas = None
        if draw_tree:
            (sb, canvas) = self._sb_canvas(self._root, 'n', 'x')
            (self._tree_sb, self._tree_canvas) = (sb, canvas)
            self._tree_canvas['height'] = 200
        else:
            self._tree_canvas = None
        self._analyze()
        self.draw()
        self._resize()
        self._grow()
        self._chart_canvas.bind('<Configure>', self._configure)

    def _init_fonts(self, root):
        if False:
            i = 10
            return i + 15
        self._boldfont = Font(family='helvetica', weight='bold', size=self._fontsize)
        self._font = Font(family='helvetica', size=self._fontsize)
        self._sysfont = Font(font=Button()['font'])
        root.option_add('*Font', self._sysfont)

    def _sb_canvas(self, root, expand='y', fill='both', side='bottom'):
        if False:
            i = 10
            return i + 15
        '\n        Helper for __init__: construct a canvas with a scrollbar.\n        '
        cframe = Frame(root, relief='sunk', border=2)
        cframe.pack(fill=fill, expand=expand, side=side)
        canvas = Canvas(cframe, background='#e0e0e0')
        sb = Scrollbar(cframe, orient='vertical')
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill=fill, expand='yes')
        sb['command'] = canvas.yview
        canvas['yscrollcommand'] = sb.set
        return (sb, canvas)

    def scroll_up(self, *e):
        if False:
            for i in range(10):
                print('nop')
        self._chart_canvas.yview('scroll', -1, 'units')

    def scroll_down(self, *e):
        if False:
            for i in range(10):
                print('nop')
        self._chart_canvas.yview('scroll', 1, 'units')

    def page_up(self, *e):
        if False:
            for i in range(10):
                print('nop')
        self._chart_canvas.yview('scroll', -1, 'pages')

    def page_down(self, *e):
        if False:
            return 10
        self._chart_canvas.yview('scroll', 1, 'pages')

    def _grow(self):
        if False:
            print('Hello World!')
        '\n        Grow the window, if necessary\n        '
        N = self._chart.num_leaves()
        width = max(int(self._chart_canvas['width']), N * self._unitsize + ChartView._MARGIN * 2)
        self._chart_canvas.configure(width=width)
        self._chart_canvas.configure(height=self._chart_canvas['height'])
        self._unitsize = (width - 2 * ChartView._MARGIN) / N
        if self._sentence_canvas is not None:
            self._sentence_canvas['height'] = self._sentence_height

    def set_font_size(self, size):
        if False:
            for i in range(10):
                print('nop')
        self._font.configure(size=-abs(size))
        self._boldfont.configure(size=-abs(size))
        self._sysfont.configure(size=-abs(size))
        self._analyze()
        self._grow()
        self.draw()

    def get_font_size(self):
        if False:
            for i in range(10):
                print('nop')
        return abs(self._fontsize)

    def _configure(self, e):
        if False:
            print('Hello World!')
        '\n        The configure callback.  This is called whenever the window is\n        resized.  It is also called when the window is first mapped.\n        It figures out the unit size, and redraws the contents of each\n        canvas.\n        '
        N = self._chart.num_leaves()
        self._unitsize = (e.width - 2 * ChartView._MARGIN) / N
        self.draw()

    def update(self, chart=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Draw any edges that have not been drawn.  This is typically\n        called when a after modifies the canvas that a CanvasView is\n        displaying.  ``update`` will cause any edges that have been\n        added to the chart to be drawn.\n\n        If update is given a ``chart`` argument, then it will replace\n        the current chart with the given chart.\n        '
        if chart is not None:
            self._chart = chart
            self._edgelevels = []
            self._marks = {}
            self._analyze()
            self._grow()
            self.draw()
            self.erase_tree()
            self._resize()
        else:
            for edge in self._chart:
                if edge not in self._edgetags:
                    self._add_edge(edge)
            self._resize()

    def _edge_conflict(self, edge, lvl):
        if False:
            print('Hello World!')
        '\n        Return True if the given edge overlaps with any edge on the given\n        level.  This is used by _add_edge to figure out what level a\n        new edge should be added to.\n        '
        (s1, e1) = edge.span()
        for otheredge in self._edgelevels[lvl]:
            (s2, e2) = otheredge.span()
            if s1 <= s2 < e1 or s2 <= s1 < e2 or s1 == s2 == e1 == e2:
                return True
        return False

    def _analyze_edge(self, edge):
        if False:
            while True:
                i = 10
        '\n        Given a new edge, recalculate:\n\n            - _text_height\n            - _unitsize (if the edge text is too big for the current\n              _unitsize, then increase _unitsize)\n        '
        c = self._chart_canvas
        if isinstance(edge, TreeEdge):
            lhs = edge.lhs()
            rhselts = []
            for elt in edge.rhs():
                if isinstance(elt, Nonterminal):
                    rhselts.append(str(elt.symbol()))
                else:
                    rhselts.append(repr(elt))
            rhs = ' '.join(rhselts)
        else:
            lhs = edge.lhs()
            rhs = ''
        for s in (lhs, rhs):
            tag = c.create_text(0, 0, text=s, font=self._boldfont, anchor='nw', justify='left')
            bbox = c.bbox(tag)
            c.delete(tag)
            width = bbox[2]
            edgelen = max(edge.length(), 1)
            self._unitsize = max(self._unitsize, width / edgelen)
            self._text_height = max(self._text_height, bbox[3] - bbox[1])

    def _add_edge(self, edge, minlvl=0):
        if False:
            i = 10
            return i + 15
        '\n        Add a single edge to the ChartView:\n\n            - Call analyze_edge to recalculate display parameters\n            - Find an available level\n            - Call _draw_edge\n        '
        if isinstance(edge, LeafEdge):
            return
        if edge in self._edgetags:
            return
        self._analyze_edge(edge)
        self._grow()
        if not self._compact:
            self._edgelevels.append([edge])
            lvl = len(self._edgelevels) - 1
            self._draw_edge(edge, lvl)
            self._resize()
            return
        lvl = 0
        while True:
            while lvl >= len(self._edgelevels):
                self._edgelevels.append([])
                self._resize()
            if lvl >= minlvl and (not self._edge_conflict(edge, lvl)):
                self._edgelevels[lvl].append(edge)
                break
            lvl += 1
        self._draw_edge(edge, lvl)

    def view_edge(self, edge):
        if False:
            print('Hello World!')
        level = None
        for i in range(len(self._edgelevels)):
            if edge in self._edgelevels[i]:
                level = i
                break
        if level is None:
            return
        y = (level + 1) * self._chart_level_size
        dy = self._text_height + 10
        self._chart_canvas.yview('moveto', 1.0)
        if self._chart_height != 0:
            self._chart_canvas.yview('moveto', (y - dy) / self._chart_height)

    def _draw_edge(self, edge, lvl):
        if False:
            while True:
                i = 10
        '\n        Draw a single edge on the ChartView.\n        '
        c = self._chart_canvas
        x1 = edge.start() * self._unitsize + ChartView._MARGIN
        x2 = edge.end() * self._unitsize + ChartView._MARGIN
        if x2 == x1:
            x2 += max(4, self._unitsize / 5)
        y = (lvl + 1) * self._chart_level_size
        linetag = c.create_line(x1, y, x2, y, arrow='last', width=3)
        if isinstance(edge, TreeEdge):
            rhs = []
            for elt in edge.rhs():
                if isinstance(elt, Nonterminal):
                    rhs.append(str(elt.symbol()))
                else:
                    rhs.append(repr(elt))
            pos = edge.dot()
        else:
            rhs = []
            pos = 0
        rhs1 = ' '.join(rhs[:pos])
        rhs2 = ' '.join(rhs[pos:])
        rhstag1 = c.create_text(x1 + 3, y, text=rhs1, font=self._font, anchor='nw')
        dotx = c.bbox(rhstag1)[2] + 6
        doty = (c.bbox(rhstag1)[1] + c.bbox(rhstag1)[3]) / 2
        dottag = c.create_oval(dotx - 2, doty - 2, dotx + 2, doty + 2)
        rhstag2 = c.create_text(dotx + 6, y, text=rhs2, font=self._font, anchor='nw')
        lhstag = c.create_text((x1 + x2) / 2, y, text=str(edge.lhs()), anchor='s', font=self._boldfont)
        self._edgetags[edge] = (linetag, rhstag1, dottag, rhstag2, lhstag)

        def cb(event, self=self, edge=edge):
            if False:
                while True:
                    i = 10
            self._fire_callbacks('select', edge)
        c.tag_bind(rhstag1, '<Button-1>', cb)
        c.tag_bind(rhstag2, '<Button-1>', cb)
        c.tag_bind(linetag, '<Button-1>', cb)
        c.tag_bind(dottag, '<Button-1>', cb)
        c.tag_bind(lhstag, '<Button-1>', cb)
        self._color_edge(edge)

    def _color_edge(self, edge, linecolor=None, textcolor=None):
        if False:
            i = 10
            return i + 15
        '\n        Color in an edge with the given colors.\n        If no colors are specified, use intelligent defaults\n        (dependent on selection, etc.)\n        '
        if edge not in self._edgetags:
            return
        c = self._chart_canvas
        if linecolor is not None and textcolor is not None:
            if edge in self._marks:
                linecolor = self._marks[edge]
            tags = self._edgetags[edge]
            c.itemconfig(tags[0], fill=linecolor)
            c.itemconfig(tags[1], fill=textcolor)
            c.itemconfig(tags[2], fill=textcolor, outline=textcolor)
            c.itemconfig(tags[3], fill=textcolor)
            c.itemconfig(tags[4], fill=textcolor)
            return
        else:
            N = self._chart.num_leaves()
            if edge in self._marks:
                self._color_edge(self._marks[edge])
            if edge.is_complete() and edge.span() == (0, N):
                self._color_edge(edge, '#084', '#042')
            elif isinstance(edge, LeafEdge):
                self._color_edge(edge, '#48c', '#246')
            else:
                self._color_edge(edge, '#00f', '#008')

    def mark_edge(self, edge, mark='#0df'):
        if False:
            i = 10
            return i + 15
        '\n        Mark an edge\n        '
        self._marks[edge] = mark
        self._color_edge(edge)

    def unmark_edge(self, edge=None):
        if False:
            i = 10
            return i + 15
        '\n        Unmark an edge (or all edges)\n        '
        if edge is None:
            old_marked_edges = list(self._marks.keys())
            self._marks = {}
            for edge in old_marked_edges:
                self._color_edge(edge)
        else:
            del self._marks[edge]
            self._color_edge(edge)

    def markonly_edge(self, edge, mark='#0df'):
        if False:
            i = 10
            return i + 15
        self.unmark_edge()
        self.mark_edge(edge, mark)

    def _analyze(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Analyze the sentence string, to figure out how big a unit needs\n        to be, How big the tree should be, etc.\n        '
        unitsize = 70
        text_height = 0
        c = self._chart_canvas
        for leaf in self._chart.leaves():
            tag = c.create_text(0, 0, text=repr(leaf), font=self._font, anchor='nw', justify='left')
            bbox = c.bbox(tag)
            c.delete(tag)
            width = bbox[2] + ChartView._LEAF_SPACING
            unitsize = max(width, unitsize)
            text_height = max(text_height, bbox[3] - bbox[1])
        self._unitsize = unitsize
        self._text_height = text_height
        self._sentence_height = self._text_height + 2 * ChartView._MARGIN
        for edge in self._chart.edges():
            self._analyze_edge(edge)
        self._chart_level_size = self._text_height * 2
        self._tree_height = 3 * (ChartView._TREE_LEVEL_SIZE + self._text_height)
        self._resize()

    def _resize(self):
        if False:
            i = 10
            return i + 15
        '\n        Update the scroll-regions for each canvas.  This ensures that\n        everything is within a scroll-region, so the user can use the\n        scrollbars to view the entire display.  This does *not*\n        resize the window.\n        '
        c = self._chart_canvas
        width = self._chart.num_leaves() * self._unitsize + ChartView._MARGIN * 2
        levels = len(self._edgelevels)
        self._chart_height = (levels + 2) * self._chart_level_size
        c['scrollregion'] = (0, 0, width, self._chart_height)
        if self._tree_canvas:
            self._tree_canvas['scrollregion'] = (0, 0, width, self._tree_height)

    def _draw_loclines(self):
        if False:
            return 10
        '\n        Draw location lines.  These are vertical gridlines used to\n        show where each location unit is.\n        '
        BOTTOM = 50000
        c1 = self._tree_canvas
        c2 = self._sentence_canvas
        c3 = self._chart_canvas
        margin = ChartView._MARGIN
        self._loclines = []
        for i in range(0, self._chart.num_leaves() + 1):
            x = i * self._unitsize + margin
            if c1:
                t1 = c1.create_line(x, 0, x, BOTTOM)
                c1.tag_lower(t1)
            if c2:
                t2 = c2.create_line(x, 0, x, self._sentence_height)
                c2.tag_lower(t2)
            t3 = c3.create_line(x, 0, x, BOTTOM)
            c3.tag_lower(t3)
            t4 = c3.create_text(x + 2, 0, text=repr(i), anchor='nw', font=self._font)
            c3.tag_lower(t4)
            if i % 2 == 0:
                if c1:
                    c1.itemconfig(t1, fill='gray60')
                if c2:
                    c2.itemconfig(t2, fill='gray60')
                c3.itemconfig(t3, fill='gray60')
            else:
                if c1:
                    c1.itemconfig(t1, fill='gray80')
                if c2:
                    c2.itemconfig(t2, fill='gray80')
                c3.itemconfig(t3, fill='gray80')

    def _draw_sentence(self):
        if False:
            for i in range(10):
                print('nop')
        'Draw the sentence string.'
        if self._chart.num_leaves() == 0:
            return
        c = self._sentence_canvas
        margin = ChartView._MARGIN
        y = ChartView._MARGIN
        for (i, leaf) in enumerate(self._chart.leaves()):
            x1 = i * self._unitsize + margin
            x2 = x1 + self._unitsize
            x = (x1 + x2) / 2
            tag = c.create_text(x, y, text=repr(leaf), font=self._font, anchor='n', justify='left')
            bbox = c.bbox(tag)
            rt = c.create_rectangle(x1 + 2, bbox[1] - ChartView._LEAF_SPACING / 2, x2 - 2, bbox[3] + ChartView._LEAF_SPACING / 2, fill='#f0f0f0', outline='#f0f0f0')
            c.tag_lower(rt)

    def erase_tree(self):
        if False:
            for i in range(10):
                print('nop')
        for tag in self._tree_tags:
            self._tree_canvas.delete(tag)
        self._treetoks = []
        self._treetoks_edge = None
        self._treetoks_index = 0

    def draw_tree(self, edge=None):
        if False:
            for i in range(10):
                print('nop')
        if edge is None and self._treetoks_edge is None:
            return
        if edge is None:
            edge = self._treetoks_edge
        if self._treetoks_edge != edge:
            self._treetoks = [t for t in self._chart.trees(edge) if isinstance(t, Tree)]
            self._treetoks_edge = edge
            self._treetoks_index = 0
        if len(self._treetoks) == 0:
            return
        for tag in self._tree_tags:
            self._tree_canvas.delete(tag)
        tree = self._treetoks[self._treetoks_index]
        self._draw_treetok(tree, edge.start())
        self._draw_treecycle()
        w = self._chart.num_leaves() * self._unitsize + 2 * ChartView._MARGIN
        h = tree.height() * (ChartView._TREE_LEVEL_SIZE + self._text_height)
        self._tree_canvas['scrollregion'] = (0, 0, w, h)

    def cycle_tree(self):
        if False:
            for i in range(10):
                print('nop')
        self._treetoks_index = (self._treetoks_index + 1) % len(self._treetoks)
        self.draw_tree(self._treetoks_edge)

    def _draw_treecycle(self):
        if False:
            i = 10
            return i + 15
        if len(self._treetoks) <= 1:
            return
        label = '%d Trees' % len(self._treetoks)
        c = self._tree_canvas
        margin = ChartView._MARGIN
        right = self._chart.num_leaves() * self._unitsize + margin - 2
        tag = c.create_text(right, 2, anchor='ne', text=label, font=self._boldfont)
        self._tree_tags.append(tag)
        (_, _, _, y) = c.bbox(tag)
        for i in range(len(self._treetoks)):
            x = right - 20 * (len(self._treetoks) - i - 1)
            if i == self._treetoks_index:
                fill = '#084'
            else:
                fill = '#fff'
            tag = c.create_polygon(x, y + 10, x - 5, y, x - 10, y + 10, fill=fill, outline='black')
            self._tree_tags.append(tag)

            def cb(event, self=self, i=i):
                if False:
                    return 10
                self._treetoks_index = i
                self.draw_tree()
            c.tag_bind(tag, '<Button-1>', cb)

    def _draw_treetok(self, treetok, index, depth=0):
        if False:
            while True:
                i = 10
        '\n        :param index: The index of the first leaf in the tree.\n        :return: The index of the first leaf after the tree.\n        '
        c = self._tree_canvas
        margin = ChartView._MARGIN
        child_xs = []
        for child in treetok:
            if isinstance(child, Tree):
                (child_x, index) = self._draw_treetok(child, index, depth + 1)
                child_xs.append(child_x)
            else:
                child_xs.append((2 * index + 1) * self._unitsize / 2 + margin)
                index += 1
        if child_xs:
            nodex = sum(child_xs) / len(child_xs)
        else:
            nodex = (2 * index + 1) * self._unitsize / 2 + margin
            index += 1
        nodey = depth * (ChartView._TREE_LEVEL_SIZE + self._text_height)
        tag = c.create_text(nodex, nodey, anchor='n', justify='center', text=str(treetok.label()), fill='#042', font=self._boldfont)
        self._tree_tags.append(tag)
        childy = nodey + ChartView._TREE_LEVEL_SIZE + self._text_height
        for (childx, child) in zip(child_xs, treetok):
            if isinstance(child, Tree) and child:
                tag = c.create_line(nodex, nodey + self._text_height, childx, childy, width=2, fill='#084')
                self._tree_tags.append(tag)
            if isinstance(child, Tree) and (not child):
                tag = c.create_line(nodex, nodey + self._text_height, childx, childy, width=2, fill='#048', dash='2 3')
                self._tree_tags.append(tag)
            if not isinstance(child, Tree):
                tag = c.create_line(nodex, nodey + self._text_height, childx, 10000, width=2, fill='#084')
                self._tree_tags.append(tag)
        return (nodex, index)

    def draw(self):
        if False:
            i = 10
            return i + 15
        '\n        Draw everything (from scratch).\n        '
        if self._tree_canvas:
            self._tree_canvas.delete('all')
            self.draw_tree()
        if self._sentence_canvas:
            self._sentence_canvas.delete('all')
            self._draw_sentence()
        self._chart_canvas.delete('all')
        self._edgetags = {}
        for lvl in range(len(self._edgelevels)):
            for edge in self._edgelevels[lvl]:
                self._draw_edge(edge, lvl)
        for edge in self._chart:
            self._add_edge(edge)
        self._draw_loclines()

    def add_callback(self, event, func):
        if False:
            i = 10
            return i + 15
        self._callbacks.setdefault(event, {})[func] = 1

    def remove_callback(self, event, func=None):
        if False:
            for i in range(10):
                print('nop')
        if func is None:
            del self._callbacks[event]
        else:
            try:
                del self._callbacks[event][func]
            except:
                pass

    def _fire_callbacks(self, event, *args):
        if False:
            while True:
                i = 10
        if event not in self._callbacks:
            return
        for cb_func in list(self._callbacks[event].keys()):
            cb_func(*args)

class EdgeRule:
    """
    To create an edge rule, make an empty base class that uses
    EdgeRule as the first base class, and the basic rule as the
    second base class.  (Order matters!)
    """

    def __init__(self, edge):
        if False:
            print('Hello World!')
        super = self.__class__.__bases__[1]
        self._edge = edge
        self.NUM_EDGES = super.NUM_EDGES - 1

    def apply(self, chart, grammar, *edges):
        if False:
            for i in range(10):
                print('nop')
        super = self.__class__.__bases__[1]
        edges += (self._edge,)
        yield from super.apply(self, chart, grammar, *edges)

    def __str__(self):
        if False:
            print('Hello World!')
        super = self.__class__.__bases__[1]
        return super.__str__(self)

class TopDownPredictEdgeRule(EdgeRule, TopDownPredictRule):
    pass

class BottomUpEdgeRule(EdgeRule, BottomUpPredictRule):
    pass

class BottomUpLeftCornerEdgeRule(EdgeRule, BottomUpPredictCombineRule):
    pass

class FundamentalEdgeRule(EdgeRule, SingleEdgeFundamentalRule):
    pass

class ChartParserApp:

    def __init__(self, grammar, tokens, title='Chart Parser Application'):
        if False:
            print('Hello World!')
        self._init_parser(grammar, tokens)
        self._root = None
        try:
            self._root = Tk()
            self._root.title(title)
            self._root.bind('<Control-q>', self.destroy)
            frame3 = Frame(self._root)
            frame2 = Frame(self._root)
            frame1 = Frame(self._root)
            frame3.pack(side='bottom', fill='none')
            frame2.pack(side='bottom', fill='x')
            frame1.pack(side='bottom', fill='both', expand=1)
            self._init_fonts(self._root)
            self._init_animation()
            self._init_chartview(frame1)
            self._init_rulelabel(frame2)
            self._init_buttons(frame3)
            self._init_menubar()
            self._matrix = None
            self._results = None
            self._init_bindings()
        except:
            print('Error creating Tree View')
            self.destroy()
            raise

    def destroy(self, *args):
        if False:
            for i in range(10):
                print('nop')
        if self._root is None:
            return
        self._root.destroy()
        self._root = None

    def mainloop(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Enter the Tkinter mainloop.  This function must be called if\n        this demo is created from a non-interactive program (e.g.\n        from a secript); otherwise, the demo will close as soon as\n        the script completes.\n        '
        if in_idle():
            return
        self._root.mainloop(*args, **kwargs)

    def _init_parser(self, grammar, tokens):
        if False:
            print('Hello World!')
        self._grammar = grammar
        self._tokens = tokens
        self._reset_parser()

    def _reset_parser(self):
        if False:
            for i in range(10):
                print('nop')
        self._cp = SteppingChartParser(self._grammar)
        self._cp.initialize(self._tokens)
        self._chart = self._cp.chart()
        for _new_edge in LeafInitRule().apply(self._chart, self._grammar):
            pass
        self._cpstep = self._cp.step()
        self._selection = None

    def _init_fonts(self, root):
        if False:
            while True:
                i = 10
        self._sysfont = Font(font=Button()['font'])
        root.option_add('*Font', self._sysfont)
        self._size = IntVar(root)
        self._size.set(self._sysfont.cget('size'))
        self._boldfont = Font(family='helvetica', weight='bold', size=self._size.get())
        self._font = Font(family='helvetica', size=self._size.get())

    def _init_animation(self):
        if False:
            return 10
        self._step = IntVar(self._root)
        self._step.set(1)
        self._animate = IntVar(self._root)
        self._animate.set(3)
        self._animating = 0

    def _init_chartview(self, parent):
        if False:
            while True:
                i = 10
        self._cv = ChartView(self._chart, parent, draw_tree=1, draw_sentence=1)
        self._cv.add_callback('select', self._click_cv_edge)

    def _init_rulelabel(self, parent):
        if False:
            return 10
        ruletxt = 'Last edge generated by:'
        self._rulelabel1 = Label(parent, text=ruletxt, font=self._boldfont)
        self._rulelabel2 = Label(parent, width=40, relief='groove', anchor='w', font=self._boldfont)
        self._rulelabel1.pack(side='left')
        self._rulelabel2.pack(side='left')
        step = Checkbutton(parent, variable=self._step, text='Step')
        step.pack(side='right')

    def _init_buttons(self, parent):
        if False:
            i = 10
            return i + 15
        frame1 = Frame(parent)
        frame2 = Frame(parent)
        frame1.pack(side='bottom', fill='x')
        frame2.pack(side='top', fill='none')
        Button(frame1, text='Reset\nParser', background='#90c0d0', foreground='black', command=self.reset).pack(side='right')
        Button(frame1, text='Top Down\nStrategy', background='#90c0d0', foreground='black', command=self.top_down_strategy).pack(side='left')
        Button(frame1, text='Bottom Up\nStrategy', background='#90c0d0', foreground='black', command=self.bottom_up_strategy).pack(side='left')
        Button(frame1, text='Bottom Up\nLeft-Corner Strategy', background='#90c0d0', foreground='black', command=self.bottom_up_leftcorner_strategy).pack(side='left')
        Button(frame2, text='Top Down Init\nRule', background='#90f090', foreground='black', command=self.top_down_init).pack(side='left')
        Button(frame2, text='Top Down Predict\nRule', background='#90f090', foreground='black', command=self.top_down_predict).pack(side='left')
        Frame(frame2, width=20).pack(side='left')
        Button(frame2, text='Bottom Up Predict\nRule', background='#90f090', foreground='black', command=self.bottom_up).pack(side='left')
        Frame(frame2, width=20).pack(side='left')
        Button(frame2, text='Bottom Up Left-Corner\nPredict Rule', background='#90f090', foreground='black', command=self.bottom_up_leftcorner).pack(side='left')
        Frame(frame2, width=20).pack(side='left')
        Button(frame2, text='Fundamental\nRule', background='#90f090', foreground='black', command=self.fundamental).pack(side='left')

    def _init_bindings(self):
        if False:
            print('Hello World!')
        self._root.bind('<Up>', self._cv.scroll_up)
        self._root.bind('<Down>', self._cv.scroll_down)
        self._root.bind('<Prior>', self._cv.page_up)
        self._root.bind('<Next>', self._cv.page_down)
        self._root.bind('<Control-q>', self.destroy)
        self._root.bind('<Control-x>', self.destroy)
        self._root.bind('<F1>', self.help)
        self._root.bind('<Control-s>', self.save_chart)
        self._root.bind('<Control-o>', self.load_chart)
        self._root.bind('<Control-r>', self.reset)
        self._root.bind('t', self.top_down_strategy)
        self._root.bind('b', self.bottom_up_strategy)
        self._root.bind('c', self.bottom_up_leftcorner_strategy)
        self._root.bind('<space>', self._stop_animation)
        self._root.bind('<Control-g>', self.edit_grammar)
        self._root.bind('<Control-t>', self.edit_sentence)
        self._root.bind('-', lambda e, a=self._animate: a.set(1))
        self._root.bind('=', lambda e, a=self._animate: a.set(2))
        self._root.bind('+', lambda e, a=self._animate: a.set(3))
        self._root.bind('s', lambda e, s=self._step: s.set(not s.get()))

    def _init_menubar(self):
        if False:
            while True:
                i = 10
        menubar = Menu(self._root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label='Save Chart', underline=0, command=self.save_chart, accelerator='Ctrl-s')
        filemenu.add_command(label='Load Chart', underline=0, command=self.load_chart, accelerator='Ctrl-o')
        filemenu.add_command(label='Reset Chart', underline=0, command=self.reset, accelerator='Ctrl-r')
        filemenu.add_separator()
        filemenu.add_command(label='Save Grammar', command=self.save_grammar)
        filemenu.add_command(label='Load Grammar', command=self.load_grammar)
        filemenu.add_separator()
        filemenu.add_command(label='Exit', underline=1, command=self.destroy, accelerator='Ctrl-x')
        menubar.add_cascade(label='File', underline=0, menu=filemenu)
        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label='Edit Grammar', underline=5, command=self.edit_grammar, accelerator='Ctrl-g')
        editmenu.add_command(label='Edit Text', underline=5, command=self.edit_sentence, accelerator='Ctrl-t')
        menubar.add_cascade(label='Edit', underline=0, menu=editmenu)
        viewmenu = Menu(menubar, tearoff=0)
        viewmenu.add_command(label='Chart Matrix', underline=6, command=self.view_matrix)
        viewmenu.add_command(label='Results', underline=0, command=self.view_results)
        menubar.add_cascade(label='View', underline=0, menu=viewmenu)
        rulemenu = Menu(menubar, tearoff=0)
        rulemenu.add_command(label='Top Down Strategy', underline=0, command=self.top_down_strategy, accelerator='t')
        rulemenu.add_command(label='Bottom Up Strategy', underline=0, command=self.bottom_up_strategy, accelerator='b')
        rulemenu.add_command(label='Bottom Up Left-Corner Strategy', underline=0, command=self.bottom_up_leftcorner_strategy, accelerator='c')
        rulemenu.add_separator()
        rulemenu.add_command(label='Bottom Up Rule', command=self.bottom_up)
        rulemenu.add_command(label='Bottom Up Left-Corner Rule', command=self.bottom_up_leftcorner)
        rulemenu.add_command(label='Top Down Init Rule', command=self.top_down_init)
        rulemenu.add_command(label='Top Down Predict Rule', command=self.top_down_predict)
        rulemenu.add_command(label='Fundamental Rule', command=self.fundamental)
        menubar.add_cascade(label='Apply', underline=0, menu=rulemenu)
        animatemenu = Menu(menubar, tearoff=0)
        animatemenu.add_checkbutton(label='Step', underline=0, variable=self._step, accelerator='s')
        animatemenu.add_separator()
        animatemenu.add_radiobutton(label='No Animation', underline=0, variable=self._animate, value=0)
        animatemenu.add_radiobutton(label='Slow Animation', underline=0, variable=self._animate, value=1, accelerator='-')
        animatemenu.add_radiobutton(label='Normal Animation', underline=0, variable=self._animate, value=2, accelerator='=')
        animatemenu.add_radiobutton(label='Fast Animation', underline=0, variable=self._animate, value=3, accelerator='+')
        menubar.add_cascade(label='Animate', underline=1, menu=animatemenu)
        zoommenu = Menu(menubar, tearoff=0)
        zoommenu.add_radiobutton(label='Tiny', variable=self._size, underline=0, value=10, command=self.resize)
        zoommenu.add_radiobutton(label='Small', variable=self._size, underline=0, value=12, command=self.resize)
        zoommenu.add_radiobutton(label='Medium', variable=self._size, underline=0, value=14, command=self.resize)
        zoommenu.add_radiobutton(label='Large', variable=self._size, underline=0, value=18, command=self.resize)
        zoommenu.add_radiobutton(label='Huge', variable=self._size, underline=0, value=24, command=self.resize)
        menubar.add_cascade(label='Zoom', underline=0, menu=zoommenu)
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label='About', underline=0, command=self.about)
        helpmenu.add_command(label='Instructions', underline=0, command=self.help, accelerator='F1')
        menubar.add_cascade(label='Help', underline=0, menu=helpmenu)
        self._root.config(menu=menubar)

    def _click_cv_edge(self, edge):
        if False:
            return 10
        if edge != self._selection:
            self._select_edge(edge)
        else:
            self._cv.cycle_tree()

    def _select_matrix_edge(self, edge):
        if False:
            return 10
        self._select_edge(edge)
        self._cv.view_edge(edge)

    def _select_edge(self, edge):
        if False:
            while True:
                i = 10
        self._selection = edge
        self._cv.markonly_edge(edge, '#f00')
        self._cv.draw_tree(edge)
        if self._matrix:
            self._matrix.markonly_edge(edge)
        if self._matrix:
            self._matrix.view_edge(edge)

    def _deselect_edge(self):
        if False:
            return 10
        self._selection = None
        self._cv.unmark_edge()
        self._cv.erase_tree()
        if self._matrix:
            self._matrix.unmark_edge()

    def _show_new_edge(self, edge):
        if False:
            print('Hello World!')
        self._display_rule(self._cp.current_chartrule())
        self._cv.update()
        self._cv.draw_tree(edge)
        self._cv.markonly_edge(edge, '#0df')
        self._cv.view_edge(edge)
        if self._matrix:
            self._matrix.update()
        if self._matrix:
            self._matrix.markonly_edge(edge)
        if self._matrix:
            self._matrix.view_edge(edge)
        if self._results:
            self._results.update(edge)

    def help(self, *e):
        if False:
            print('Hello World!')
        self._animating = 0
        try:
            ShowText(self._root, 'Help: Chart Parser Application', (__doc__ or '').strip(), width=75, font='fixed')
        except:
            ShowText(self._root, 'Help: Chart Parser Application', (__doc__ or '').strip(), width=75)

    def about(self, *e):
        if False:
            print('Hello World!')
        ABOUT = 'NLTK Chart Parser Application\n' + 'Written by Edward Loper'
        showinfo('About: Chart Parser Application', ABOUT)
    CHART_FILE_TYPES = [('Pickle file', '.pickle'), ('All files', '*')]
    GRAMMAR_FILE_TYPES = [('Plaintext grammar file', '.cfg'), ('Pickle file', '.pickle'), ('All files', '*')]

    def load_chart(self, *args):
        if False:
            i = 10
            return i + 15
        'Load a chart from a pickle file'
        filename = askopenfilename(filetypes=self.CHART_FILE_TYPES, defaultextension='.pickle')
        if not filename:
            return
        try:
            with open(filename, 'rb') as infile:
                chart = pickle.load(infile)
            self._chart = chart
            self._cv.update(chart)
            if self._matrix:
                self._matrix.set_chart(chart)
            if self._matrix:
                self._matrix.deselect_cell()
            if self._results:
                self._results.set_chart(chart)
            self._cp.set_chart(chart)
        except Exception as e:
            raise
            showerror('Error Loading Chart', 'Unable to open file: %r' % filename)

    def save_chart(self, *args):
        if False:
            return 10
        'Save a chart to a pickle file'
        filename = asksaveasfilename(filetypes=self.CHART_FILE_TYPES, defaultextension='.pickle')
        if not filename:
            return
        try:
            with open(filename, 'wb') as outfile:
                pickle.dump(self._chart, outfile)
        except Exception as e:
            raise
            showerror('Error Saving Chart', 'Unable to open file: %r' % filename)

    def load_grammar(self, *args):
        if False:
            return 10
        'Load a grammar from a pickle file'
        filename = askopenfilename(filetypes=self.GRAMMAR_FILE_TYPES, defaultextension='.cfg')
        if not filename:
            return
        try:
            if filename.endswith('.pickle'):
                with open(filename, 'rb') as infile:
                    grammar = pickle.load(infile)
            else:
                with open(filename) as infile:
                    grammar = CFG.fromstring(infile.read())
            self.set_grammar(grammar)
        except Exception as e:
            showerror('Error Loading Grammar', 'Unable to open file: %r' % filename)

    def save_grammar(self, *args):
        if False:
            print('Hello World!')
        filename = asksaveasfilename(filetypes=self.GRAMMAR_FILE_TYPES, defaultextension='.cfg')
        if not filename:
            return
        try:
            if filename.endswith('.pickle'):
                with open(filename, 'wb') as outfile:
                    pickle.dump((self._chart, self._tokens), outfile)
            else:
                with open(filename, 'w') as outfile:
                    prods = self._grammar.productions()
                    start = [p for p in prods if p.lhs() == self._grammar.start()]
                    rest = [p for p in prods if p.lhs() != self._grammar.start()]
                    for prod in start:
                        outfile.write('%s\n' % prod)
                    for prod in rest:
                        outfile.write('%s\n' % prod)
        except Exception as e:
            showerror('Error Saving Grammar', 'Unable to open file: %r' % filename)

    def reset(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self._animating = 0
        self._reset_parser()
        self._cv.update(self._chart)
        if self._matrix:
            self._matrix.set_chart(self._chart)
        if self._matrix:
            self._matrix.deselect_cell()
        if self._results:
            self._results.set_chart(self._chart)

    def edit_grammar(self, *e):
        if False:
            while True:
                i = 10
        CFGEditor(self._root, self._grammar, self.set_grammar)

    def set_grammar(self, grammar):
        if False:
            for i in range(10):
                print('nop')
        self._grammar = grammar
        self._cp.set_grammar(grammar)
        if self._results:
            self._results.set_grammar(grammar)

    def edit_sentence(self, *e):
        if False:
            for i in range(10):
                print('nop')
        sentence = ' '.join(self._tokens)
        title = 'Edit Text'
        instr = 'Enter a new sentence to parse.'
        EntryDialog(self._root, sentence, instr, self.set_sentence, title)

    def set_sentence(self, sentence):
        if False:
            i = 10
            return i + 15
        self._tokens = list(sentence.split())
        self.reset()

    def view_matrix(self, *e):
        if False:
            i = 10
            return i + 15
        if self._matrix is not None:
            self._matrix.destroy()
        self._matrix = ChartMatrixView(self._root, self._chart)
        self._matrix.add_callback('select', self._select_matrix_edge)

    def view_results(self, *e):
        if False:
            while True:
                i = 10
        if self._results is not None:
            self._results.destroy()
        self._results = ChartResultsView(self._root, self._chart, self._grammar)

    def resize(self):
        if False:
            return 10
        self._animating = 0
        self.set_font_size(self._size.get())

    def set_font_size(self, size):
        if False:
            return 10
        self._cv.set_font_size(size)
        self._font.configure(size=-abs(size))
        self._boldfont.configure(size=-abs(size))
        self._sysfont.configure(size=-abs(size))

    def get_font_size(self):
        if False:
            while True:
                i = 10
        return abs(self._size.get())

    def apply_strategy(self, strategy, edge_strategy=None):
        if False:
            while True:
                i = 10
        if self._animating:
            self._animating = 0
            return
        self._display_rule(None)
        if self._step.get():
            selection = self._selection
            if selection is not None and edge_strategy is not None:
                self._cp.set_strategy([edge_strategy(selection)])
                newedge = self._apply_strategy()
                if newedge is None:
                    self._cv.unmark_edge()
                    self._selection = None
            else:
                self._cp.set_strategy(strategy)
                self._apply_strategy()
        else:
            self._cp.set_strategy(strategy)
            if self._animate.get():
                self._animating = 1
                self._animate_strategy()
            else:
                for edge in self._cpstep:
                    if edge is None:
                        break
                self._cv.update()
                if self._matrix:
                    self._matrix.update()
                if self._results:
                    self._results.update()

    def _stop_animation(self, *e):
        if False:
            return 10
        self._animating = 0

    def _animate_strategy(self, speed=1):
        if False:
            for i in range(10):
                print('nop')
        if self._animating == 0:
            return
        if self._apply_strategy() is not None:
            if self._animate.get() == 0 or self._step.get() == 1:
                return
            if self._animate.get() == 1:
                self._root.after(3000, self._animate_strategy)
            elif self._animate.get() == 2:
                self._root.after(1000, self._animate_strategy)
            else:
                self._root.after(20, self._animate_strategy)

    def _apply_strategy(self):
        if False:
            while True:
                i = 10
        new_edge = next(self._cpstep)
        if new_edge is not None:
            self._show_new_edge(new_edge)
        return new_edge

    def _display_rule(self, rule):
        if False:
            print('Hello World!')
        if rule is None:
            self._rulelabel2['text'] = ''
        else:
            name = str(rule)
            self._rulelabel2['text'] = name
            size = self._cv.get_font_size()
    _TD_INIT = [TopDownInitRule()]
    _TD_PREDICT = [TopDownPredictRule()]
    _BU_RULE = [BottomUpPredictRule()]
    _BU_LC_RULE = [BottomUpPredictCombineRule()]
    _FUNDAMENTAL = [SingleEdgeFundamentalRule()]
    _TD_STRATEGY = _TD_INIT + _TD_PREDICT + _FUNDAMENTAL
    _BU_STRATEGY = _BU_RULE + _FUNDAMENTAL
    _BU_LC_STRATEGY = _BU_LC_RULE + _FUNDAMENTAL

    def top_down_init(self, *e):
        if False:
            print('Hello World!')
        self.apply_strategy(self._TD_INIT, None)

    def top_down_predict(self, *e):
        if False:
            i = 10
            return i + 15
        self.apply_strategy(self._TD_PREDICT, TopDownPredictEdgeRule)

    def bottom_up(self, *e):
        if False:
            for i in range(10):
                print('nop')
        self.apply_strategy(self._BU_RULE, BottomUpEdgeRule)

    def bottom_up_leftcorner(self, *e):
        if False:
            for i in range(10):
                print('nop')
        self.apply_strategy(self._BU_LC_RULE, BottomUpLeftCornerEdgeRule)

    def fundamental(self, *e):
        if False:
            return 10
        self.apply_strategy(self._FUNDAMENTAL, FundamentalEdgeRule)

    def bottom_up_strategy(self, *e):
        if False:
            for i in range(10):
                print('nop')
        self.apply_strategy(self._BU_STRATEGY, BottomUpEdgeRule)

    def bottom_up_leftcorner_strategy(self, *e):
        if False:
            print('Hello World!')
        self.apply_strategy(self._BU_LC_STRATEGY, BottomUpLeftCornerEdgeRule)

    def top_down_strategy(self, *e):
        if False:
            return 10
        self.apply_strategy(self._TD_STRATEGY, TopDownPredictEdgeRule)

def app():
    if False:
        return 10
    grammar = CFG.fromstring("\n    # Grammatical productions.\n        S -> NP VP\n        VP -> VP PP | V NP | V\n        NP -> Det N | NP PP\n        PP -> P NP\n    # Lexical productions.\n        NP -> 'John' | 'I'\n        Det -> 'the' | 'my' | 'a'\n        N -> 'dog' | 'cookie' | 'table' | 'cake' | 'fork'\n        V -> 'ate' | 'saw'\n        P -> 'on' | 'under' | 'with'\n    ")
    sent = 'John ate the cake on the table with a fork'
    sent = 'John ate the cake on the table'
    tokens = list(sent.split())
    print('grammar= (')
    for rule in grammar.productions():
        print(('    ', repr(rule) + ','))
    print(')')
    print('tokens = %r' % tokens)
    print('Calling "ChartParserApp(grammar, tokens)"...')
    ChartParserApp(grammar, tokens).mainloop()
if __name__ == '__main__':
    app()
__all__ = ['app']