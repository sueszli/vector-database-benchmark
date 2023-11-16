import click
import collections
import logging
import numpy as np
import os
from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.contrib.tensorboard.tensorboard_exporter as tb_exporter
try:
    from tensorboard.compat.proto.summary_pb2 import Summary, HistogramProto
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.summary.writer.event_file_writer import EventFileWriter as FileWriter
except ImportError:
    from tensorflow.core.framework.summary_pb2 import Summary, HistogramProto
    from tensorflow.core.util.event_pb2 import Event
    try:
        from tensorflow.summary import FileWriter
    except ImportError:
        from tensorflow.train import SummaryWriter as FileWriter

class Config:
    HEIGHT = 600
    ASPECT_RATIO = 1.6
CODE_TEMPLATE = '\n<script>\n  function load() {{\n    document.getElementById("{id}").pbtxt = {data};\n  }}\n</script>\n<link rel="import"\n  href="https://tensorboard.appspot.com/tf-graph-basic.build.html"\n  onload=load()\n>\n<div style="height:{height}px">\n  <tf-graph-basic id="{id}"></tf-graph-basic>\n</div>\n'
IFRAME_TEMPLATE = '\n<iframe\n  seamless\n  style="width:{width}px;height:{height}px;border:0"\n  srcdoc="{code}">\n</iframe>\n'

def _show_graph(graph_def):
    if False:
        for i in range(10):
            print('nop')
    import IPython.display
    code = CODE_TEMPLATE.format(data=repr(str(graph_def)), id='graph' + str(np.random.rand()), height=Config.HEIGHT)
    iframe = IFRAME_TEMPLATE.format(code=code.replace('"', '&quot;'), width=Config.HEIGHT * Config.ASPECT_RATIO, height=Config.HEIGHT + 20)
    IPython.display.display(IPython.display.HTML(iframe))

def visualize_cnn(cnn, **kwargs):
    if False:
        i = 10
        return i + 15
    g = tb_exporter.cnn_to_graph_def(cnn, **kwargs)
    _show_graph(g)

def visualize_net(nets, **kwargs):
    if False:
        i = 10
        return i + 15
    g = tb_exporter.nets_to_graph_def(nets, **kwargs)
    _show_graph(g)

def visualize_ops(ops, **kwargs):
    if False:
        return 10
    g = tb_exporter.ops_to_graph_def(ops, **kwargs)
    _show_graph(g)

@click.group()
def cli():
    if False:
        return 10
    pass

def write_events(tf_dir, events):
    if False:
        return 10
    writer = FileWriter(tf_dir, len(events))
    for event in events:
        writer.add_event(event)
    writer.flush()
    writer.close()

def graph_def_to_event(step, graph_def):
    if False:
        return 10
    return Event(wall_time=step, step=step, graph_def=graph_def.SerializeToString())

@cli.command('tensorboard-graphs')
@click.option('--c2-netdef', type=click.Path(exists=True, dir_okay=False), multiple=True)
@click.option('--tf-dir', type=click.Path(exists=True))
def tensorboard_graphs(c2_netdef, tf_dir):
    if False:
        return 10
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def parse_net_def(path):
        if False:
            while True:
                i = 10
        import google.protobuf.text_format
        net_def = caffe2_pb2.NetDef()
        with open(path) as f:
            google.protobuf.text_format.Merge(f.read(), net_def)
        return core.Net(net_def)
    graph_defs = [tb_exporter.nets_to_graph_def([parse_net_def(path)]) for path in c2_netdef]
    events = [graph_def_to_event(i, graph_def) for (i, graph_def) in enumerate(graph_defs, start=1)]
    write_events(tf_dir, events)
    log.info('Wrote %s graphs to logdir %s', len(events), tf_dir)

@cli.command('tensorboard-events')
@click.option('--c2-dir', type=click.Path(exists=True, file_okay=False), help='Root directory of the Caffe2 run')
@click.option('--tf-dir', type=click.Path(writable=True), help='Output path to the logdir used by TensorBoard')
def tensorboard_events(c2_dir, tf_dir):
    if False:
        print('Hello World!')
    np.random.seed(1701)
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    S = collections.namedtuple('S', ['min', 'max', 'mean', 'std'])

    def parse_summary(filename):
        if False:
            for i in range(10):
                print('nop')
        try:
            with open(filename) as f:
                rows = [(float(el) for el in line.split()) for line in f]
                return [S(*r) for r in rows]
        except Exception as e:
            log.exception(e)
            return None

    def get_named_summaries(root):
        if False:
            for i in range(10):
                print('nop')
        summaries = [(fname, parse_summary(os.path.join(dirname, fname))) for (dirname, _, fnames) in os.walk(root) for fname in fnames]
        return [(n, s) for (n, s) in summaries if s]

    def inferred_histo(summary, samples=1000):
        if False:
            while True:
                i = 10
        np.random.seed(hash(summary.std + summary.mean + summary.min + summary.max) % np.iinfo(np.int32).max)
        samples = np.random.randn(samples) * summary.std + summary.mean
        samples = np.clip(samples, a_min=summary.min, a_max=summary.max)
        (hist, edges) = np.histogram(samples)
        upper_edges = edges[1:]
        r = HistogramProto(min=summary.min, max=summary.max, num=len(samples), sum=samples.sum(), sum_squares=(samples * samples).sum())
        r.bucket_limit.extend(upper_edges)
        r.bucket.extend(hist)
        return r

    def named_summaries_to_events(named_summaries):
        if False:
            i = 10
            return i + 15
        names = [n for (n, _) in named_summaries]
        summaries = [s for (_, s) in named_summaries]
        summaries = list(zip(*summaries))

        def event(step, values):
            if False:
                for i in range(10):
                    print('nop')
            s = Summary()
            scalar = [Summary.Value(tag='{}/{}'.format(name, field), simple_value=v) for (name, value) in zip(names, values) for (field, v) in value._asdict().items()]
            hist = [Summary.Value(tag='{}/inferred_normal_hist'.format(name), histo=inferred_histo(value)) for (name, value) in zip(names, values)]
            s.value.extend(scalar + hist)
            return Event(wall_time=int(step), step=step, summary=s)
        return [event(step, values) for (step, values) in enumerate(summaries, start=1)]
    named_summaries = get_named_summaries(c2_dir)
    events = named_summaries_to_events(named_summaries)
    write_events(tf_dir, events)
    log.info('Wrote %s events to logdir %s', len(events), tf_dir)
if __name__ == '__main__':
    cli()