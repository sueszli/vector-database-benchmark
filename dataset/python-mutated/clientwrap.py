import sys
import asyncio
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from traitlets import Bool, Instance

class PapermillNotebookClient(NotebookClient):
    """
    Module containing a  that executes the code cells
    and updates outputs
    """
    log_output = Bool(False).tag(config=True)
    stdout_file = Instance(object, default_value=None).tag(config=True)
    stderr_file = Instance(object, default_value=None).tag(config=True)

    def __init__(self, nb_man, km=None, raise_on_iopub_timeout=True, **kw):
        if False:
            i = 10
            return i + 15
        'Initializes the execution manager.\n\n        Parameters\n        ----------\n        nb_man : NotebookExecutionManager\n            Notebook execution manager wrapper being executed.\n        km : KernerlManager (optional)\n            Optional kernel manager. If none is provided, a kernel manager will\n            be created.\n        '
        super().__init__(nb_man.nb, km=km, raise_on_iopub_timeout=raise_on_iopub_timeout, **kw)
        self.nb_man = nb_man

    def execute(self, **kwargs):
        if False:
            return 10
        '\n        Wraps the parent class process call slightly\n        '
        self.reset_execution_trackers()
        if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        with self.setup_kernel(**kwargs):
            self.log.info('Executing notebook with kernel: %s' % self.kernel_name)
            self.papermill_execute_cells()
            info_msg = self.wait_for_reply(self.kc.kernel_info())
            self.nb.metadata['language_info'] = info_msg['content']['language_info']
            self.set_widgets_metadata()
        return self.nb

    def papermill_execute_cells(self):
        if False:
            print('Hello World!')
        "\n        This function replaces cell execution with it's own wrapper.\n\n        We are doing this for the following reasons:\n\n        1. Notebooks will stop executing when they encounter a failure but not\n           raise a `CellException`. This allows us to save the notebook with the\n           traceback even though a `CellExecutionError` was encountered.\n\n        2. We want to write the notebook as cells are executed. We inject our\n           logic for that here.\n\n        3. We want to include timing and execution status information with the\n           metadata of each cell.\n        "
        for (index, cell) in enumerate(self.nb.cells):
            try:
                self.nb_man.cell_start(cell, index)
                self.execute_cell(cell, index)
            except CellExecutionError as ex:
                self.nb_man.cell_exception(self.nb.cells[index], cell_index=index, exception=ex)
                break
            finally:
                self.nb_man.cell_complete(self.nb.cells[index], cell_index=index)

    def log_output_message(self, output):
        if False:
            print('Hello World!')
        '\n        Process a given output. May log it in the configured logger and/or write it into\n        the configured stdout/stderr files.\n\n        :param output: nbformat.notebooknode.NotebookNode\n        :return:\n        '
        if output.output_type == 'stream':
            content = ''.join(output.text)
            if output.name == 'stdout':
                if self.log_output:
                    self.log.info(content)
                if self.stdout_file:
                    self.stdout_file.write(content)
                    self.stdout_file.flush()
            elif output.name == 'stderr':
                if self.log_output:
                    self.log.warning(content)
                if self.stderr_file:
                    self.stderr_file.write(content)
                    self.stderr_file.flush()
        elif self.log_output and ('data' in output and 'text/plain' in output.data):
            self.log.info(''.join(output.data['text/plain']))

    def process_message(self, *arg, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        output = super().process_message(*arg, **kwargs)
        self.nb_man.autosave_cell()
        if output and (self.log_output or self.stderr_file or self.stdout_file):
            self.log_output_message(output)
        return output