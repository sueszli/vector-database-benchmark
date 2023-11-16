try:
    from IPython.core.magic import Magics, cell_magic, magics_class, needs_local_scope
    from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

    @magics_class
    class VizTracerMagics(Magics):

        @magic_arguments()
        @argument('--port', '-p', default=9001, type=int, help='specify the port vizviewer will use')
        @argument('--output_file', default='./viztracer_report.json', help='output file path. End with .json or .html or .gz')
        @argument('--max_stack_depth', type=int, default=-1, help='maximum stack depth you want to trace.')
        @argument('--ignore_c_function', action='store_true', default=False, help='ignore all c functions including most builtin functions and libraries')
        @argument('--ignore_frozen', action='store_true', default=False, help='ignore all functions that are frozen(like import)')
        @argument('--log_func_args', action='store_true', default=False, help='log all function arguments, this will introduce large overhead')
        @argument('--log_print', action='store_true', default=False, help='replace all print() function to adding an event to the result')
        @argument('--log_sparse', action='store_true', default=False, help='log only selected functions with @log_sparse')
        @needs_local_scope
        @cell_magic
        def viztracer(self, line, cell, local_ns) -> None:
            if False:
                print('Hello World!')
            from IPython.display import display
            from ipywidgets import Button
            from .viewer import ServerThread
            from .viztracer import VizTracer
            options = parse_argstring(self.viztracer, line)
            assert self.shell is not None
            code = self.shell.transform_cell(cell)
            file_path = options.output_file
            tracer_kwargs = {'output_file': file_path, 'verbose': 0, 'max_stack_depth': options.max_stack_depth, 'ignore_c_function': options.ignore_c_function, 'ignore_frozen': options.ignore_frozen, 'log_func_args': options.log_func_args, 'log_print': options.log_print, 'log_sparse': options.log_sparse}
            with VizTracer(**tracer_kwargs):
                exec(code, local_ns, local_ns)

            def view():
                if False:
                    for i in range(10):
                        print('nop')
                server = ServerThread(file_path, port=options.port, once=True)
                server.start()
                server.ready.wait()
                import webbrowser
                webbrowser.open_new_tab(f'http://127.0.0.1:{server.port}')
            button = Button(description='VizTracer Report')
            button.on_click(lambda b: view())
            display(button)
except ImportError:
    pass

def load_ipython_extension(ipython) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Use `%load_ext viztracer`\n    '
    ipython.register_magics(VizTracerMagics)