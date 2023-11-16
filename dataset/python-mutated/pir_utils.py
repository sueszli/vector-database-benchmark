from functools import wraps
import paddle

class IrGuard:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.in_dygraph_outside = False
        old_flag = paddle.base.framework.get_flags('FLAGS_enable_pir_api')
        paddle.base.framework.set_flags({'FLAGS_enable_pir_api': False})
        paddle.base.framework.global_var._use_pir_api_ = False
        if not paddle.base.framework.get_flags('FLAGS_enable_pir_api')['FLAGS_enable_pir_api']:
            self.old_Program = paddle.static.Program
            self.old_program_guard = paddle.base.program_guard
            self.old_default_main_program = paddle.static.default_main_program
            self.old_default_startup_program = paddle.static.default_startup_program
        else:
            raise RuntimeError('IrGuard only init when paddle.framework.in_pir_mode(): is false,                 please set FLAGS_enable_pir_api = false')
        paddle.base.framework.set_flags(old_flag)
        paddle.base.framework.global_var._use_pir_api_ = old_flag['FLAGS_enable_pir_api']

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.in_dygraph_outside = paddle.base.framework.in_dygraph_mode()
        if self.in_dygraph_outside:
            paddle.enable_static()
        paddle.framework.set_flags({'FLAGS_enable_pir_api': True})
        paddle.base.framework.global_var._use_pir_api_ = True
        self._switch_to_pir()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        paddle.framework.set_flags({'FLAGS_enable_pir_api': False})
        paddle.base.framework.global_var._use_pir_api_ = False
        self._switch_to_old_ir()
        if self.in_dygraph_outside:
            paddle.disable_static()

    def _switch_to_pir(self):
        if False:
            i = 10
            return i + 15
        if paddle.base.framework.get_flags('FLAGS_enable_pir_api')['FLAGS_enable_pir_api']:
            paddle.framework.set_flags({'FLAGS_enable_pir_in_executor': True})
            paddle.pir.register_paddle_dialect()
            paddle.base.Program = paddle.pir.Program
            paddle.base.program_guard = paddle.pir.core.program_guard
            paddle.static.Program = paddle.pir.Program
            paddle.static.program_guard = paddle.pir.core.program_guard
            paddle.static.default_main_program = paddle.pir.core.default_main_program
            paddle.static.default_startup_program = paddle.pir.core.default_startup_program

    def _switch_to_old_ir(self):
        if False:
            i = 10
            return i + 15
        if not paddle.base.framework.get_flags('FLAGS_enable_pir_api')['FLAGS_enable_pir_api']:
            paddle.framework.set_flags({'FLAGS_enable_pir_in_executor': False})
            paddle.base.Program = self.old_Program
            paddle.base.program_guard = self.old_program_guard
            paddle.static.Program = self.old_Program
            paddle.static.program_guard = self.old_program_guard
            paddle.static.default_main_program = self.old_default_main_program
            paddle.static.default_startup_program = self.old_default_startup_program
        else:
            raise RuntimeError('IrGuard._switch_to_old_ir only work when paddle.framework.in_pir_mode() is false,                 please set FLAGS_enable_pir_api = false')

def test_with_pir_api(func):
    if False:
        print('Hello World!')

    @wraps(func)
    def impl(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        func(*args, **kwargs)
        with IrGuard():
            func(*args, **kwargs)
    return impl