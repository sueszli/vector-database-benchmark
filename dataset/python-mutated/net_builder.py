from caffe2.python import core, context
from caffe2.python.task import Task, TaskGroup
from caffe2.python.control_ops_util import add_if_op, add_while_op

class NetBuilder(context.Managed):
    """
    Scope-driven mechanism for building nets, loops and conditional blocks.
    Args:
      name: NetBuilder's name
      initial_scope: list of blobs that are available for reading/writing
    Example:
        from caffe2.python.net_builder import NetBuilder, ops
        with NetBuilder() as nb:
            c = ops.Const(5)
            d = ops.Const(0)
            with ops.loop():
                ops.stop_if(ops.LE([c, ops.Const(0)]))
                ops.Add([c, ops.Const(-1)], [c])
                with ops.If(ops.GE([c, ops.Const(3)])):
                    ops.Add([d, ops.Const(10)], [d])
            ops.Print(c, [])
            ops.Print(d, [])
        step = core.to_execution_step(nb)
    """

    def __init__(self, name=None, initial_scope=None, _stop_blob_required=False, _stop_blob=None, _fullname=None, _use_control_ops=False):
        if False:
            return 10
        parent = NetBuilder.current(required=False)
        assert not _fullname or not name, 'Cannot set both _fullname and name'
        assert not _use_control_ops or (not _stop_blob_required and (not _stop_blob)), 'Stop blobs are not used with control operators'
        self.name = _fullname or '/'.join((n for n in (parent.name if parent else None, name) if n))
        self._frozen = False
        self._current_net = None
        self._children = []
        if parent:
            parent._update_lexical_scope()
        self._init_lexical_scope = set(parent._lexical_scope) if parent else set()
        if initial_scope:
            self._init_lexical_scope |= set([str(b) for b in initial_scope])
        self._lexical_scope = set(self._init_lexical_scope)
        self._stop_blob = _stop_blob
        self._stop_blob_required = _stop_blob_required
        self._use_control_ops = _use_control_ops

    def stop_blob(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the BlobReference to the stop_blob of this NetBuilder.\n        If one is not yet available, creates one.\n        This function assumes that the stop_blob() will be used immediatelly\n        in the current net, so it doesn't initialize it if the current net is\n        the first of the builder.\n        "
        assert not self._use_control_ops, 'Stop blobs are not used with control operators'
        if self._stop_blob is None:
            net = self.current_net()
            self._stop_blob = core.BlobReference(net.NextName('stop_blob'), net=net)
            net.Const(False, blob_out=self._stop_blob)
            if self._current_net != self._children[0]:
                self._children.insert(0, core.Net('stop_blob_init'))
                self._children[0].Const(False, blob_out=self._stop_blob)
        return self._stop_blob

    def stop_if(self, blob):
        if False:
            while True:
                i = 10
        assert not self._use_control_ops, 'Stop blobs are not used with control operators'
        stop_blob = self.stop_blob()
        ops.Or([stop_blob, blob], [stop_blob])
        self._current_net = None

    def _assert_mutable(self):
        if False:
            return 10
        assert not self._frozen, 'This NetBuilder (%s) has been built already.' % self.name

    def _update_lexical_scope(self):
        if False:
            return 10
        '\n        Updates lexical scope based on the current list of children.\n        Lexical scope contains names of blobs that are currently available\n        and were introduced in the net builder\n        '
        self._lexical_scope = set(self._init_lexical_scope)
        for child in self._children:
            if isinstance(child, core.Net):
                self._lexical_scope |= child.UsedBlobNames()
            elif isinstance(child, NetBuilder) and child._use_control_ops:
                self._lexical_scope |= child._lexical_scope

    def _reset_children(self):
        if False:
            return 10
        self._current_net = None
        self._children = []
        self._lexical_scope = set(self._init_lexical_scope)

    def add(self, child):
        if False:
            print('Hello World!')
        self._assert_mutable()
        if self._use_control_ops:
            assert isinstance(child, core.Net) or (isinstance(child, NetBuilder) and child._use_control_ops), 'Expected Net or NetBuilder with control ops'
        self._current_net = None
        self._children.append(child)
        if isinstance(child, core.Net):
            self._current_net = child
        self._update_lexical_scope()
        return child

    def current_net(self, name=None):
        if False:
            return 10
        self._assert_mutable()
        if self._current_net is None or name is not None:
            self.add(core.Net(name))
        return self._current_net

    def freeze(self):
        if False:
            return 10
        for child in self._children:
            if hasattr(child, 'freeze'):
                child.freeze()
        self._current_net = None
        self._frozen = True

    def get(self):
        if False:
            while True:
                i = 10
        self.freeze()
        return self._children

    def __exit__(self, etype, *args):
        if False:
            while True:
                i = 10
        super().__exit__(etype, *args)
        if self._use_control_ops and len(self._children) > 0:
            _children = self._children
            self._reset_children()
            merged_net = NetBuilder.merge_nets(_children, self._lexical_scope)
            assert merged_net, 'Expected a non-empty merge of children'
            self._children = [merged_net]
        self.freeze()
        if etype is not None:
            return
        assert not self._stop_blob_required or self._stop_blob is not None, 'This NetBuilder (%s) requires a stop condition ' % self.name + 'to be set with `stop` or `stop_if`'

    @staticmethod
    def merge_nets(nets_or_builders, outer_blob_names):
        if False:
            i = 10
            return i + 15
        net = None
        for n in nets_or_builders:
            cur = None
            if isinstance(n, NetBuilder):
                assert n._use_control_ops, 'Merging of NetBuilder supported only for control ops'
                nets = n.get()
                assert len(nets) == 1 and isinstance(nets[0], core.Net), 'Invalid control op net builder'
                cur = nets[0]
            else:
                assert isinstance(n, core.Net)
                cur = n
            if net:
                net.AppendNet(cur)
            else:
                net = cur
        if net:
            external_outputs = [o for o in net.Proto().external_output if o in outer_blob_names]
            net.Proto().external_output[:] = external_outputs
        return net

    def __str__(self):
        if False:
            return 10
        return self.name or 'Un-named NetBuilder'

class Operations:
    """
    Operations to be used in the context of a NetBuilder.
    """

    def net(self, net=None, name=None):
        if False:
            i = 10
            return i + 15
        '\n        Retrieves the current net, or add a new net to the builder.\n        Args:\n            net:   If provided, add the given net to the active builder.\n                   Else, returns the current Net or creates a new one as needed.\n            name:  if provided, creates a new Net with given name and makes\n                   it the new current net of the active builder. Cannot\n                   be provided if net is provided.\n        '
        assert name is None or net is None, 'Cannot provide both `net` and `name`.'
        if net is not None:
            NetBuilder.current().add(net)
            return net
        return NetBuilder.current().current_net(name=name)

    def __getattr__(self, op_type):
        if False:
            print('Hello World!')
        '\n        Adds an operator call to the currently active Net.\n        '
        if op_type.startswith('__'):
            raise AttributeError()
        if NetBuilder.current(required=False) is None:
            raise AttributeError('No active NetBuilder.')
        return getattr(self.net(), op_type)

    def task_group(self):
        if False:
            while True:
                i = 10
        '\n        Creates a local task group which will execute as the next step of\n        the current NetBuilder.\n        '
        from caffe2.python import task
        group = NetBuilder.current()
        with task.Cluster():
            with task.Node('local'):
                tg = task.TaskGroup()
                group.add(tg)
                return tg

    def stop(self):
        if False:
            return 10
        "\n        Stop execution of the current execution step.\n            Example:\n                ops.Print(a, 0)\n                ops.stop()\n                ops.Print(b, 0)\n            In the example, 'b' will never be printed.\n        "
        return self.stop_if(ops.Const(True))

    def stop_if(self, blob):
        if False:
            return 10
        "\n        Stop execution of the current execution step if the\n        condition `blob` is met.\n            Example:\n                ops.Print(a, 0)\n                ops.stop_if(ops.LE([x, ops.Const(0)]))\n                ops.Print(b, 0)\n            In the example, 'b' will only be printed if the value of scalar\n            tensor 'x' is greater than 0.\n        "
        return NetBuilder.current().stop_if(blob)

    def loop(self, iters=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Creates a NetBuilder that will execute in a loop as the next step of\n        the current NetBuilder. If `iters` is provided, the loop will execute\n        for `iters` iterations and then stop. `iters` can be a constant or a\n        BlobReference. If `iters` is not provided, the loop will execute\n        until `ops.stop` or `ops.stop_if` is called.\n            Examples:\n                a = ops.Const(5)\n                with ops.loop():\n                    ops.stop_if(ops.LE([a, ops.Const(0)]))\n                    ops.Print(a, 0)\n                    ops.Add([a, ops.Const(-1)], [a])\n            Above, 'a' will be printed 5 times, with values 5 to 1.\n\n                with ops.loop(10) as loop:\n                    ops.LogInfo(loop.iter())\n            This will print the numbers from 0 to 9.\n\n                x = ops.Add([ops.Const(10), ops.Const(10)])\n                with ops.loop(x) as loop:\n                    ops.LogInfo(loop.iter())\n            This will print the numbers from 0 to 19.\n        "
        return NetBuilder.current().add(_Loop(iters, name=name))

    def stop_guard(self, has_stopped_blob=None, name=None):
        if False:
            while True:
                i = 10
        "\n        Creates a NetBuilder that will execute once as the next step of the\n        current NetBuilder. After execution, a bool tensor will indicate\n        whether the inner execution was halted with `stop` or `stop_if`.\n            Example:\n                a = ops.Const(True)\n                with ops.stop_guard() as sg1:\n                    ops.stop_if(a)\n                    ops.Print(ops.Const('did not stop'))\n                b = ops.Const(False)\n                with ops.stop_guard() as sg2:\n                    ops.stop_if(b)\n                    ops.Print(ops.Const('did not stop'))\n                ops.Print(sg1.has_stopped(), [])\n                ops.Print(sg2.has_stopped(), [])\n            In the example, 'did not stop' will be printed once,\n            followed by True and False.\n        "
        return NetBuilder.current().add(_StopGuard(has_stopped_blob=has_stopped_blob, name=name))

    def If(self, cond, name=None):
        if False:
            return 10
        "\n        Creates a NetBuilder that will execute once as the next step of the\n        current NetBuilder if the blob `cond` is True.\n            Example:\n                with ops.If(ops.Const(True)):\n                    ops.Print(ops.Const('Will print'))\n                with ops.If(ops.Const(False)):\n                    ops.Print(ops.Const('Wont print'))\n            The example will print 'Will print' once.\n        "
        return NetBuilder.current().add(_RunIf(cond, name=name))

    def IfNet(self, cond, name=None):
        if False:
            return 10
        "\n        Same as If, but uses 'If' operator instead of execution step logic\n        "
        return NetBuilder.current().add(_RunIfNet(cond, name=name))

    def Else(self, name=None):
        if False:
            while True:
                i = 10
        '\n        Else branch of IfNet, has to be specified immediately after IfNet.\n            Example:\n                with ops.IfNet(ops.LT([x, y])):\n                    ...\n                with ops.Else():\n                    ...\n        '
        return _RunElseNet(name=name)

    def WhileNet(self, name=None):
        if False:
            print('Hello World!')
        "\n        NetBuilder for 'While' control operator\n        "
        return NetBuilder.current().add(_RunWhileNet(name=name))

    def Condition(self, name=None):
        if False:
            return 10
        "\n        Loop's condition, executed within WhileNet context\n        "
        assert isinstance(NetBuilder.current(), _RunWhileNet), 'Use of Condition outside of WhileNet'
        return _RunWhileCondition(name=name)

    def task_init(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Defines operations that will be executed once at task startup.\n        Useful when implementing processors, that don't have access to the Task\n        top-level structure.\n\n        This setup will be run only once, even if multiple instances of the task\n        will run in parallel. For instance-local initialization, use\n        `task_instance_init` instead.\n\n            Example:\n                def my_processor(rec):\n                    with ops.task_init():\n                        one = ops.Const(1)\n                        two = ops.Const(1)\n                    return Tuple(\n                        ops.Add(rec[0](), zero), ops.Add(rec[1](), two))\n        "
        setup = _SetupBuilder(_SetupBuilder.INIT)
        self.net().add_attribute(Task.TASK_SETUP, setup)
        return setup

    def task_exit(self):
        if False:
            return 10
        "\n        Define operations to be executed once at task shutdown.\n        Useful when implementing processors, that don't have access to the Task\n        top-level structure.\n\n        This shutdown will be run only once, after all concurrent instances of\n        the task have already finished. For instance-local shutdown,\n        use `task_instance_exit` instead.\n\n            Example:\n                def read_queue(queue):\n                    with ops.task_exit():\n                        queue.close(ops.net())\n                    return queue.read(ops.net())\n        "
        setup = _SetupBuilder(_SetupBuilder.EXIT)
        self.net().add_attribute(Task.TASK_SETUP, setup)
        return setup

    def task_instance_init(self):
        if False:
            i = 10
            return i + 15
        '\n        Defines operations that will be executed once at startup of each\n        instance of a task. This can be seen as "thread_local" initialization.\n        It is guaranteed to run only after all `task_init` logic finishes.\n\n        This setup will be run concurrently for each instance of a task.\n        For global task initialization, use `task_init` instead.\n        '
        setup = _SetupBuilder(_SetupBuilder.INIT)
        self.net().add_attribute(Task.TASK_INSTANCE_SETUP, setup)
        return setup

    def task_instance_exit(self):
        if False:
            while True:
                i = 10
        '\n        Defines operations that will be executed once at shutdown of each\n        instance of a task. This can be seen as "thread_local" finalization.\n\n        This shutdown will be run concurrently for each instance of a task.\n        For global task shutdown, use `task_exit` instead.\n        '
        setup = _SetupBuilder(_SetupBuilder.EXIT)
        self.net().add_attribute(Task.TASK_INSTANCE_SETUP, setup)
        return setup

    def local_init(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Similar to `task_init`, but executes at TaskGroup's startup instead,\n        before any task of the group starts executing. This will run only\n        once on each node, before initialization of any task, so it can be\n        used e.g. to initialize blobs shared across tasks.\n        "
        setup = _SetupBuilder(_SetupBuilder.INIT)
        self.net().add_attribute(TaskGroup.LOCAL_SETUP, setup)
        return setup

    def local_exit(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Similar to `task_exit`, but executes at TaskGroup's exit instead,\n        after all tasks of the group finished execution.\n        This will run only once on each node.\n        "
        setup = _SetupBuilder(_SetupBuilder.EXIT, name)
        self.net().add_attribute(TaskGroup.LOCAL_SETUP, setup)
        return setup

    def task_reporter(self, interval_ms=1000, name=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Define operations to be executed at every time interval from\n        task start-up to finish. These operations are guaranteed to\n        execute at least once after all other operations of the task are\n        finished.\n\n            Example:\n                with ops.task_reporter(interval_ms=10000):\n                    ops.LogInfo('10s elapsed')\n        "
        return _ReporterBuilder(interval_ms, net=self.net(), name=name)

    def local_reporter(self, interval_ms=1000, name=None):
        if False:
            print('Hello World!')
        '\n        Similar to task_report, but operations defined within this block\n        will run repeatedly for as long as any of the tasks in the current\n        TaskGroup have not finished.\n        '
        return _ReporterBuilder(interval_ms, name=name)
ops = Operations()

class _ReporterBuilder(NetBuilder):

    def __init__(self, interval_ms, net=None, name=None):
        if False:
            i = 10
            return i + 15
        NetBuilder.__init__(self, name)
        self._net = net
        self.interval_ms = interval_ms

    def __exit__(self, etype, *args):
        if False:
            while True:
                i = 10
        if etype is None:
            step = core.to_execution_step(self)
            step.RunEveryMillis(self.interval_ms)
            if self._net:
                self._net.add_attribute(Task.REPORT_STEP, step)
            else:
                TaskGroup.current().report_step(step, interval_ms=self.interval_ms)
        NetBuilder.__exit__(self, etype, *args)

class _SetupBuilder(NetBuilder):
    INIT = 'init'
    EXIT = 'exit'

    def __init__(self, type, name=None):
        if False:
            return 10
        NetBuilder.__init__(self, name)
        self.type = type

    def setup(self, net):
        if False:
            print('Hello World!')
        if self.type == _SetupBuilder.INIT:
            return core.to_execution_step(self)

    def exit(self, net):
        if False:
            for i in range(10):
                print('nop')
        if self.type == _SetupBuilder.EXIT:
            return core.to_execution_step(self)

class _RunOnce(NetBuilder):

    def __init__(self, name=None):
        if False:
            i = 10
            return i + 15
        NetBuilder.__init__(self, name)

    def __exit__(self, etype, *args):
        if False:
            while True:
                i = 10
        if etype is None and self._stop_blob is not None:
            ops.stop()
        NetBuilder.__exit__(self, etype, *args)

class _StopGuard(_RunOnce):

    def __init__(self, has_stopped_blob=None, name=None):
        if False:
            i = 10
            return i + 15
        _RunOnce.__init__(self, name)
        self._stopped = has_stopped_blob
        self._ran = False

    def __enter__(self):
        if False:
            while True:
                i = 10
        r = _RunOnce.__enter__(self)
        self._stopped = ops.Const(True, blob_out=self._stopped)
        return r

    def __exit__(self, etype, *args):
        if False:
            while True:
                i = 10
        if etype is None:
            self._ran = True
            ops.Const(False, blob_out=self._stopped)
        _RunOnce.__exit__(self, etype, *args)

    def has_stopped(self):
        if False:
            return 10
        '\n        Return a blob that will be set to scalar bool `True` after\n        this net builder ran, iff it was halted early.\n        '
        assert self._ran, 'Context not used yet.'
        return self._stopped

class _Loop(NetBuilder):

    def __init__(self, iters=None, name=None):
        if False:
            i = 10
            return i + 15
        NetBuilder.__init__(self, name, _stop_blob_required=True)
        if iters is not None:
            self._inc = ops.Const(1)
            self._iter = ops.Const(0)
            self._num_iters = iters if isinstance(iters, core.BlobReference) else ops.Const(iters)
        else:
            self._num_iters = None

    def iter(self):
        if False:
            while True:
                i = 10
        assert self._num_iters is not None, 'This loop does not have a number of iterations.'
        assert self._iter is not None, 'iter() must be called from inside the loop context'
        return self._iter

    def __enter__(self):
        if False:
            print('Hello World!')
        builder = NetBuilder.__enter__(self)
        if self._num_iters is not None:
            ops.stop_if(ops.GE([self._iter, self._num_iters]))
        return builder

    def __exit__(self, type, *args):
        if False:
            print('Hello World!')
        if type is None and self._num_iters is not None:
            self.current_net().Add([self._iter, self._inc], [self._iter])
        NetBuilder.__exit__(self, type, *args)

class _RunIf(_RunOnce):

    def __init__(self, cond_blob=None, name=None, _already_ran=None):
        if False:
            i = 10
            return i + 15
        _RunOnce.__init__(self, name)
        assert cond_blob or _already_ran
        self._is_else = cond_blob is None
        if _already_ran is None:
            self._else_blob = ops.Not(cond_blob)
            self._already_ran = ops.Const(False)
        else:
            self._already_ran = _already_ran
            self._else_blob = _already_ran if cond_blob is None else ops.Or([_already_ran, ops.Not(cond_blob)])

    def __enter__(self):
        if False:
            while True:
                i = 10
        r = _RunOnce.__enter__(self)
        ops.stop_if(self._else_blob)
        ops.Const(True, blob_out=self._already_ran)
        return r

    def Elif(self, cond, name=None):
        if False:
            print('Hello World!')
        assert not self._is_else, 'Else not allowed for an Else.'
        return NetBuilder.current().add(_RunIf(cond, name=name or self.name, _already_ran=self._already_ran))

    def Else(self, name=None):
        if False:
            i = 10
            return i + 15
        assert not self._is_else, 'Elif not allowed for an Else.'
        return NetBuilder.current().add(_RunIf(name=name or self.name, _already_ran=self._already_ran))

class _RunIfNet(NetBuilder):
    """
    Generates a single net that uses If operator
    """

    def __init__(self, cond_blob, name=None):
        if False:
            return 10
        NetBuilder.__init__(self, name=name, _use_control_ops=True)
        assert cond_blob, 'Conditional blob is not specified for an If net'
        self._cond_blob = cond_blob
        self._then_net = None
        self._else_net = None

    def add(self, child):
        if False:
            print('Hello World!')
        return NetBuilder.add(self, child)

    def __exit__(self, type, *args):
        if False:
            for i in range(10):
                print('nop')
        if type is None:
            _then_nets = self._children
            self._reset_children()
            self._then_net = NetBuilder.merge_nets(_then_nets, self._lexical_scope)
            if not self._then_net:
                self._then_net = core.Net('empty_then_net')
            if_net = core.Net(self.name + '/if_net')
            add_if_op(if_net, self._cond_blob, self._lexical_scope, self._then_net, self._else_net)
            self._current_net = if_net
            self._children = [if_net]
        NetBuilder.__exit__(self, type, *args)

class _RunElseNet(NetBuilder):
    """
    Else branch for _RunIfNet builder
    """

    def __init__(self, name=None):
        if False:
            i = 10
            return i + 15
        NetBuilder.__init__(self, name=name, _use_control_ops=True)
        parent = NetBuilder.current(required=False)
        assert parent and len(parent._children) > 0 and isinstance(parent._children[-1], _RunIfNet), 'Invalid use of Else builder'
        self._if_builder = parent._children[-1]

    def __exit__(self, type, *args):
        if False:
            return 10
        if type is None:
            _else_nets = self._children
            self._reset_children()
            self._if_builder._else_net = NetBuilder.merge_nets(_else_nets, self._lexical_scope)
            if self._if_builder._else_net:
                if_else_net = core.Net(self.name + '/if_else_net')
                add_if_op(if_else_net, self._if_builder._cond_blob, self._lexical_scope, self._if_builder._then_net, self._if_builder._else_net)
                self._if_builder._current_net = if_else_net
                self._if_builder._children = [if_else_net]
        NetBuilder.__exit__(self, type, *args)

class _RunWhileNet(NetBuilder):
    """
    Generates a single net that uses While operator
    """

    def __init__(self, name=None):
        if False:
            return 10
        NetBuilder.__init__(self, name=name, _use_control_ops=True)
        self._cond_builder = None

    def __exit__(self, type, *args):
        if False:
            return 10
        if type is None:
            assert self._cond_builder, 'Condition builder must be specified in While op'
            _cond_blob = self._cond_builder._cond_blob
            _cond_net = self._cond_builder._cond_net
            loop_body = self._children
            self._reset_children()
            loop_body_net = NetBuilder.merge_nets(loop_body, self._lexical_scope)
            if not loop_body_net:
                loop_body_net = core.Net('empty_loop_body_net')
            while_net = core.Net(self.name + '/while_net')
            add_while_op(while_net, _cond_blob, self._lexical_scope, loop_body_net, _cond_net)
            self._current_net = while_net
            self._children = [while_net]
        NetBuilder.__exit__(self, type, *args)

class _RunWhileCondition(NetBuilder):
    """
    Computes loop's condition, used in the context of WhileNet.
    Last operator must have a single scalar boolean output that will be used
    as a condition value, no other blobs created in the condition net are
    visible outside of it
    """

    def __init__(self, name=None):
        if False:
            print('Hello World!')
        NetBuilder.__init__(self, name=name, _use_control_ops=True)
        parent = NetBuilder.current(required=False)
        assert parent and isinstance(parent, _RunWhileNet), 'Invalid use of loop condition builder'
        assert not parent._cond_builder, 'Multiple loop condition builders specified'
        assert len(parent._children) == 0, "Condition definition must be specified before the loop's body"
        parent._cond_builder = self
        self._cond_blob = None
        self._cond_net = None

    def __exit__(self, type, *args):
        if False:
            print('Hello World!')
        if type is None:
            condition_body = self._children
            self._reset_children()
            self._cond_net = NetBuilder.merge_nets(condition_body, self._lexical_scope)
            assert self._cond_net, 'Invalid loop condition specified'
            assert len(self._cond_net.Proto().op) > 0, 'Invalid condition net'
            last_op = self._cond_net.Proto().op[-1]
            assert len(last_op.output) == 1, 'Invalid condition net'
            self._cond_blob = core.BlobReference(name=last_op.output[0], net=None)
            self._current_net = self._cond_net
            self._children = [self._cond_net]
        NetBuilder.__exit__(self, type, *args)