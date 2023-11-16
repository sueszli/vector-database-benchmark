from __future__ import print_function
import sys
__doc__ = '\n    Generic Taskmaster module for the SCons build engine.\n    =====================================================\n\n    This module contains the primary interface(s) between a wrapping user\n    interface and the SCons build engine.  There are two key classes here:\n\n    Taskmaster\n    ----------\n        This is the main engine for walking the dependency graph and\n        calling things to decide what does or doesn\'t need to be built.\n\n    Task\n    ----\n        This is the base class for allowing a wrapping interface to\n        decide what does or doesn\'t actually need to be done.  The\n        intention is for a wrapping interface to subclass this as\n        appropriate for different types of behavior it may need.\n\n        The canonical example is the SCons native Python interface,\n        which has Task subclasses that handle its specific behavior,\n        like printing "\'foo\' is up to date" when a top-level target\n        doesn\'t need to be built, and handling the -c option by removing\n        targets as its "build" action.  There is also a separate subclass\n        for suppressing this output when the -q option is used.\n\n        The Taskmaster instantiates a Task object for each (set of)\n        target(s) that it decides need to be evaluated and/or built.\n'
__revision__ = 'src/engine/SCons/Taskmaster.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
from itertools import chain
import operator
import sys
import traceback
import SCons.Errors
import SCons.Node
import SCons.Warnings
StateString = SCons.Node.StateString
NODE_NO_STATE = SCons.Node.no_state
NODE_PENDING = SCons.Node.pending
NODE_EXECUTING = SCons.Node.executing
NODE_UP_TO_DATE = SCons.Node.up_to_date
NODE_EXECUTED = SCons.Node.executed
NODE_FAILED = SCons.Node.failed
print_prepare = 0
CollectStats = None

class Stats(object):
    """
    A simple class for holding statistics about the disposition of a
    Node by the Taskmaster.  If we're collecting statistics, each Node
    processed by the Taskmaster gets one of these attached, in which case
    the Taskmaster records its decision each time it processes the Node.
    (Ideally, that's just once per Node.)
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Instantiates a Taskmaster.Stats object, initializing all\n        appropriate counters to zero.\n        '
        self.considered = 0
        self.already_handled = 0
        self.problem = 0
        self.child_failed = 0
        self.not_built = 0
        self.side_effects = 0
        self.build = 0
StatsNodes = []
fmt = '%(considered)3d %(already_handled)3d %(problem)3d %(child_failed)3d %(not_built)3d %(side_effects)3d %(build)3d '

def dump_stats():
    if False:
        while True:
            i = 10
    for n in sorted(StatsNodes, key=lambda a: str(a)):
        print(fmt % n.attributes.stats.__dict__ + str(n))

class Task(object):
    """
    Default SCons build engine task.

    This controls the interaction of the actual building of node
    and the rest of the engine.

    This is expected to handle all of the normally-customizable
    aspects of controlling a build, so any given application
    *should* be able to do what it wants by sub-classing this
    class and overriding methods as appropriate.  If an application
    needs to customize something by sub-classing Taskmaster (or
    some other build engine class), we should first try to migrate
    that functionality into this class.

    Note that it's generally a good idea for sub-classes to call
    these methods explicitly to update state, etc., rather than
    roll their own interaction with Taskmaster from scratch.
    """

    def __init__(self, tm, targets, top, node):
        if False:
            i = 10
            return i + 15
        self.tm = tm
        self.targets = targets
        self.top = top
        self.node = node
        self.exc_clear()

    def trace_message(self, method, node, description='node'):
        if False:
            for i in range(10):
                print('nop')
        fmt = '%-20s %s %s\n'
        return fmt % (method + ':', description, self.tm.trace_node(node))

    def display(self, message):
        if False:
            while True:
                i = 10
        '\n        Hook to allow the calling interface to display a message.\n\n        This hook gets called as part of preparing a task for execution\n        (that is, a Node to be built).  As part of figuring out what Node\n        should be built next, the actual target list may be altered,\n        along with a message describing the alteration.  The calling\n        interface can subclass Task and provide a concrete implementation\n        of this method to see those messages.\n        '
        pass

    def prepare(self):
        if False:
            print('Hello World!')
        '\n        Called just before the task is executed.\n\n        This is mainly intended to give the target Nodes a chance to\n        unlink underlying files and make all necessary directories before\n        the Action is actually called to build the targets.\n        '
        global print_prepare
        T = self.tm.trace
        if T:
            T.write(self.trace_message(u'Task.prepare()', self.node))
        self.exception_raise()
        if self.tm.message:
            self.display(self.tm.message)
            self.tm.message = None
        executor = self.targets[0].get_executor()
        if executor is None:
            return
        executor.prepare()
        for t in executor.get_action_targets():
            if print_prepare:
                print('Preparing target %s...' % t)
                for s in t.side_effects:
                    print('...with side-effect %s...' % s)
            t.prepare()
            for s in t.side_effects:
                if print_prepare:
                    print('...Preparing side-effect %s...' % s)
                s.prepare()

    def get_target(self):
        if False:
            while True:
                i = 10
        'Fetch the target being built or updated by this task.\n        '
        return self.node

    def needs_execute(self):
        if False:
            print('Hello World!')
        msg = 'Taskmaster.Task is an abstract base class; instead of\n\tusing it directly, derive from it and override the abstract methods.'
        SCons.Warnings.warn(SCons.Warnings.TaskmasterNeedsExecuteWarning, msg)
        return True

    def execute(self):
        if False:
            i = 10
            return i + 15
        '\n        Called to execute the task.\n\n        This method is called from multiple threads in a parallel build,\n        so only do thread safe stuff here.  Do thread unsafe stuff in\n        prepare(), executed() or failed().\n        '
        T = self.tm.trace
        if T:
            T.write(self.trace_message(u'Task.execute()', self.node))
        try:
            cached_targets = []
            for t in self.targets:
                if not t.retrieve_from_cache():
                    break
                cached_targets.append(t)
            if len(cached_targets) < len(self.targets):
                for t in cached_targets:
                    try:
                        t.fs.unlink(t.get_internal_path())
                    except (IOError, OSError):
                        pass
                self.targets[0].build()
            else:
                for t in cached_targets:
                    t.cached = 1
        except SystemExit:
            exc_value = sys.exc_info()[1]
            raise SCons.Errors.ExplicitExit(self.targets[0], exc_value.code)
        except SCons.Errors.UserError:
            raise
        except SCons.Errors.BuildError:
            raise
        except Exception as e:
            buildError = SCons.Errors.convert_to_BuildError(e)
            buildError.node = self.targets[0]
            buildError.exc_info = sys.exc_info()
            raise buildError

    def executed_without_callbacks(self):
        if False:
            return 10
        "\n        Called when the task has been successfully executed\n        and the Taskmaster instance doesn't want to call\n        the Node's callback methods.\n        "
        T = self.tm.trace
        if T:
            T.write(self.trace_message('Task.executed_without_callbacks()', self.node))
        for t in self.targets:
            if t.get_state() == NODE_EXECUTING:
                for side_effect in t.side_effects:
                    side_effect.set_state(NODE_NO_STATE)
                t.set_state(NODE_EXECUTED)

    def executed_with_callbacks(self):
        if False:
            i = 10
            return i + 15
        '\n        Called when the task has been successfully executed and\n        the Taskmaster instance wants to call the Node\'s callback\n        methods.\n\n        This may have been a do-nothing operation (to preserve build\n        order), so we must check the node\'s state before deciding whether\n        it was "built", in which case we call the appropriate Node method.\n        In any event, we always call "visited()", which will handle any\n        post-visit actions that must take place regardless of whether\n        or not the target was an actual built target or a source Node.\n        '
        global print_prepare
        T = self.tm.trace
        if T:
            T.write(self.trace_message('Task.executed_with_callbacks()', self.node))
        for t in self.targets:
            if t.get_state() == NODE_EXECUTING:
                for side_effect in t.side_effects:
                    side_effect.set_state(NODE_NO_STATE)
                t.set_state(NODE_EXECUTED)
                if not t.cached:
                    t.push_to_cache()
                t.built()
                t.visited()
                if not print_prepare and (not hasattr(self, 'options') or not self.options.debug_includes):
                    t.release_target_info()
            else:
                t.visited()
    executed = executed_with_callbacks

    def failed(self):
        if False:
            return 10
        '\n        Default action when a task fails:  stop the build.\n\n        Note: Although this function is normally invoked on nodes in\n        the executing state, it might also be invoked on up-to-date\n        nodes when using Configure().\n        '
        self.fail_stop()

    def fail_stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Explicit stop-the-build failure.\n\n        This sets failure status on the target nodes and all of\n        their dependent parent nodes.\n\n        Note: Although this function is normally invoked on nodes in\n        the executing state, it might also be invoked on up-to-date\n        nodes when using Configure().\n        '
        T = self.tm.trace
        if T:
            T.write(self.trace_message('Task.failed_stop()', self.node))
        self.tm.will_not_build(self.targets, lambda n: n.set_state(NODE_FAILED))
        self.tm.stop()
        self.targets = [self.tm.current_top]
        self.top = 1

    def fail_continue(self):
        if False:
            while True:
                i = 10
        '\n        Explicit continue-the-build failure.\n\n        This sets failure status on the target nodes and all of\n        their dependent parent nodes.\n\n        Note: Although this function is normally invoked on nodes in\n        the executing state, it might also be invoked on up-to-date\n        nodes when using Configure().\n        '
        T = self.tm.trace
        if T:
            T.write(self.trace_message('Task.failed_continue()', self.node))
        self.tm.will_not_build(self.targets, lambda n: n.set_state(NODE_FAILED))

    def make_ready_all(self):
        if False:
            return 10
        '\n        Marks all targets in a task ready for execution.\n\n        This is used when the interface needs every target Node to be\n        visited--the canonical example being the "scons -c" option.\n        '
        T = self.tm.trace
        if T:
            T.write(self.trace_message('Task.make_ready_all()', self.node))
        self.out_of_date = self.targets[:]
        for t in self.targets:
            t.disambiguate().set_state(NODE_EXECUTING)
            for s in t.side_effects:
                s.disambiguate().set_state(NODE_EXECUTING)

    def make_ready_current(self):
        if False:
            while True:
                i = 10
        "\n        Marks all targets in a task ready for execution if any target\n        is not current.\n\n        This is the default behavior for building only what's necessary.\n        "
        global print_prepare
        T = self.tm.trace
        if T:
            T.write(self.trace_message(u'Task.make_ready_current()', self.node))
        self.out_of_date = []
        needs_executing = False
        for t in self.targets:
            try:
                t.disambiguate().make_ready()
                is_up_to_date = not t.has_builder() or (not t.always_build and t.is_up_to_date())
            except EnvironmentError as e:
                raise SCons.Errors.BuildError(node=t, errstr=e.strerror, filename=e.filename)
            if not is_up_to_date:
                self.out_of_date.append(t)
                needs_executing = True
        if needs_executing:
            for t in self.targets:
                t.set_state(NODE_EXECUTING)
                for s in t.side_effects:
                    s.disambiguate().set_state(NODE_EXECUTING)
        else:
            for t in self.targets:
                t.visited()
                t.set_state(NODE_UP_TO_DATE)
                if not print_prepare and (not hasattr(self, 'options') or not self.options.debug_includes):
                    t.release_target_info()
    make_ready = make_ready_current

    def postprocess(self):
        if False:
            i = 10
            return i + 15
        "\n        Post-processes a task after it's been executed.\n\n        This examines all the targets just built (or not, we don't care\n        if the build was successful, or even if there was no build\n        because everything was up-to-date) to see if they have any\n        waiting parent Nodes, or Nodes waiting on a common side effect,\n        that can be put back on the candidates list.\n        "
        T = self.tm.trace
        if T:
            T.write(self.trace_message(u'Task.postprocess()', self.node))
        targets = set(self.targets)
        pending_children = self.tm.pending_children
        parents = {}
        for t in targets:
            if t.waiting_parents:
                if T:
                    T.write(self.trace_message(u'Task.postprocess()', t, 'removing'))
                pending_children.discard(t)
            for p in t.waiting_parents:
                parents[p] = parents.get(p, 0) + 1
            t.waiting_parents = set()
        for t in targets:
            if t.side_effects is not None:
                for s in t.side_effects:
                    if s.get_state() == NODE_EXECUTING:
                        s.set_state(NODE_NO_STATE)
                    if s.get_state() == NODE_NO_STATE and s.waiting_parents:
                        pending_children.discard(s)
                        for p in s.waiting_parents:
                            parents[p] = parents.get(p, 0) + 1
                        s.waiting_parents = set()
                    for p in s.waiting_s_e:
                        if p.ref_count == 0:
                            self.tm.candidates.append(p)
        for (p, subtract) in parents.items():
            p.ref_count = p.ref_count - subtract
            if T:
                T.write(self.trace_message(u'Task.postprocess()', p, 'adjusted parent ref count'))
            if p.ref_count == 0:
                self.tm.candidates.append(p)
        for t in targets:
            t.postprocess()

    def exc_info(self):
        if False:
            return 10
        '\n        Returns info about a recorded exception.\n        '
        return self.exception

    def exc_clear(self):
        if False:
            i = 10
            return i + 15
        '\n        Clears any recorded exception.\n\n        This also changes the "exception_raise" attribute to point\n        to the appropriate do-nothing method.\n        '
        self.exception = (None, None, None)
        self.exception_raise = self._no_exception_to_raise

    def exception_set(self, exception=None):
        if False:
            while True:
                i = 10
        '\n        Records an exception to be raised at the appropriate time.\n\n        This also changes the "exception_raise" attribute to point\n        to the method that will, in fact\n        '
        if not exception:
            exception = sys.exc_info()
        self.exception = exception
        self.exception_raise = self._exception_raise

    def _no_exception_to_raise(self):
        if False:
            i = 10
            return i + 15
        pass

    def _exception_raise(self):
        if False:
            while True:
                i = 10
        '\n        Raises a pending exception that was recorded while getting a\n        Task ready for execution.\n        '
        exc = self.exc_info()[:]
        try:
            (exc_type, exc_value, exc_traceback) = exc
        except ValueError:
            (exc_type, exc_value) = exc
            exc_traceback = None
        if sys.version_info[0] == 2:
            exec('raise exc_type, exc_value, exc_traceback')
        elif isinstance(exc_value, Exception):
            exec('raise exc_value.with_traceback(exc_traceback)')
        else:
            exec('raise exc_type(exc_value).with_traceback(exc_traceback)')

class AlwaysTask(Task):

    def needs_execute(self):
        if False:
            while True:
                i = 10
        '\n        Always returns True (indicating this Task should always\n        be executed).\n\n        Subclasses that need this behavior (as opposed to the default\n        of only executing Nodes that are out of date w.r.t. their\n        dependencies) can use this as follows:\n\n            class MyTaskSubclass(SCons.Taskmaster.Task):\n                needs_execute = SCons.Taskmaster.Task.execute_always\n        '
        return True

class OutOfDateTask(Task):

    def needs_execute(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns True (indicating this Task should be executed) if this\n        Task's target state indicates it needs executing, which has\n        already been determined by an earlier up-to-date check.\n        "
        return self.targets[0].get_state() == SCons.Node.executing

def find_cycle(stack, visited):
    if False:
        for i in range(10):
            print('nop')
    if stack[-1] in visited:
        return None
    visited.add(stack[-1])
    for n in stack[-1].waiting_parents:
        stack.append(n)
        if stack[0] == stack[-1]:
            return stack
        if find_cycle(stack, visited):
            return stack
        stack.pop()
    return None

class Taskmaster(object):
    """
    The Taskmaster for walking the dependency DAG.
    """

    def __init__(self, targets=[], tasker=None, order=None, trace=None):
        if False:
            while True:
                i = 10
        self.original_top = targets
        self.top_targets_left = targets[:]
        self.top_targets_left.reverse()
        self.candidates = []
        if tasker is None:
            tasker = OutOfDateTask
        self.tasker = tasker
        if not order:
            order = lambda l: l
        self.order = order
        self.message = None
        self.trace = trace
        self.next_candidate = self.find_next_candidate
        self.pending_children = set()

    def find_next_candidate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the next candidate Node for (potential) evaluation.\n\n        The candidate list (really a stack) initially consists of all of\n        the top-level (command line) targets provided when the Taskmaster\n        was initialized.  While we walk the DAG, visiting Nodes, all the\n        children that haven\'t finished processing get pushed on to the\n        candidate list.  Each child can then be popped and examined in\n        turn for whether *their* children are all up-to-date, in which\n        case a Task will be created for their actual evaluation and\n        potential building.\n\n        Here is where we also allow candidate Nodes to alter the list of\n        Nodes that should be examined.  This is used, for example, when\n        invoking SCons in a source directory.  A source directory Node can\n        return its corresponding build directory Node, essentially saying,\n        "Hey, you really need to build this thing over here instead."\n        '
        try:
            return self.candidates.pop()
        except IndexError:
            pass
        try:
            node = self.top_targets_left.pop()
        except IndexError:
            return None
        self.current_top = node
        (alt, message) = node.alter_targets()
        if alt:
            self.message = message
            self.candidates.append(node)
            self.candidates.extend(self.order(alt))
            node = self.candidates.pop()
        return node

    def no_next_candidate(self):
        if False:
            while True:
                i = 10
        '\n        Stops Taskmaster processing by not returning a next candidate.\n\n        Note that we have to clean-up the Taskmaster candidate list\n        because the cycle detection depends on the fact all nodes have\n        been processed somehow.\n        '
        while self.candidates:
            candidates = self.candidates
            self.candidates = []
            self.will_not_build(candidates)
        return None

    def _validate_pending_children(self):
        if False:
            i = 10
            return i + 15
        '\n        Validate the content of the pending_children set. Assert if an\n        internal error is found.\n\n        This function is used strictly for debugging the taskmaster by\n        checking that no invariants are violated. It is not used in\n        normal operation.\n\n        The pending_children set is used to detect cycles in the\n        dependency graph. We call a "pending child" a child that is\n        found in the "pending" state when checking the dependencies of\n        its parent node.\n\n        A pending child can occur when the Taskmaster completes a loop\n        through a cycle. For example, let\'s imagine a graph made of\n        three nodes (A, B and C) making a cycle. The evaluation starts\n        at node A. The Taskmaster first considers whether node A\'s\n        child B is up-to-date. Then, recursively, node B needs to\n        check whether node C is up-to-date. This leaves us with a\n        dependency graph looking like::\n\n                                          Next candidate                                                                       Node A (Pending) --> Node B(Pending) --> Node C (NoState)\n                    ^                                     |\n                    |                                     |\n                    +-------------------------------------+\n\n        Now, when the Taskmaster examines the Node C\'s child Node A,\n        it finds that Node A is in the "pending" state. Therefore,\n        Node A is a pending child of node C.\n\n        Pending children indicate that the Taskmaster has potentially\n        loop back through a cycle. We say potentially because it could\n        also occur when a DAG is evaluated in parallel. For example,\n        consider the following graph::\n\n            Node A (Pending) --> Node B(Pending) --> Node C (Pending) --> ...\n                    |                                     ^\n                    |                                     |\n                    +----------> Node D (NoState) --------+\n                                      /\n                      Next candidate /\n\n        The Taskmaster first evaluates the nodes A, B, and C and\n        starts building some children of node C. Assuming, that the\n        maximum parallel level has not been reached, the Taskmaster\n        will examine Node D. It will find that Node C is a pending\n        child of Node D.\n\n        In summary, evaluating a graph with a cycle will always\n        involve a pending child at one point. A pending child might\n        indicate either a cycle or a diamond-shaped DAG. Only a\n        fraction of the nodes ends-up being a "pending child" of\n        another node. This keeps the pending_children set small in\n        practice.\n\n        We can differentiate between the two cases if we wait until\n        the end of the build. At this point, all the pending children\n        nodes due to a diamond-shaped DAG will have been properly\n        built (or will have failed to build). But, the pending\n        children involved in a cycle will still be in the pending\n        state.\n\n        The taskmaster removes nodes from the pending_children set as\n        soon as a pending_children node moves out of the pending\n        state. This also helps to keep the pending_children set small.\n        '
        for n in self.pending_children:
            assert n.state in (NODE_PENDING, NODE_EXECUTING), (str(n), StateString[n.state])
            assert len(n.waiting_parents) != 0, (str(n), len(n.waiting_parents))
            for p in n.waiting_parents:
                assert p.ref_count > 0, (str(n), str(p), p.ref_count)

    def trace_message(self, message):
        if False:
            while True:
                i = 10
        return 'Taskmaster: %s\n' % message

    def trace_node(self, node):
        if False:
            print('Hello World!')
        return '<%-10s %-3s %s>' % (StateString[node.get_state()], node.ref_count, repr(str(node)))

    def _find_next_ready_node(self):
        if False:
            while True:
                i = 10
        '\n        Finds the next node that is ready to be built.\n\n        This is *the* main guts of the DAG walk.  We loop through the\n        list of candidates, looking for something that has no un-built\n        children (i.e., that is a leaf Node or has dependencies that are\n        all leaf Nodes or up-to-date).  Candidate Nodes are re-scanned\n        (both the target Node itself and its sources, which are always\n        scanned in the context of a given target) to discover implicit\n        dependencies.  A Node that must wait for some children to be\n        built will be put back on the candidates list after the children\n        have finished building.  A Node that has been put back on the\n        candidates list in this way may have itself (or its sources)\n        re-scanned, in order to handle generated header files (e.g.) and\n        the implicit dependencies therein.\n\n        Note that this method does not do any signature calculation or\n        up-to-date check itself.  All of that is handled by the Task\n        class.  This is purely concerned with the dependency graph walk.\n        '
        self.ready_exc = None
        T = self.trace
        if T:
            T.write(SCons.Util.UnicodeType('\n') + self.trace_message('Looking for a node to evaluate'))
        while True:
            node = self.next_candidate()
            if node is None:
                if T:
                    T.write(self.trace_message('No candidate anymore.') + u'\n')
                return None
            node = node.disambiguate()
            state = node.get_state()
            if CollectStats:
                if not hasattr(node.attributes, 'stats'):
                    node.attributes.stats = Stats()
                    StatsNodes.append(node)
                S = node.attributes.stats
                S.considered = S.considered + 1
            else:
                S = None
            if T:
                T.write(self.trace_message(u'    Considering node %s and its children:' % self.trace_node(node)))
            if state == NODE_NO_STATE:
                node.set_state(NODE_PENDING)
            elif state > NODE_PENDING:
                if S:
                    S.already_handled = S.already_handled + 1
                if T:
                    T.write(self.trace_message(u'       already handled (executed)'))
                continue
            executor = node.get_executor()
            try:
                children = executor.get_all_children()
            except SystemExit:
                exc_value = sys.exc_info()[1]
                e = SCons.Errors.ExplicitExit(node, exc_value.code)
                self.ready_exc = (SCons.Errors.ExplicitExit, e)
                if T:
                    T.write(self.trace_message('       SystemExit'))
                return node
            except Exception as e:
                self.ready_exc = sys.exc_info()
                if S:
                    S.problem = S.problem + 1
                if T:
                    T.write(self.trace_message('       exception %s while scanning children.\n' % e))
                return node
            children_not_visited = []
            children_pending = set()
            children_not_ready = []
            children_failed = False
            for child in chain(executor.get_all_prerequisites(), children):
                childstate = child.get_state()
                if T:
                    T.write(self.trace_message(u'       ' + self.trace_node(child)))
                if childstate == NODE_NO_STATE:
                    children_not_visited.append(child)
                elif childstate == NODE_PENDING:
                    children_pending.add(child)
                elif childstate == NODE_FAILED:
                    children_failed = True
                if childstate <= NODE_EXECUTING:
                    children_not_ready.append(child)
            if children_not_visited:
                if len(children_not_visited) > 1:
                    children_not_visited.reverse()
                self.candidates.extend(self.order(children_not_visited))
            if children_failed:
                for n in executor.get_action_targets():
                    n.set_state(NODE_FAILED)
                if S:
                    S.child_failed = S.child_failed + 1
                if T:
                    T.write(self.trace_message('****** %s\n' % self.trace_node(node)))
                continue
            if children_not_ready:
                for child in children_not_ready:
                    if S:
                        S.not_built = S.not_built + 1
                    node.ref_count = node.ref_count + child.add_to_waiting_parents(node)
                    if T:
                        T.write(self.trace_message(u'     adjusted ref count: %s, child %s' % (self.trace_node(node), repr(str(child)))))
                if T:
                    for pc in children_pending:
                        T.write(self.trace_message('       adding %s to the pending children set\n' % self.trace_node(pc)))
                self.pending_children = self.pending_children | children_pending
                continue
            wait_side_effects = False
            for se in executor.get_action_side_effects():
                if se.get_state() == NODE_EXECUTING:
                    se.add_to_waiting_s_e(node)
                    wait_side_effects = True
            if wait_side_effects:
                if S:
                    S.side_effects = S.side_effects + 1
                continue
            if S:
                S.build = S.build + 1
            if T:
                T.write(self.trace_message(u'Evaluating %s\n' % self.trace_node(node)))
            return node
        return None

    def next_task(self):
        if False:
            print('Hello World!')
        '\n        Returns the next task to be executed.\n\n        This simply asks for the next Node to be evaluated, and then wraps\n        it in the specific Task subclass with which we were initialized.\n        '
        node = self._find_next_ready_node()
        if node is None:
            return None
        executor = node.get_executor()
        if executor is None:
            return None
        tlist = executor.get_all_targets()
        task = self.tasker(self, tlist, node in self.original_top, node)
        try:
            task.make_ready()
        except Exception as e:
            self.ready_exc = sys.exc_info()
        if self.ready_exc:
            task.exception_set(self.ready_exc)
        self.ready_exc = None
        return task

    def will_not_build(self, nodes, node_func=lambda n: None):
        if False:
            while True:
                i = 10
        '\n        Perform clean-up about nodes that will never be built. Invokes\n        a user defined function on all of these nodes (including all\n        of their parents).\n        '
        T = self.trace
        pending_children = self.pending_children
        to_visit = set(nodes)
        pending_children = pending_children - to_visit
        if T:
            for n in nodes:
                T.write(self.trace_message('       removing node %s from the pending children set\n' % self.trace_node(n)))
        try:
            while len(to_visit):
                node = to_visit.pop()
                node_func(node)
                parents = node.waiting_parents
                node.waiting_parents = set()
                to_visit = to_visit | parents
                pending_children = pending_children - parents
                for p in parents:
                    p.ref_count = p.ref_count - 1
                    if T:
                        T.write(self.trace_message('       removing parent %s from the pending children set\n' % self.trace_node(p)))
        except KeyError:
            pass
        self.pending_children = pending_children

    def stop(self):
        if False:
            i = 10
            return i + 15
        '\n        Stops the current build completely.\n        '
        self.next_candidate = self.no_next_candidate

    def cleanup(self):
        if False:
            print('Hello World!')
        '\n        Check for dependency cycles.\n        '
        if not self.pending_children:
            return
        nclist = [(n, find_cycle([n], set())) for n in self.pending_children]
        genuine_cycles = [node for (node, cycle) in nclist if cycle or node.get_state() != NODE_EXECUTED]
        if not genuine_cycles:
            return
        desc = 'Found dependency cycle(s):\n'
        for (node, cycle) in nclist:
            if cycle:
                desc = desc + '  ' + ' -> '.join(map(str, cycle)) + '\n'
            else:
                desc = desc + '  Internal Error: no cycle found for node %s (%s) in state %s\n' % (node, repr(node), StateString[node.get_state()])
        raise SCons.Errors.UserError(desc)