from __future__ import annotations
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
display = Display()
__all__ = ['PlayIterator', 'IteratingStates', 'FailedStates']

class IteratingStates(IntEnum):
    SETUP = 0
    TASKS = 1
    RESCUE = 2
    ALWAYS = 3
    HANDLERS = 4
    COMPLETE = 5

class FailedStates(IntFlag):
    NONE = 0
    SETUP = 1
    TASKS = 2
    RESCUE = 4
    ALWAYS = 8
    HANDLERS = 16

class HostState:

    def __init__(self, blocks):
        if False:
            print('Hello World!')
        self._blocks = blocks[:]
        self.handlers = []
        self.handler_notifications = []
        self.cur_block = 0
        self.cur_regular_task = 0
        self.cur_rescue_task = 0
        self.cur_always_task = 0
        self.cur_handlers_task = 0
        self.run_state = IteratingStates.SETUP
        self.fail_state = FailedStates.NONE
        self.pre_flushing_run_state = None
        self.update_handlers = True
        self.pending_setup = False
        self.tasks_child_state = None
        self.rescue_child_state = None
        self.always_child_state = None
        self.did_rescue = False
        self.did_start_at_task = False

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'HostState(%r)' % self._blocks

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'HOST STATE: block=%d, task=%d, rescue=%d, always=%d, handlers=%d, run_state=%s, fail_state=%s, pre_flushing_run_state=%s, update_handlers=%s, pending_setup=%s, tasks child state? (%s), rescue child state? (%s), always child state? (%s), did rescue? %s, did start at task? %s' % (self.cur_block, self.cur_regular_task, self.cur_rescue_task, self.cur_always_task, self.cur_handlers_task, self.run_state, self.fail_state, self.pre_flushing_run_state, self.update_handlers, self.pending_setup, self.tasks_child_state, self.rescue_child_state, self.always_child_state, self.did_rescue, self.did_start_at_task)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, HostState):
            return False
        for attr in ('_blocks', 'cur_block', 'cur_regular_task', 'cur_rescue_task', 'cur_always_task', 'cur_handlers_task', 'run_state', 'fail_state', 'pre_flushing_run_state', 'update_handlers', 'pending_setup', 'tasks_child_state', 'rescue_child_state', 'always_child_state'):
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def get_current_block(self):
        if False:
            return 10
        return self._blocks[self.cur_block]

    def copy(self):
        if False:
            print('Hello World!')
        new_state = HostState(self._blocks)
        new_state.handlers = self.handlers[:]
        new_state.handler_notifications = self.handler_notifications[:]
        new_state.cur_block = self.cur_block
        new_state.cur_regular_task = self.cur_regular_task
        new_state.cur_rescue_task = self.cur_rescue_task
        new_state.cur_always_task = self.cur_always_task
        new_state.cur_handlers_task = self.cur_handlers_task
        new_state.run_state = self.run_state
        new_state.fail_state = self.fail_state
        new_state.pre_flushing_run_state = self.pre_flushing_run_state
        new_state.update_handlers = self.update_handlers
        new_state.pending_setup = self.pending_setup
        new_state.did_rescue = self.did_rescue
        new_state.did_start_at_task = self.did_start_at_task
        if self.tasks_child_state is not None:
            new_state.tasks_child_state = self.tasks_child_state.copy()
        if self.rescue_child_state is not None:
            new_state.rescue_child_state = self.rescue_child_state.copy()
        if self.always_child_state is not None:
            new_state.always_child_state = self.always_child_state.copy()
        return new_state

class PlayIterator:

    def __init__(self, inventory, play, play_context, variable_manager, all_vars, start_at_done=False):
        if False:
            print('Hello World!')
        self._play = play
        self._blocks = []
        self._variable_manager = variable_manager
        setup_block = Block(play=self._play)
        setup_block.run_once = False
        setup_task = Task(block=setup_block)
        setup_task.action = 'gather_facts'
        setup_task.resolved_action = 'ansible.builtin.gather_facts'
        setup_task.name = 'Gathering Facts'
        setup_task.args = {}
        if not self._play.tags:
            setup_task.tags = ['always']
        for option in ('gather_subset', 'gather_timeout', 'fact_path'):
            value = getattr(self._play, option, None)
            if value is not None:
                setup_task.args[option] = value
        setup_task.set_loader(self._play._loader)
        if self._play._included_conditional is not None:
            setup_task.when = self._play._included_conditional[:]
        setup_block.block = [setup_task]
        setup_block = setup_block.filter_tagged_tasks(all_vars)
        self._blocks.append(setup_block)
        self.all_tasks = setup_block.get_tasks()
        for block in self._play.compile():
            new_block = block.filter_tagged_tasks(all_vars)
            if new_block.has_tasks():
                self._blocks.append(new_block)
                self.all_tasks.extend(new_block.get_tasks())
        self.handlers = [h for b in self._play.handlers for h in b.block]
        self._host_states = {}
        start_at_matched = False
        batch = inventory.get_hosts(self._play.hosts, order=self._play.order)
        self.batch_size = len(batch)
        for host in batch:
            self.set_state_for_host(host.name, HostState(blocks=self._blocks))
            if play_context.start_at_task is not None and (not start_at_done):
                while True:
                    (s, task) = self.get_next_task_for_host(host, peek=True)
                    if s.run_state == IteratingStates.COMPLETE:
                        break
                    if task.name == play_context.start_at_task or (task.name and fnmatch.fnmatch(task.name, play_context.start_at_task)) or task.get_name() == play_context.start_at_task or fnmatch.fnmatch(task.get_name(), play_context.start_at_task):
                        start_at_matched = True
                        break
                    self.set_state_for_host(host.name, s)
                if start_at_matched:
                    self._host_states[host.name].did_start_at_task = True
                    self._host_states[host.name].run_state = IteratingStates.SETUP
        if start_at_matched:
            play_context.start_at_task = None
        self.end_play = False
        self.cur_task = 0

    def get_host_state(self, host):
        if False:
            print('Hello World!')
        if host.name not in self._host_states:
            self.set_state_for_host(host.name, HostState(blocks=[]))
        return self._host_states[host.name].copy()

    def get_next_task_for_host(self, host, peek=False):
        if False:
            print('Hello World!')
        display.debug('getting the next task for host %s' % host.name)
        s = self.get_host_state(host)
        task = None
        if s.run_state == IteratingStates.COMPLETE:
            display.debug('host %s is done iterating, returning' % host.name)
            return (s, None)
        (s, task) = self._get_next_task_from_state(s, host=host)
        if not peek:
            self.set_state_for_host(host.name, s)
        display.debug('done getting next task for host %s' % host.name)
        display.debug(' ^ task is: %s' % task)
        display.debug(' ^ state is: %s' % s)
        return (s, task)

    def _get_next_task_from_state(self, state, host):
        if False:
            return 10
        task = None
        while True:
            try:
                block = state._blocks[state.cur_block]
            except IndexError:
                state.run_state = IteratingStates.COMPLETE
                return (state, None)
            if state.run_state == IteratingStates.SETUP:
                if not state.pending_setup:
                    state.pending_setup = True
                    gathering = C.DEFAULT_GATHERING
                    implied = self._play.gather_facts is None or boolean(self._play.gather_facts, strict=False)
                    if gathering == 'implicit' and implied or (gathering == 'explicit' and boolean(self._play.gather_facts, strict=False)) or (gathering == 'smart' and implied and (not self._variable_manager._fact_cache.get(host.name, {}).get('_ansible_facts_gathered', False))):
                        setup_block = self._blocks[0]
                        if setup_block.has_tasks() and len(setup_block.block) > 0:
                            task = setup_block.block[0]
                else:
                    state.pending_setup = False
                    state.run_state = IteratingStates.TASKS
                    if not state.did_start_at_task:
                        state.cur_block += 1
                        state.cur_regular_task = 0
                        state.cur_rescue_task = 0
                        state.cur_always_task = 0
                        state.tasks_child_state = None
                        state.rescue_child_state = None
                        state.always_child_state = None
            elif state.run_state == IteratingStates.TASKS:
                if state.pending_setup:
                    state.pending_setup = False
                if state.tasks_child_state:
                    (state.tasks_child_state, task) = self._get_next_task_from_state(state.tasks_child_state, host=host)
                    if self._check_failed_state(state.tasks_child_state):
                        state.tasks_child_state = None
                        self._set_failed_state(state)
                    elif task is None or state.tasks_child_state.run_state == IteratingStates.COMPLETE:
                        state.tasks_child_state = None
                        continue
                elif self._check_failed_state(state):
                    state.run_state = IteratingStates.RESCUE
                elif state.cur_regular_task >= len(block.block):
                    state.run_state = IteratingStates.ALWAYS
                else:
                    task = block.block[state.cur_regular_task]
                    if isinstance(task, Block):
                        state.tasks_child_state = HostState(blocks=[task])
                        state.tasks_child_state.run_state = IteratingStates.TASKS
                        task = None
                    state.cur_regular_task += 1
            elif state.run_state == IteratingStates.RESCUE:
                if state.rescue_child_state:
                    (state.rescue_child_state, task) = self._get_next_task_from_state(state.rescue_child_state, host=host)
                    if self._check_failed_state(state.rescue_child_state):
                        state.rescue_child_state = None
                        self._set_failed_state(state)
                    elif task is None or state.rescue_child_state.run_state == IteratingStates.COMPLETE:
                        state.rescue_child_state = None
                        continue
                elif state.fail_state & FailedStates.RESCUE == FailedStates.RESCUE:
                    state.run_state = IteratingStates.ALWAYS
                elif state.cur_rescue_task >= len(block.rescue):
                    if len(block.rescue) > 0:
                        state.fail_state = FailedStates.NONE
                    state.run_state = IteratingStates.ALWAYS
                    state.did_rescue = True
                else:
                    task = block.rescue[state.cur_rescue_task]
                    if isinstance(task, Block):
                        state.rescue_child_state = HostState(blocks=[task])
                        state.rescue_child_state.run_state = IteratingStates.TASKS
                        task = None
                    state.cur_rescue_task += 1
            elif state.run_state == IteratingStates.ALWAYS:
                if state.always_child_state:
                    (state.always_child_state, task) = self._get_next_task_from_state(state.always_child_state, host=host)
                    if self._check_failed_state(state.always_child_state):
                        state.always_child_state = None
                        self._set_failed_state(state)
                    elif task is None or state.always_child_state.run_state == IteratingStates.COMPLETE:
                        state.always_child_state = None
                        continue
                elif state.cur_always_task >= len(block.always):
                    if state.fail_state != FailedStates.NONE:
                        state.run_state = IteratingStates.COMPLETE
                    else:
                        state.cur_block += 1
                        state.cur_regular_task = 0
                        state.cur_rescue_task = 0
                        state.cur_always_task = 0
                        state.run_state = IteratingStates.TASKS
                        state.tasks_child_state = None
                        state.rescue_child_state = None
                        state.always_child_state = None
                        state.did_rescue = False
                else:
                    task = block.always[state.cur_always_task]
                    if isinstance(task, Block):
                        state.always_child_state = HostState(blocks=[task])
                        state.always_child_state.run_state = IteratingStates.TASKS
                        task = None
                    state.cur_always_task += 1
            elif state.run_state == IteratingStates.HANDLERS:
                if state.update_handlers:
                    state.handlers = self.handlers[:]
                    state.update_handlers = False
                    state.cur_handlers_task = 0
                while True:
                    try:
                        task = state.handlers[state.cur_handlers_task]
                    except IndexError:
                        task = None
                        state.run_state = state.pre_flushing_run_state
                        state.update_handlers = True
                        break
                    else:
                        state.cur_handlers_task += 1
                        if task.is_host_notified(host):
                            break
            elif state.run_state == IteratingStates.COMPLETE:
                return (state, None)
            if task:
                break
        return (state, task)

    def _set_failed_state(self, state):
        if False:
            i = 10
            return i + 15
        if state.run_state == IteratingStates.SETUP:
            state.fail_state |= FailedStates.SETUP
            state.run_state = IteratingStates.COMPLETE
        elif state.run_state == IteratingStates.TASKS:
            if state.tasks_child_state is not None:
                state.tasks_child_state = self._set_failed_state(state.tasks_child_state)
            else:
                state.fail_state |= FailedStates.TASKS
                if state._blocks[state.cur_block].rescue:
                    state.run_state = IteratingStates.RESCUE
                elif state._blocks[state.cur_block].always:
                    state.run_state = IteratingStates.ALWAYS
                else:
                    state.run_state = IteratingStates.COMPLETE
        elif state.run_state == IteratingStates.RESCUE:
            if state.rescue_child_state is not None:
                state.rescue_child_state = self._set_failed_state(state.rescue_child_state)
            else:
                state.fail_state |= FailedStates.RESCUE
                if state._blocks[state.cur_block].always:
                    state.run_state = IteratingStates.ALWAYS
                else:
                    state.run_state = IteratingStates.COMPLETE
        elif state.run_state == IteratingStates.ALWAYS:
            if state.always_child_state is not None:
                state.always_child_state = self._set_failed_state(state.always_child_state)
            else:
                state.fail_state |= FailedStates.ALWAYS
                state.run_state = IteratingStates.COMPLETE
        return state

    def mark_host_failed(self, host):
        if False:
            return 10
        s = self.get_host_state(host)
        display.debug('marking host %s failed, current state: %s' % (host, s))
        if s.run_state == IteratingStates.HANDLERS:
            s.run_state = s.pre_flushing_run_state
            s.update_handlers = True
        s = self._set_failed_state(s)
        display.debug('^ failed state is now: %s' % s)
        self.set_state_for_host(host.name, s)
        self._play._removed_hosts.append(host.name)

    def get_failed_hosts(self):
        if False:
            print('Hello World!')
        return dict(((host, True) for (host, state) in self._host_states.items() if self._check_failed_state(state)))

    def _check_failed_state(self, state):
        if False:
            return 10
        if state is None:
            return False
        elif state.run_state == IteratingStates.RESCUE and self._check_failed_state(state.rescue_child_state):
            return True
        elif state.run_state == IteratingStates.ALWAYS and self._check_failed_state(state.always_child_state):
            return True
        elif state.fail_state != FailedStates.NONE:
            if state.run_state == IteratingStates.RESCUE and state.fail_state & FailedStates.RESCUE == 0:
                return False
            elif state.run_state == IteratingStates.ALWAYS and state.fail_state & FailedStates.ALWAYS == 0:
                return False
            else:
                return not (state.did_rescue and state.fail_state & FailedStates.ALWAYS == 0)
        elif state.run_state == IteratingStates.TASKS and self._check_failed_state(state.tasks_child_state):
            cur_block = state._blocks[state.cur_block]
            if len(cur_block.rescue) > 0 and state.fail_state & FailedStates.RESCUE == 0:
                return False
            else:
                return True
        return False

    def is_failed(self, host):
        if False:
            while True:
                i = 10
        s = self.get_host_state(host)
        return self._check_failed_state(s)

    def clear_host_errors(self, host):
        if False:
            while True:
                i = 10
        self._clear_state_errors(self.get_state_for_host(host.name))

    def _clear_state_errors(self, state: HostState) -> None:
        if False:
            print('Hello World!')
        state.fail_state = FailedStates.NONE
        if state.tasks_child_state is not None:
            self._clear_state_errors(state.tasks_child_state)
        elif state.rescue_child_state is not None:
            self._clear_state_errors(state.rescue_child_state)
        elif state.always_child_state is not None:
            self._clear_state_errors(state.always_child_state)

    def get_active_state(self, state):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finds the active state, recursively if necessary when there are child states.\n        '
        if state.run_state == IteratingStates.TASKS and state.tasks_child_state is not None:
            return self.get_active_state(state.tasks_child_state)
        elif state.run_state == IteratingStates.RESCUE and state.rescue_child_state is not None:
            return self.get_active_state(state.rescue_child_state)
        elif state.run_state == IteratingStates.ALWAYS and state.always_child_state is not None:
            return self.get_active_state(state.always_child_state)
        return state

    def is_any_block_rescuing(self, state):
        if False:
            return 10
        '\n        Given the current HostState state, determines if the current block, or any child blocks,\n        are in rescue mode.\n        '
        if state.run_state == IteratingStates.TASKS and state.get_current_block().rescue:
            return True
        if state.tasks_child_state is not None:
            return self.is_any_block_rescuing(state.tasks_child_state)
        if state.rescue_child_state is not None:
            return self.is_any_block_rescuing(state.rescue_child_state)
        if state.always_child_state is not None:
            return self.is_any_block_rescuing(state.always_child_state)
        return False

    def _insert_tasks_into_state(self, state, task_list):
        if False:
            for i in range(10):
                print('nop')
        if state.fail_state != FailedStates.NONE and state.run_state == IteratingStates.TASKS or not task_list:
            return state
        if state.run_state == IteratingStates.TASKS:
            if state.tasks_child_state:
                state.tasks_child_state = self._insert_tasks_into_state(state.tasks_child_state, task_list)
            else:
                target_block = state._blocks[state.cur_block].copy()
                before = target_block.block[:state.cur_regular_task]
                after = target_block.block[state.cur_regular_task:]
                target_block.block = before + task_list + after
                state._blocks[state.cur_block] = target_block
        elif state.run_state == IteratingStates.RESCUE:
            if state.rescue_child_state:
                state.rescue_child_state = self._insert_tasks_into_state(state.rescue_child_state, task_list)
            else:
                target_block = state._blocks[state.cur_block].copy()
                before = target_block.rescue[:state.cur_rescue_task]
                after = target_block.rescue[state.cur_rescue_task:]
                target_block.rescue = before + task_list + after
                state._blocks[state.cur_block] = target_block
        elif state.run_state == IteratingStates.ALWAYS:
            if state.always_child_state:
                state.always_child_state = self._insert_tasks_into_state(state.always_child_state, task_list)
            else:
                target_block = state._blocks[state.cur_block].copy()
                before = target_block.always[:state.cur_always_task]
                after = target_block.always[state.cur_always_task:]
                target_block.always = before + task_list + after
                state._blocks[state.cur_block] = target_block
        elif state.run_state == IteratingStates.HANDLERS:
            state.handlers[state.cur_handlers_task:state.cur_handlers_task] = [h for b in task_list for h in b.block]
        return state

    def add_tasks(self, host, task_list):
        if False:
            for i in range(10):
                print('nop')
        self.set_state_for_host(host.name, self._insert_tasks_into_state(self.get_host_state(host), task_list))

    @property
    def host_states(self):
        if False:
            while True:
                i = 10
        return self._host_states

    def get_state_for_host(self, hostname: str) -> HostState:
        if False:
            for i in range(10):
                print('nop')
        return self._host_states[hostname]

    def set_state_for_host(self, hostname: str, state: HostState) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(state, HostState):
            raise AnsibleAssertionError('Expected state to be a HostState but was a %s' % type(state))
        self._host_states[hostname] = state

    def set_run_state_for_host(self, hostname: str, run_state: IteratingStates) -> None:
        if False:
            print('Hello World!')
        if not isinstance(run_state, IteratingStates):
            raise AnsibleAssertionError('Expected run_state to be a IteratingStates but was %s' % type(run_state))
        self._host_states[hostname].run_state = run_state

    def set_fail_state_for_host(self, hostname: str, fail_state: FailedStates) -> None:
        if False:
            i = 10
            return i + 15
        if not isinstance(fail_state, FailedStates):
            raise AnsibleAssertionError('Expected fail_state to be a FailedStates but was %s' % type(fail_state))
        self._host_states[hostname].fail_state = fail_state

    def add_notification(self, hostname: str, notification: str) -> None:
        if False:
            return 10
        host_state = self._host_states[hostname]
        if notification not in host_state.handler_notifications:
            host_state.handler_notifications.append(notification)

    def clear_notification(self, hostname: str, notification: str) -> None:
        if False:
            while True:
                i = 10
        self._host_states[hostname].handler_notifications.remove(notification)