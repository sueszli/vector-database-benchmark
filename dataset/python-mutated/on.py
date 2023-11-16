"""本模块定义事件响应器便携定义函数。

FrontMatter:
    sidebar_position: 2
    description: nonebot.plugin.on 模块
"""
import re
import inspect
import warnings
from types import ModuleType
from datetime import datetime, timedelta
from typing import Any, Set, Dict, List, Type, Tuple, Union, Optional
from nonebot.adapters import Event
from nonebot.permission import Permission
from nonebot.dependencies import Dependent
from nonebot.matcher import Matcher, MatcherSource
from nonebot.typing import T_State, T_Handler, T_RuleChecker, T_PermissionChecker
from nonebot.rule import Rule, ArgumentParser, regex, command, is_type, keyword, endswith, fullmatch, startswith, shell_command
from .model import Plugin
from . import get_plugin_by_module_name
from .manager import _current_plugin_chain

def store_matcher(matcher: Type[Matcher]) -> None:
    if False:
        for i in range(10):
            print('nop')
    '存储一个事件响应器到插件。\n\n    参数:\n        matcher: 事件响应器\n    '
    if (plugin_chain := _current_plugin_chain.get()):
        plugin_chain[-1].matcher.add(matcher)

def get_matcher_plugin(depth: int=1) -> Optional[Plugin]:
    if False:
        print('Hello World!')
    '获取事件响应器定义所在插件。\n\n    **Deprecated**, 请使用 {ref}`nonebot.plugin.on.get_matcher_source` 获取信息。\n\n    参数:\n        depth: 调用栈深度\n    '
    warnings.warn('`get_matcher_plugin` is deprecated, please use `get_matcher_source` instead', DeprecationWarning)
    return (source := get_matcher_source(depth + 1)) and source.plugin

def get_matcher_module(depth: int=1) -> Optional[ModuleType]:
    if False:
        print('Hello World!')
    '获取事件响应器定义所在模块。\n\n    **Deprecated**, 请使用 {ref}`nonebot.plugin.on.get_matcher_source` 获取信息。\n\n    参数:\n        depth: 调用栈深度\n    '
    warnings.warn('`get_matcher_module` is deprecated, please use `get_matcher_source` instead', DeprecationWarning)
    return (source := get_matcher_source(depth + 1)) and source.module

def get_matcher_source(depth: int=1) -> Optional[MatcherSource]:
    if False:
        while True:
            i = 10
    '获取事件响应器定义所在源码信息。\n\n    参数:\n        depth: 调用栈深度\n    '
    current_frame = inspect.currentframe()
    if current_frame is None:
        return None
    frame = inspect.getouterframes(current_frame)[depth + 1].frame
    module_name = (module := inspect.getmodule(frame)) and module.__name__
    plugin: Optional['Plugin'] = None
    if (plugin_chain := _current_plugin_chain.get()):
        plugin = plugin_chain[-1]
    elif module_name:
        plugin = get_plugin_by_module_name(module_name)
    return MatcherSource(plugin_name=plugin and plugin.name, module_name=module_name, lineno=frame.f_lineno)

def on(type: str='', rule: Optional[Union[Rule, T_RuleChecker]]=None, permission: Optional[Union[Permission, T_PermissionChecker]]=None, *, handlers: Optional[List[Union[T_Handler, Dependent]]]=None, temp: bool=False, expire_time: Optional[Union[datetime, timedelta]]=None, priority: int=1, block: bool=False, state: Optional[T_State]=None, _depth: int=0) -> Type[Matcher]:
    if False:
        for i in range(10):
            print('nop')
    '注册一个基础事件响应器，可自定义类型。\n\n    参数:\n        type: 事件响应器类型\n        rule: 事件响应规则\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    matcher = Matcher.new(type, Rule() & rule, Permission() | permission, temp=temp, expire_time=expire_time, priority=priority, block=block, handlers=handlers, source=get_matcher_source(_depth + 1), default_state=state)
    store_matcher(matcher)
    return matcher

def on_metaevent(*args, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        i = 10
        return i + 15
    '注册一个元事件响应器。\n\n    参数:\n        rule: 事件响应规则\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    return on('meta_event', *args, **kwargs, _depth=_depth + 1)

def on_message(*args, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        i = 10
        return i + 15
    '注册一个消息事件响应器。\n\n    参数:\n        rule: 事件响应规则\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    kwargs.setdefault('block', True)
    return on('message', *args, **kwargs, _depth=_depth + 1)

def on_notice(*args, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        print('Hello World!')
    '注册一个通知事件响应器。\n\n    参数:\n        rule: 事件响应规则\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    return on('notice', *args, **kwargs, _depth=_depth + 1)

def on_request(*args, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        i = 10
        return i + 15
    '注册一个请求事件响应器。\n\n    参数:\n        rule: 事件响应规则\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    return on('request', *args, **kwargs, _depth=_depth + 1)

def on_startswith(msg: Union[str, Tuple[str, ...]], rule: Optional[Union[Rule, T_RuleChecker]]=None, ignorecase: bool=False, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        i = 10
        return i + 15
    '注册一个消息事件响应器，并且当消息的**文本部分**以指定内容开头时响应。\n\n    参数:\n        msg: 指定消息开头内容\n        rule: 事件响应规则\n        ignorecase: 是否忽略大小写\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    return on_message(startswith(msg, ignorecase) & rule, **kwargs, _depth=_depth + 1)

def on_endswith(msg: Union[str, Tuple[str, ...]], rule: Optional[Union[Rule, T_RuleChecker]]=None, ignorecase: bool=False, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        while True:
            i = 10
    '注册一个消息事件响应器，并且当消息的**文本部分**以指定内容结尾时响应。\n\n    参数:\n        msg: 指定消息结尾内容\n        rule: 事件响应规则\n        ignorecase: 是否忽略大小写\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    return on_message(endswith(msg, ignorecase) & rule, **kwargs, _depth=_depth + 1)

def on_fullmatch(msg: Union[str, Tuple[str, ...]], rule: Optional[Union[Rule, T_RuleChecker]]=None, ignorecase: bool=False, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        print('Hello World!')
    '注册一个消息事件响应器，并且当消息的**文本部分**与指定内容完全一致时响应。\n\n    参数:\n        msg: 指定消息全匹配内容\n        rule: 事件响应规则\n        ignorecase: 是否忽略大小写\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    return on_message(fullmatch(msg, ignorecase) & rule, **kwargs, _depth=_depth + 1)

def on_keyword(keywords: Set[str], rule: Optional[Union[Rule, T_RuleChecker]]=None, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        for i in range(10):
            print('nop')
    '注册一个消息事件响应器，并且当消息纯文本部分包含关键词时响应。\n\n    参数:\n        keywords: 关键词列表\n        rule: 事件响应规则\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    return on_message(keyword(*keywords) & rule, **kwargs, _depth=_depth + 1)

def on_command(cmd: Union[str, Tuple[str, ...]], rule: Optional[Union[Rule, T_RuleChecker]]=None, aliases: Optional[Set[Union[str, Tuple[str, ...]]]]=None, force_whitespace: Optional[Union[str, bool]]=None, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        while True:
            i = 10
    '注册一个消息事件响应器，并且当消息以指定命令开头时响应。\n\n    命令匹配规则参考: `命令形式匹配 <rule.md#command-command>`_\n\n    参数:\n        cmd: 指定命令内容\n        rule: 事件响应规则\n        aliases: 命令别名\n        force_whitespace: 是否强制命令后必须有指定空白符\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    commands = {cmd} | (aliases or set())
    kwargs.setdefault('block', False)
    return on_message(command(*commands, force_whitespace=force_whitespace) & rule, **kwargs, _depth=_depth + 1)

def on_shell_command(cmd: Union[str, Tuple[str, ...]], rule: Optional[Union[Rule, T_RuleChecker]]=None, aliases: Optional[Set[Union[str, Tuple[str, ...]]]]=None, parser: Optional[ArgumentParser]=None, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        for i in range(10):
            print('nop')
    '注册一个支持 `shell_like` 解析参数的命令消息事件响应器。\n\n    与普通的 `on_command` 不同的是，在添加 `parser` 参数时, 响应器会自动处理消息。\n\n    可以通过 {ref}`nonebot.params.ShellCommandArgv` 获取原始参数列表，\n    通过 {ref}`nonebot.params.ShellCommandArgs` 获取解析后的参数字典。\n\n    参数:\n        cmd: 指定命令内容\n        rule: 事件响应规则\n        aliases: 命令别名\n        parser: `nonebot.rule.ArgumentParser` 对象\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    commands = {cmd} | (aliases or set())
    return on_message(shell_command(*commands, parser=parser) & rule, **kwargs, _depth=_depth + 1)

def on_regex(pattern: str, flags: Union[int, re.RegexFlag]=0, rule: Optional[Union[Rule, T_RuleChecker]]=None, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        while True:
            i = 10
    '注册一个消息事件响应器，并且当消息匹配正则表达式时响应。\n\n    命令匹配规则参考: `正则匹配 <rule.md#regex-regex-flags-0>`_\n\n    参数:\n        pattern: 正则表达式\n        flags: 正则匹配标志\n        rule: 事件响应规则\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    return on_message(regex(pattern, flags) & rule, **kwargs, _depth=_depth + 1)

def on_type(types: Union[Type[Event], Tuple[Type[Event], ...]], rule: Optional[Union[Rule, T_RuleChecker]]=None, *, _depth: int=0, **kwargs) -> Type[Matcher]:
    if False:
        for i in range(10):
            print('nop')
    '注册一个事件响应器，并且当事件为指定类型时响应。\n\n    参数:\n        types: 事件类型\n        rule: 事件响应规则\n        permission: 事件响应权限\n        handlers: 事件处理函数列表\n        temp: 是否为临时事件响应器（仅执行一次）\n        expire_time: 事件响应器最终有效时间点，过时即被删除\n        priority: 事件响应器优先级\n        block: 是否阻止事件向更低优先级传递\n        state: 默认 state\n    '
    event_types = types if isinstance(types, tuple) else (types,)
    return on(rule=is_type(*event_types) & rule, **kwargs, _depth=_depth + 1)

class _Group:

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        '创建一个事件响应器组合，参数为默认值，与 `on` 一致'
        self.matchers: List[Type[Matcher]] = []
        '组内事件响应器列表'
        self.base_kwargs: Dict[str, Any] = kwargs
        '其他传递给 `on` 的参数默认值'

    def _get_final_kwargs(self, update: Dict[str, Any], *, exclude: Optional[Set[str]]=None) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '获取最终传递给 `on` 的参数\n\n        参数:\n            update: 更新的关键字参数\n            exclude: 需要排除的参数\n        '
        final_kwargs = self.base_kwargs.copy()
        final_kwargs.update(update)
        if exclude:
            for key in exclude:
                final_kwargs.pop(key, None)
        final_kwargs['_depth'] = 1
        return final_kwargs

class CommandGroup(_Group):
    """命令组，用于声明一组有相同名称前缀的命令。

    参数:
        cmd: 指定命令内容
        prefix_aliases: 是否影响命令别名，给命令别名加前缀
        rule: 事件响应规则
        permission: 事件响应权限
        handlers: 事件处理函数列表
        temp: 是否为临时事件响应器（仅执行一次）
        expire_time: 事件响应器最终有效时间点，过时即被删除
        priority: 事件响应器优先级
        block: 是否阻止事件向更低优先级传递
        state: 默认 state
    """

    def __init__(self, cmd: Union[str, Tuple[str, ...]], prefix_aliases: bool=False, **kwargs):
        if False:
            while True:
                i = 10
        '命令前缀'
        super().__init__(**kwargs)
        self.basecmd: Tuple[str, ...] = (cmd,) if isinstance(cmd, str) else cmd
        self.base_kwargs.pop('aliases', None)
        self.prefix_aliases = prefix_aliases

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'CommandGroup(cmd={self.basecmd}, matchers={len(self.matchers)})'

    def command(self, cmd: Union[str, Tuple[str, ...]], **kwargs) -> Type[Matcher]:
        if False:
            return 10
        '注册一个新的命令。新参数将会覆盖命令组默认值\n\n        参数:\n            cmd: 指定命令内容\n            aliases: 命令别名\n            force_whitespace: 是否强制命令后必须有指定空白符\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        sub_cmd = (cmd,) if isinstance(cmd, str) else cmd
        cmd = self.basecmd + sub_cmd
        if self.prefix_aliases and (aliases := kwargs.get('aliases')):
            kwargs['aliases'] = {self.basecmd + ((alias,) if isinstance(alias, str) else alias) for alias in aliases}
        matcher = on_command(cmd, **self._get_final_kwargs(kwargs))
        self.matchers.append(matcher)
        return matcher

    def shell_command(self, cmd: Union[str, Tuple[str, ...]], **kwargs) -> Type[Matcher]:
        if False:
            while True:
                i = 10
        '注册一个新的 `shell_like` 命令。新参数将会覆盖命令组默认值\n\n        参数:\n            cmd: 指定命令内容\n            rule: 事件响应规则\n            aliases: 命令别名\n            parser: `nonebot.rule.ArgumentParser` 对象\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        sub_cmd = (cmd,) if isinstance(cmd, str) else cmd
        cmd = self.basecmd + sub_cmd
        if self.prefix_aliases and (aliases := kwargs.get('aliases')):
            kwargs['aliases'] = {self.basecmd + ((alias,) if isinstance(alias, str) else alias) for alias in aliases}
        matcher = on_shell_command(cmd, **self._get_final_kwargs(kwargs))
        self.matchers.append(matcher)
        return matcher

class MatcherGroup(_Group):
    """事件响应器组合，统一管理。为 `Matcher` 创建提供默认属性。"""

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'MatcherGroup(matchers={len(self.matchers)})'

    def on(self, **kwargs) -> Type[Matcher]:
        if False:
            for i in range(10):
                print('nop')
        '注册一个基础事件响应器，可自定义类型。\n\n        参数:\n            type: 事件响应器类型\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        matcher = on(**self._get_final_kwargs(kwargs))
        self.matchers.append(matcher)
        return matcher

    def on_metaevent(self, **kwargs) -> Type[Matcher]:
        if False:
            i = 10
            return i + 15
        '注册一个元事件响应器。\n\n        参数:\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type', 'permission'})
        matcher = on_metaevent(**final_kwargs)
        self.matchers.append(matcher)
        return matcher

    def on_message(self, **kwargs) -> Type[Matcher]:
        if False:
            for i in range(10):
                print('nop')
        '注册一个消息事件响应器。\n\n        参数:\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type'})
        matcher = on_message(**final_kwargs)
        self.matchers.append(matcher)
        return matcher

    def on_notice(self, **kwargs) -> Type[Matcher]:
        if False:
            while True:
                i = 10
        '注册一个通知事件响应器。\n\n        参数:\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type', 'permission'})
        matcher = on_notice(**final_kwargs)
        self.matchers.append(matcher)
        return matcher

    def on_request(self, **kwargs) -> Type[Matcher]:
        if False:
            for i in range(10):
                print('nop')
        '注册一个请求事件响应器。\n\n        参数:\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type', 'permission'})
        matcher = on_request(**final_kwargs)
        self.matchers.append(matcher)
        return matcher

    def on_startswith(self, msg: Union[str, Tuple[str, ...]], **kwargs) -> Type[Matcher]:
        if False:
            i = 10
            return i + 15
        '注册一个消息事件响应器，并且当消息的**文本部分**以指定内容开头时响应。\n\n        参数:\n            msg: 指定消息开头内容\n            ignorecase: 是否忽略大小写\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type'})
        matcher = on_startswith(msg, **final_kwargs)
        self.matchers.append(matcher)
        return matcher

    def on_endswith(self, msg: Union[str, Tuple[str, ...]], **kwargs) -> Type[Matcher]:
        if False:
            while True:
                i = 10
        '注册一个消息事件响应器，并且当消息的**文本部分**以指定内容结尾时响应。\n\n        参数:\n            msg: 指定消息结尾内容\n            ignorecase: 是否忽略大小写\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type'})
        matcher = on_endswith(msg, **final_kwargs)
        self.matchers.append(matcher)
        return matcher

    def on_fullmatch(self, msg: Union[str, Tuple[str, ...]], **kwargs) -> Type[Matcher]:
        if False:
            for i in range(10):
                print('nop')
        '注册一个消息事件响应器，并且当消息的**文本部分**与指定内容完全一致时响应。\n\n        参数:\n            msg: 指定消息全匹配内容\n            rule: 事件响应规则\n            ignorecase: 是否忽略大小写\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type'})
        matcher = on_fullmatch(msg, **final_kwargs)
        self.matchers.append(matcher)
        return matcher

    def on_keyword(self, keywords: Set[str], **kwargs) -> Type[Matcher]:
        if False:
            return 10
        '注册一个消息事件响应器，并且当消息纯文本部分包含关键词时响应。\n\n        参数:\n            keywords: 关键词列表\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type'})
        matcher = on_keyword(keywords, **final_kwargs)
        self.matchers.append(matcher)
        return matcher

    def on_command(self, cmd: Union[str, Tuple[str, ...]], aliases: Optional[Set[Union[str, Tuple[str, ...]]]]=None, force_whitespace: Optional[Union[str, bool]]=None, **kwargs) -> Type[Matcher]:
        if False:
            while True:
                i = 10
        '注册一个消息事件响应器，并且当消息以指定命令开头时响应。\n\n        命令匹配规则参考: `命令形式匹配 <rule.md#command-command>`_\n\n        参数:\n            cmd: 指定命令内容\n            aliases: 命令别名\n            force_whitespace: 是否强制命令后必须有指定空白符\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type'})
        matcher = on_command(cmd, aliases=aliases, force_whitespace=force_whitespace, **final_kwargs)
        self.matchers.append(matcher)
        return matcher

    def on_shell_command(self, cmd: Union[str, Tuple[str, ...]], aliases: Optional[Set[Union[str, Tuple[str, ...]]]]=None, parser: Optional[ArgumentParser]=None, **kwargs) -> Type[Matcher]:
        if False:
            for i in range(10):
                print('nop')
        '注册一个支持 `shell_like` 解析参数的命令消息事件响应器。\n\n        与普通的 `on_command` 不同的是，在添加 `parser` 参数时, 响应器会自动处理消息。\n\n        可以通过 {ref}`nonebot.params.ShellCommandArgv` 获取原始参数列表，\n        通过 {ref}`nonebot.params.ShellCommandArgs` 获取解析后的参数字典。\n\n        参数:\n            cmd: 指定命令内容\n            aliases: 命令别名\n            parser: `nonebot.rule.ArgumentParser` 对象\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type'})
        matcher = on_shell_command(cmd, aliases=aliases, parser=parser, **final_kwargs)
        self.matchers.append(matcher)
        return matcher

    def on_regex(self, pattern: str, flags: Union[int, re.RegexFlag]=0, **kwargs) -> Type[Matcher]:
        if False:
            print('Hello World!')
        '注册一个消息事件响应器，并且当消息匹配正则表达式时响应。\n\n        命令匹配规则参考: `正则匹配 <rule.md#regex-regex-flags-0>`_\n\n        参数:\n            pattern: 正则表达式\n            flags: 正则匹配标志\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type'})
        matcher = on_regex(pattern, flags=flags, **final_kwargs)
        self.matchers.append(matcher)
        return matcher

    def on_type(self, types: Union[Type[Event], Tuple[Type[Event]]], **kwargs) -> Type[Matcher]:
        if False:
            for i in range(10):
                print('nop')
        '注册一个事件响应器，并且当事件为指定类型时响应。\n\n        参数:\n            types: 事件类型\n            rule: 事件响应规则\n            permission: 事件响应权限\n            handlers: 事件处理函数列表\n            temp: 是否为临时事件响应器（仅执行一次）\n            expire_time: 事件响应器最终有效时间点，过时即被删除\n            priority: 事件响应器优先级\n            block: 是否阻止事件向更低优先级传递\n            state: 默认 state\n        '
        final_kwargs = self._get_final_kwargs(kwargs, exclude={'type'})
        matcher = on_type(types, **final_kwargs)
        self.matchers.append(matcher)
        return matcher