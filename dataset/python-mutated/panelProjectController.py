import os, sys, public, json, re

class ProjectController:

    def __init__(self):
        if False:
            return 10
        pass

    def model(self, args):
        if False:
            while True:
                i = 10
        '\n            @name 调用指定项目模型\n            @author hwliang<2021-07-15>\n            @param args<dict_obj> {\n                mod_name: string<模型名称>\n                def_name: string<方法名称>\n                data: JSON\n            }\n        '
        try:
            if args['mod_name'] in ['base']:
                return public.return_status_code(1000, '错误的调用!')
            public.exists_args('def_name,mod_name', args)
            if args['def_name'].find('__') != -1:
                return public.return_status_code(1000, '调用的方法名称中不能包含“__”字符')
            if not re.match('^\\w+$', args['mod_name']):
                return public.return_status_code(1000, '调用的模块名称中不能包含\\w以外的字符')
            if not re.match('^\\w+$', args['def_name']):
                return public.return_status_code(1000, '调用的方法名称中不能包含\\w以外的字符')
        except:
            return public.get_error_object()
        mod_name = '{}Model'.format(args['mod_name'].strip())
        def_name = args['def_name'].strip()
        mod_file = '{}/projectModel/{}.py'.format(public.get_class_path(), mod_name)
        if not os.path.exists(mod_file):
            return public.return_status_code(1003, mod_name)
        def_object = public.get_script_object(mod_file)
        if not def_object:
            return public.return_status_code(1000, '没有找到{}模型'.format(mod_name))
        run_object = getattr(def_object.main(), def_name, None)
        if not run_object:
            return public.return_status_code(1000, '没有在{}模型中找到{}方法'.format(mod_name, def_name))
        if not hasattr(args, 'data'):
            args.data = {}
        if args.data:
            if isinstance(args.data, str):
                try:
                    pdata = public.to_dict_obj(json.loads(args.data))
                except:
                    return public.get_error_object()
            else:
                pdata = args.data
        else:
            pdata = args
        hook_index = '{}_{}_LAST'.format(mod_name.upper(), def_name.upper())
        hook_result = public.exec_hook(hook_index, pdata)
        if isinstance(hook_result, public.dict_obj):
            pdata = hook_result
        elif isinstance(hook_result, dict):
            return hook_result
        elif isinstance(hook_result, bool):
            if not hook_result:
                return public.return_data(False, {}, error_msg='前置HOOK中断操作')
        result = run_object(pdata)
        hook_index = '{}_{}_END'.format(mod_name.upper(), def_name.upper())
        hook_data = public.to_dict_obj({'args': pdata, 'result': result})
        hook_result = public.exec_hook(hook_index, hook_data)
        if isinstance(hook_result, dict):
            result = hook_result['result']
        return result