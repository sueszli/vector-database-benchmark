import os, sys, public

class panelSiteController:

    def __init__(self):
        if False:
            return 10
        pass

    def get_parser_list(self, args):
        if False:
            print('Hello World!')
        '\n            @name 获取支持的解释器列表\n            @author hwliang<2021-07-13>\n            @param args<dict_obj>\n            @return list\n        '
        return public.return_data(True, public.read_config('parser'))

    def get_parser_versions(self, args):
        if False:
            return 10
        '\n            @name 获取指定解释器可用版本列表\n            @author hwliang<2021-07-13>\n            @param args<dict_obj>{\n                parser_name: string<解释器名称>\n            }\n            @return list\n        '
        try:
            public.exists_args('parser_name', args)
        except Exception as ex:
            return public.return_data(False, None, 1001, ex)
        parser_name = args.parser_name.strip()
        config_data = public.read_config('parser')
        versions = []
        result = public.return_data(False, versions)
        for parser_data in config_data:
            if parser_data['name'] != parser_name:
                continue
            if not parser_data['show']:
                return result
            if not parser_data['versions']:
                return result
            for version in parser_data['versions']:
                if isinstance(version['check'], str):
                    version['check'] = [version['check']]
                for check in version['check']:
                    if os.path.exists(check):
                        versions.append(version)
        return public.return_data(True, versions)

    def create_site(self, args):
        if False:
            return 10
        '\n            @name 创建网站\n            @author hwliang<2021-07-13>\n            @param args<dict_obj> {\n                data: {\n                    siteName: string<网站名称>,\n                    domains: list<域名列表>,  // 如：["www.bt.cn:80","bt.cn:80"]\n                    parser_type: string<解释器类型>, // 从 get_parser_list 接口中获取\n                    parser_version: string<解释器版本>, // 从 get_parser_versions 接口中获取\n                    ps: string<网站备注>,\n                    type_id: int<分类标识>,\n                    path: string<网站根目录>,\n                    stream_info: { // TCP、UDP时传入\n                        is_stream: bool<是否为stream>,\n                        pool: string<协议类型TCP/UDP>,\n                        dst_address: string<目标地址>,\n                        dst_port: int<目标端口>,\n                        local_port: int<本地映射端口>\n                    },\n                    process_info: { //绑定进程时传入\n                        is_process: bool<是否为启动指定文件>,\n                        cwd: string<运行目录>,\n                        run_file: string<启动文件>,\n                        run_args: string<启动参数>,\n                        run_cmd: string<启动命令> //与 run_file/run_args 互斥\n                        env: list<环境变量>\n                    },\n                    ftp_info: { //需要同时创建FTP时传入\n                        create: bool<是否创建>,\n                        username: string<用户名>,\n                        password: string<密码>,\n                        path: string<根目录>\n                    },\n                    database_info: {  //需要同时创建数据库时传入\n                        create: bool<是否创建>,\n                        username: string<用户名>,\n                        password: string<密码>,\n                        db_name: string<数据库名>,\n                        codeing: string<字符集>\n                    }\n                }\n            }\n        '