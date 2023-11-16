from projectModel.base import projectBase
import public

class main(projectBase):

    @staticmethod
    def hosts_fun():
        if False:
            for i in range(10):
                print('nop')
        import projectModel.bt_docker.host as dh
        return dh.docker_host()

    def set_hosts(self, args):
        if False:
            print('Hello World!')
        '\n        操作主机 添加/删除\n        :param args:\n        :return:\n        '
        if args.act == 'add':
            return self.hosts_fun().add(args)
        else:
            return self.hosts_fun().delete(args)

    def get_hosts_list(self, args=None):
        if False:
            return 10
        '\n        获取主机列表\n        :param args:\n        :return:\n        '
        return self.hosts_fun().get_list()

    def compose_fun(self):
        if False:
            print('Hello World!')
        import projectModel.bt_docker.compose as bc
        return bc.compose()

    def compose_create(self, args):
        if False:
            print('Hello World!')
        return self.compose_fun().create(args)

    def compose_project_list(self, args):
        if False:
            i = 10
            return i + 15
        return self.compose_fun().compose_project_list(args)

    def compose_remove(self, args):
        if False:
            while True:
                i = 10
        return self.compose_fun().remove(args)

    def compose_start(self, args):
        if False:
            return 10
        return self.compose_fun().start(args)

    def compose_stop(self, args):
        if False:
            i = 10
            return i + 15
        return self.compose_fun().stop(args)

    def compose_restart(self, args):
        if False:
            i = 10
            return i + 15
        return self.compose_fun().restart(args)

    def compose_pull(self, args):
        if False:
            return 10
        return self.compose_fun().pull(args)

    def compose_pause(self, args):
        if False:
            print('Hello World!')
        return self.compose_fun().pause(args)

    def compose_unpause(self, args):
        if False:
            print('Hello World!')
        return self.compose_fun().unpause(args)

    def compose_add_template(self, args):
        if False:
            return 10
        return self.compose_fun().add_template(args)

    def compose_remove_template(self, args):
        if False:
            i = 10
            return i + 15
        return self.compose_fun().remove_template(args)

    def compose_template_list(self, args):
        if False:
            print('Hello World!')
        return self.compose_fun().template_list()

    @staticmethod
    def containers_fun():
        if False:
            while True:
                i = 10
        import projectModel.bt_docker.container as dc
        return dc.contianer()

    def get_all_containers(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        获取所有容器的详细配置\n        :param url\n        :param args:\n        :return:\n        '
        return self.containers_fun().get_list(args)

    def get_containers_logs(self, args):
        if False:
            return 10
        '\n        获取某个容器的日志\n        :param args:\n        :return:\n        '
        return self.containers_fun().get_logs(args)

    def run_a_container(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        创建并运行一个容器\n        :return:\n        '
        return self.containers_fun().run(args)

    def delete_a_container(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param id\n        :param args:\n        :return:\n        '
        return self.containers_fun().del_container(args)

    def commit_a_container(self, args):
        if False:
            for i in range(10):
                print('nop')
        return self.containers_fun().commit(args)

    def export_a_container(self, args):
        if False:
            for i in range(10):
                print('nop')
        return self.containers_fun().export(args)

    @staticmethod
    def image_fun():
        if False:
            for i in range(10):
                print('nop')
        import projectModel.bt_docker.image as di
        return di.image()

    def image_list(self, args):
        if False:
            print('Hello World!')
        return self.image_fun().image_list(args)

    def image_save(self, args):
        if False:
            i = 10
            return i + 15
        return self.image_fun().save(args)

    def image_load(self, args):
        if False:
            for i in range(10):
                print('nop')
        return self.image_fun().load(args)

    def image_pull(self, args):
        if False:
            return 10
        return self.image_fun().pull(args)

    def image_pull_from(self, args):
        if False:
            return 10
        return self.image_fun().pull_from_some_registry(args)

    def image_remove(self, args):
        if False:
            print('Hello World!')
        return self.image_fun().remove(args)

    def image_push(self, args):
        if False:
            print('Hello World!')
        return self.image_fun().push(args)

    def image_build(self, args):
        if False:
            while True:
                i = 10
        return self.image_fun().build(args)

    @staticmethod
    def registry_fun():
        if False:
            i = 10
            return i + 15
        import projectModel.bt_docker.registry as di
        return di.registry()

    def registry_list(self, args):
        if False:
            while True:
                i = 10
        return self.registry_fun().registry_list()

    def registry_add(self, args):
        if False:
            for i in range(10):
                print('nop')
        return self.registry_fun().add(args)

    def registry_remove(self, args):
        if False:
            return 10
        return self.registry_fun().remove(args)

    def get_screen_data(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        获取大屏数据\n        :return:\n        '
        data = {'host_lists': self.get_hosts_list(), 'container_total': self.container_for_all_hosts(), 'image_total': self.image_for_all_host()}
        return public.returnMsg(True, data)

    def container_for_all_hosts(self, args=None):
        if False:
            i = 10
            return i + 15
        '\n        获取所有服务器的容器数量\n        :param args:\n        :return:\n        '
        import projectModel.bt_docker.public as dp
        hosts = dp.sql('hosts').select()
        num = 0
        for i in hosts:
            args.url = i['url']
            res = self.container_for_host(args)
            if not res['status']:
                continue
            num += res['msg']
        return public.returnMsg(True, num)

    def container_for_host(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        获取某台服务器的docker容器数量\n        :param url\n        :param args:\n        :return:\n        '
        res = self.get_all_containers(args)
        if not res['status']:
            return res
        return public.returnMsg(True, len(res['msg']))

    def image_for_host(self, args):
        if False:
            return 10
        '\n        获取镜像大小和获取镜像数量\n        :param args:\n        :return:\n        '
        res = self.image_list(args)
        if not res['status']:
            return res
        num = len(res['msg'])
        size = 0
        for i in res['msg']:
            size += i['Size']
        return public.returnMsg(True, {'num': num, 'size': size})

    def image_for_all_host(self, args=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        获取所有服务器的镜像数量和大小\n        :param args:\n        :return:\n        '
        import projectModel.bt_docker.public as dp
        hosts = dp.sql('hosts').select()
        num = 0
        size = 0
        for i in hosts:
            args.url = i['url']
            res = self.image_for_host(args)
            if not res['status']:
                continue
            num += res['msg']['num']
            size += res['msg']['size']
        return public.returnMsg(True, {'num': num, 'size': size})

    @staticmethod
    def network_fun():
        if False:
            print('Hello World!')
        import projectModel.bt_docker.network as dn
        return dn.network()

    def get_host_network(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        获取主机上的所有网络\n        :param args:\n        :return:\n        '
        return self.network_fun().get_host_network(args)

    def add_network(self, args):
        if False:
            i = 10
            return i + 15
        '\n        添加一个网络\n        :param args:\n        :return:\n        '
        return self.network_fun().add(args)

    def del_network(self, args):
        if False:
            return 10
        '\n\n        :param args:\n        :return:\n        '
        return self.network_fun().del_network(args)

    @staticmethod
    def volume_fun():
        if False:
            return 10
        import projectModel.bt_docker.volume as dv
        return dv.volume()

    def get_volume_lists(self, args):
        if False:
            return 10
        return self.volume_fun().get_volume_list(args)

    def add_volume(self, args):
        if False:
            print('Hello World!')
        return self.volume_fun().add(args)

    def remove_volume(self, args):
        if False:
            while True:
                i = 10
        return self.volume_fun().remove(args)