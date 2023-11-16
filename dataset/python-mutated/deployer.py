import time
from datetime import datetime
import os
import pwd
import re
from flask import current_app
from flask_socketio import emit
from walle.model.project import ProjectModel
from walle.model.record import RecordModel
from walle.model.task import TaskModel
from walle.service.code import Code
from walle.service.error import WalleError
from walle.service.utils import color_clean
from walle.service.utils import excludes_format, includes_format
from walle.service.notice import Notice
from walle.service.waller import Waller
from walle.service.git.repo import Repo
from flask_login import current_user

class Deployer:
    """
    序列号
    """
    stage = 'init'
    sequence = 0
    stage_prev_deploy = 'prev_deploy'
    stage_deploy = 'deploy'
    stage_post_deploy = 'post_deploy'
    stage_prev_release = 'prev_release'
    stage_release = 'release'
    stage_post_release = 'post_release'
    task_id = '0'
    user_id = '0'
    taskMdl = None
    TaskRecord = None
    console = False
    custom_global_env = {}
    version = datetime.now().strftime('%Y%m%d%H%M%S')
    (local_codebase, dir_codebase_project, project_name) = (None, None, None)
    (dir_release, dir_webroot) = (None, None)
    (connections, success, errors) = ({}, {}, {})
    (release_version_tar, previous_release_version, release_version) = (None, None, None)
    local = None

    def __init__(self, task_id=None, project_id=None, console=False):
        if False:
            return 10
        self.local_codebase = current_app.config.get('CODE_BASE').rstrip('/') + '/'
        self.localhost = Waller(host='127.0.0.1')
        self.TaskRecord = RecordModel()
        if task_id:
            self.task_id = task_id
            current_app.logger.info(self.task_id)
            self.taskMdl = TaskModel().item(self.task_id)
            self.user_id = self.taskMdl.get('user_id')
            self.servers = self.taskMdl.get('servers_info')
            self.project_info = self.taskMdl.get('project_info')
            self.release_version = self.taskMdl.get('link_id') if self.taskMdl.get('is_rollback') else '{project_id}_{task_id}_{timestamp}'.format(project_id=self.project_info['id'], task_id=self.task_id, timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
            current_app.logger.info(self.taskMdl)
            format_export = lambda val: '"%s"' % str(val).replace('"', '').replace("'", '')
            self.custom_global_env = {'WEBROOT': str(self.project_info['target_root']), 'VERSION': str(self.release_version), 'CURRENT_RELEASE': str(self.project_info['target_releases']), 'BRANCH': format_export(self.taskMdl.get('branch')), 'TAG': str(self.taskMdl.get('tag')), 'COMMIT_ID': str(self.taskMdl.get('commit_id')), 'PROJECT_NAME': format_export(self.project_info['name']), 'PROJECT_ID': str(self.project_info['id']), 'TASK_NAME': format_export(self.taskMdl.get('name')), 'TASK_ID': str(self.task_id), 'DEPLOY_USER': str(self.taskMdl.get('user_name')), 'DEPLOY_TIME': str(time.strftime('%Y%m%d-%H:%M:%S', time.localtime(time.time())))}
            if self.project_info['task_vars']:
                task_vars = [i.strip() for i in self.project_info['task_vars'].split('\n') if i.strip() and (not i.strip().startswith('#'))]
                for var in task_vars:
                    var_list = var.split('=', 1)
                    if len(var_list) != 2:
                        continue
                    self.custom_global_env[var_list[0].strip()] = var_list[1].strip()
            self.localhost.init_env(env=self.custom_global_env)
        if project_id:
            self.project_id = project_id
            self.project_info = ProjectModel(id=project_id).item()
            self.servers = self.project_info['servers_info']
        self.project_name = self.project_info['id']
        self.dir_codebase_project = self.local_codebase + str(self.project_name)
        self.console = console

    def config(self, console=None):
        if False:
            i = 10
            return i + 15
        return {'task_id': self.task_id, 'user_id': self.user_id, 'stage': self.stage, 'sequence': self.sequence, 'console': console if console is not None else self.console}

    def start(self):
        if False:
            print('Hello World!')
        RecordModel().query.filter_by(task_id=self.task_id).delete()
        TaskModel().get_by_id(self.task_id).update({'status': TaskModel.status_doing})
        self.taskMdl = TaskModel().item(self.task_id)

    def prev_deploy(self):
        if False:
            print('Hello World!')
        '\n        # TODO\n        socketio.sleep(0.001)\n        1.代码检出前要做的基础工作\n        - 检查 当前用户\n        - 检查 python 版本\n        - 检查 git 版本\n        - 检查 目录是否存在\n        - 用户自定义命令\n\n        :return:\n        '
        self.stage = self.stage_prev_deploy
        self.sequence = 1
        self.init_repo()
        commands = self.project_info['prev_deploy']
        if commands:
            for command in commands.split('\n'):
                if command.strip().startswith('#') or not command.strip():
                    continue
                with self.localhost.cd(self.dir_codebase_project):
                    result = self.localhost.local(command, wenv=self.config())

    def deploy(self):
        if False:
            return 10
        '\n        2.检出代码\n\n        :param project_name:\n        :return:\n        '
        self.stage = self.stage_deploy
        self.sequence = 2
        with self.localhost.cd(self.local_codebase):
            command = 'cp -rf %s %s' % (self.dir_codebase_project, self.release_version)
            current_app.logger.info('cd %s  command: %s  ', self.dir_codebase_project, command)
            result = self.localhost.local(command, wenv=self.config())
        repo = Repo(self.local_codebase + self.release_version)
        if self.project_info['repo_mode'] == ProjectModel.repo_mode_branch:
            repo.checkout_2_commit(branch=self.taskMdl['branch'], commit=self.taskMdl['commit_id'])
        else:
            repo.checkout_2_tag(tag=self.taskMdl['tag'])

    def post_deploy(self):
        if False:
            print('Hello World!')
        '\n        3.检出代码后要做的任务\n        - 用户自定义操作命令\n        - 代码编译\n        - 清除日志文件及无用文件\n        -\n        - 压缩打包\n        - 传送到版本库 release\n        :return:\n        '
        self.stage = self.stage_post_deploy
        self.sequence = 3
        commands = self.project_info['post_deploy']
        if commands:
            for command in commands.split('\n'):
                if command.strip().startswith('#') or not command.strip():
                    continue
                with self.localhost.cd(self.local_codebase + self.release_version):
                    result = self.localhost.local(command, wenv=self.config())
        self.release_version_tar = '%s.tgz' % self.release_version
        with self.localhost.cd(self.local_codebase):
            if self.project_info['is_include']:
                files = includes_format(self.release_version, self.project_info['excludes'])
            else:
                files = excludes_format(self.release_version, self.project_info['excludes'])
            command = 'tar zcf %s/%s %s' % (self.local_codebase.rstrip('/'), self.release_version_tar, files)
            result = self.localhost.local(command, wenv=self.config())

    def prev_release(self, waller):
        if False:
            print('Hello World!')
        '\n        4.部署代码到目标机器前做的任务\n        - 检查 webroot 父目录是否存在\n        :return:\n        '
        self.stage = self.stage_prev_release
        self.sequence = 4
        command = 'mkdir -p %s' % self.project_info['target_releases']
        result = waller.run(command, wenv=self.config())
        result = waller.put(self.local_codebase + self.release_version_tar, remote=self.project_info['target_releases'], wenv=self.config())
        current_app.logger.info('command: %s', dir(result))
        self.release_untar(waller)
        self.prev_release_custom(waller)

    def prev_release_custom(self, waller):
        if False:
            for i in range(10):
                print('nop')
        commands = self.project_info['prev_release']
        if commands:
            for command in commands.split('\n'):
                if command.strip().startswith('#') or not command.strip():
                    continue
                target_release_version = '%s/%s' % (self.project_info['target_releases'], self.release_version)
                with waller.cd(target_release_version):
                    result = waller.run(command, wenv=self.config())

    def release(self, waller):
        if False:
            i = 10
            return i + 15
        '\n        5.部署代码到目标机器做的任务\n        - 打包代码 local\n        - scp local => remote\n        - 解压 remote\n        :return:\n        '
        self.stage = self.stage_release
        self.sequence = 5
        with waller.cd(self.project_info['target_releases']):
            command = '[ -L %s ] && readlink %s || echo ""' % (self.project_info['target_root'], self.project_info['target_root'])
            result = waller.run(command, wenv=self.config(console=False))
            self.previous_release_version = os.path.basename(result.stdout).strip()
            current_link_tmp_dir = 'current-tmp-%s' % self.task_id
            command = 'ln -sfn {library}/{version} {library}/{current_tmp}'.format(library=self.project_info['target_releases'], version=self.release_version, current_tmp=current_link_tmp_dir)
            result = waller.run(command, wenv=self.config())
            current_link_tmp_dir = '%s/current-tmp-%s' % (self.project_info['target_releases'], self.task_id)
            command = 'mv -fT %s %s' % (current_link_tmp_dir, self.project_info['target_root'])
            result = waller.run(command, wenv=self.config())

    def rollback(self, waller):
        if False:
            for i in range(10):
                print('nop')
        '\n        5.部署代码到目标机器做的任务\n        - 恢复旧版本\n        :return:\n        '
        self.stage = self.stage_release
        self.sequence = 5
        with waller.cd(self.project_info['target_releases']):
            command = '[ -L %s ] && readlink %s || echo ""' % (self.project_info['target_root'], self.project_info['target_root'])
            result = waller.run(command, wenv=self.config(console=False))
            self.previous_release_version = os.path.basename(result.stdout)
            current_link_tmp_dir = '%s/current-tmp-%s' % (self.project_info['target_releases'], self.task_id)
            command = 'ln -sfn %s/%s %s' % (self.project_info['target_releases'], self.release_version, current_link_tmp_dir)
            result = waller.run(command, wenv=self.config())
            current_link_tmp_dir = '%s/current-tmp-%s' % (self.project_info['target_releases'], self.task_id)
            command = 'mv -fT %s %s' % (current_link_tmp_dir, self.project_info['target_root'])
            result = waller.run(command, wenv=self.config())

    def release_untar(self, waller):
        if False:
            while True:
                i = 10
        '\n        解压版本包\n        :return:\n        '
        with waller.cd(self.project_info['target_releases']):
            command = 'tar zxf %s' % self.release_version_tar
            result = waller.run(command, wenv=self.config())

    def post_release(self, waller):
        if False:
            return 10
        '\n        6.部署代码到目标机器后要做的任务\n        - 切换软链\n        - 重启 nginx\n        :return:\n        '
        self.stage = self.stage_post_release
        self.sequence = 6
        commands = self.project_info['post_release']
        if commands:
            for command in commands.split('\n'):
                if command.strip().startswith('#') or not command.strip():
                    continue
                with waller.cd(self.project_info['target_root']):
                    pty = False if command.find('nohup') >= 0 else True
                    result = waller.run(command, wenv=self.config(), pty=pty)
        self.cleanup_remote(waller)

    def post_release_service(self, waller):
        if False:
            return 10
        '\n        代码部署完成后,服务启动工作,如: nginx重启\n        :param connection:\n        :return:\n        '
        with waller.cd(self.project_info['target_root']):
            command = 'sudo service nginx restart'
            result = waller.run(command, wenv=self.config())

    def project_detection(self):
        if False:
            return 10
        errors = []
        for server_info in self.servers:
            waller = Waller(host=server_info['host'], user=server_info['user'], port=server_info['port'])
            result = waller.run('id', exception=False, wenv=self.config())
            if result.failed:
                errors.append({'title': '远程目标机器免密码登录失败', 'why': '远程目标机器：%s 错误：%s' % (server_info['host'], result.stdout), 'how': '在宿主机中配置免密码登录，把宿主机用户%s的~/.ssh/id_rsa.pub添加到远程目标机器用户%s的~/.ssh/authorized_keys。了解更多：http://walle-web.io/docs/troubleshooting.html' % (pwd.getpwuid(os.getuid())[0], server_info['host'])})
            command = '[ -d {webroot} ] || mkdir -p {webroot}'.format(webroot=os.path.basename(self.project_info['target_root']))
            result = waller.run(command, exception=False, wenv=self.config(console=False))
            command = '[ -L "%s" ] && echo "true" || echo "false"' % self.project_info['target_root']
            result = waller.run(command, exception=False, wenv=self.config())
            if result.stdout == 'false':
                errors.append({'title': '远程目标机器webroot不能是已建好的目录', 'why': '远程目标机器%s webroot不能是已存在的目录，必须为软链接，你不必新建，walle会自行创建。' % server_info['host'], 'how': '手工删除远程目标机器：%s webroot目录：%s' % (server_info['host'], self.project_info['target_root'])})
        return errors

    def list_tag(self):
        if False:
            for i in range(10):
                print('nop')
        repo = Repo(self.dir_codebase_project)
        repo.init(url=self.project_info['repo_url'])
        return repo.tags()

    def list_branch(self):
        if False:
            i = 10
            return i + 15
        repo = Repo(self.dir_codebase_project)
        repo.init(url=self.project_info['repo_url'])
        return repo.branches()

    def list_commit(self, branch):
        if False:
            return 10
        repo = Repo(self.dir_codebase_project)
        repo.init(url=self.project_info['repo_url'])
        return repo.commits(branch)

    def init_repo(self):
        if False:
            for i in range(10):
                print('nop')
        repo = Repo(self.dir_codebase_project)
        repo.init(url=self.project_info['repo_url'])

    def cleanup_local(self):
        if False:
            while True:
                i = 10
        command = 'rm -rf {project_id}_{task_id}_*'.format(project_id=self.project_info['id'], task_id=self.task_id)
        with self.localhost.cd(self.local_codebase):
            result = self.localhost.local(command, wenv=self.config())

    def cleanup_remote(self, waller):
        if False:
            i = 10
            return i + 15
        command = 'rm -rf {project_id}_{task_id}_*.tgz'.format(project_id=self.project_info['id'], task_id=self.task_id)
        with waller.cd(self.project_info['target_releases']):
            result = waller.run(command, wenv=self.config())
        command = 'ls -t | grep ^{project_id}_ | tail -n +{keep_version_num} | xargs rm -rf'.format(project_id=self.project_info['id'], keep_version_num=int(self.project_info['keep_version_num']) + 1)
        with waller.cd(self.project_info['target_releases']):
            result = waller.run(command, wenv=self.config())

    def logs(self):
        if False:
            while True:
                i = 10
        return RecordModel().fetch(task_id=self.task_id)

    def end(self, success=True, update_status=True):
        if False:
            i = 10
            return i + 15
        if update_status:
            status = TaskModel.status_success if success else TaskModel.status_fail
            current_app.logger.info('success:%s, status:%s' % (success, status))
            TaskModel().get_by_id(self.task_id).update({'status': status, 'link_id': self.release_version, 'ex_link_id': self.previous_release_version})
            notice_info = {'title': '', 'username': current_user.username, 'project_name': self.project_info['name'], 'task_name': '%s ([%s](%s))' % (self.taskMdl.get('name'), self.task_id, Notice.task_url(project_name=self.project_info['name'], task_id=self.task_id)), 'branch': self.taskMdl.get('branch'), 'commit': self.taskMdl.get('commit_id'), 'tag': self.taskMdl.get('tag'), 'repo_mode': self.project_info['repo_mode']}
            notice = Notice.create(self.project_info['notice_type'])
            if success:
                notice_info['title'] = '上线部署成功'
                notice.deploy_task(project_info=self.project_info, notice_info=notice_info)
            else:
                notice_info['title'] = '上线部署失败'
                notice.deploy_task(project_info=self.project_info, notice_info=notice_info)
        self.cleanup_local()
        if success:
            emit('success', {'event': 'finish', 'data': {'message': '部署完成，辛苦了，为你的努力喝彩！'}}, room=self.task_id)
        else:
            emit('fail', {'event': 'finish', 'data': {'message': Code.code_msg[Code.deploy_fail]}}, room=self.task_id)

    def walle_deploy(self):
        if False:
            while True:
                i = 10
        self.start()
        try:
            self.prev_deploy()
            self.deploy()
            self.post_deploy()
            is_all_servers_success = True
            for server_info in self.servers:
                host = server_info['host']
                try:
                    waller = Waller(host=host, user=server_info['user'], port=server_info['port'], inline_ssh_env=True)
                    waller.init_env(env=self.custom_global_env)
                    self.connections[self.task_id] = waller
                    self.prev_release(self.connections[self.task_id])
                    self.release(self.connections[self.task_id])
                    self.post_release(self.connections[self.task_id])
                    RecordModel().save_record(stage=RecordModel.stage_end, sequence=0, user_id=current_user.id, task_id=self.task_id, status=RecordModel.status_success, host=host, user=server_info['user'], command='')
                    emit('success', {'event': 'finish', 'data': {'host': host, 'message': host + ' 部署完成！'}}, room=self.task_id)
                except Exception as e:
                    is_all_servers_success = False
                    current_app.logger.exception(e)
                    self.errors[host] = e.message
                    RecordModel().save_record(stage=RecordModel.stage_end, sequence=0, user_id=current_user.id, task_id=self.task_id, status=RecordModel.status_fail, host=host, user=server_info['user'], command='')
                    emit('fail', {'event': 'finish', 'data': {'host': host, 'message': host + Code.code_msg[Code.deploy_fail]}}, room=self.task_id)
            self.end(is_all_servers_success)
        except Exception as e:
            self.end(False)
        return {'success': self.success, 'errors': self.errors}

    def walle_rollback(self):
        if False:
            i = 10
            return i + 15
        self.start()
        try:
            is_all_servers_success = True
            for server_info in self.servers:
                host = server_info['host']
                try:
                    waller = Waller(host=host, user=server_info['user'], port=server_info['port'], inline_ssh_env=True)
                    waller.init_env(env=self.custom_global_env)
                    self.connections[self.task_id] = waller
                    self.prev_release_custom(self.connections[self.task_id])
                    self.release(self.connections[self.task_id])
                    self.post_release(self.connections[self.task_id])
                    RecordModel().save_record(stage=RecordModel.stage_end, sequence=0, user_id=current_user.id, task_id=self.task_id, status=RecordModel.status_success, host=host, user=server_info['user'], command='')
                    emit('success', {'event': 'finish', 'data': {'host': host, 'message': host + ' 部署完成！'}}, room=self.task_id)
                except Exception as e:
                    is_all_servers_success = False
                    current_app.logger.exception(e)
                    self.errors[host] = e.message
                    RecordModel().save_record(stage=RecordModel.stage_end, sequence=0, user_id=current_user.id, task_id=self.task_id, status=RecordModel.status_fail, host=host, user=server_info['user'], command='')
                    emit('fail', {'event': 'finish', 'data': {'host': host, 'message': host + Code.code_msg[Code.deploy_fail]}}, room=self.task_id)
            self.end(is_all_servers_success)
        except Exception as e:
            self.end(False)
        return {'success': self.success, 'errors': self.errors}