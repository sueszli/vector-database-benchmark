"""init walle database

此刻是walle 2.0 alpha准备工作收尾阶段中, 但内心非常孤独, 大多用户让人心寒, 缺乏基本的感恩之心

Revision ID: 2bca06a823a0
Revises:
Create Date: 2018-12-08 21:01:19.273412

"""
from alembic import op
from walle.service.extensions import db
revision = '2bca06a823a0'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    create_environments()
    init_environments()
    create_menus()
    init_menus()
    create_projects()
    create_records()
    create_servers()
    create_spaces()
    init_spaces()
    create_tasks()
    create_users()
    init_users()
    create_members()
    init_members()
    db.session.commit()

def create_environments():
    if False:
        while True:
            i = 10
    sql = u"CREATE TABLE `environments` (\n              `id` int(10) NOT NULL AUTO_INCREMENT COMMENT '',\n              `name` varchar(100) DEFAULT 'master' COMMENT '',\n              `space_id` int(10) NOT NULL DEFAULT '0' COMMENT '',\n              `status` tinyint(1) DEFAULT '1' COMMENT '',\n              `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '',\n              `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '',\n              PRIMARY KEY (`id`)\n            ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='';"
    db.session.execute(sql)

def init_environments():
    if False:
        print('Hello World!')
    sql = u"INSERT INTO `environments` VALUES\n            (1,'开发环境', 1, 1, '2017-03-08 17:26:07', '2018-11-26 15:38:14'),\n            (2,'测试环境', 1, 1, '2017-05-13 11:26:42', '2018-11-26 15:38:14'),\n            (3,'生产环境', 1, 1, '2017-05-14 10:46:31', '2018-11-26 17:10:02');"
    db.session.execute(sql)

def create_menus():
    if False:
        i = 10
        return i + 15
    sql = u"CREATE TABLE `menus` (\n              `id` int(15) NOT NULL AUTO_INCREMENT,\n              `name_cn` varchar(30) NOT NULL COMMENT '',\n              `name_en` varchar(30) NOT NULL COMMENT '',\n              `pid` int(6) NOT NULL COMMENT '',\n              `type` enum('action','controller','module') DEFAULT 'action' COMMENT '',\n              `sequence` int(11) DEFAULT '0' COMMENT '',\n              `role` varchar(10) NOT NULL DEFAULT '' COMMENT '',\n              `archive` tinyint(1) DEFAULT '0' COMMENT '',\n              `icon` varchar(30) DEFAULT '' COMMENT '',\n              `url` varchar(100) DEFAULT '' COMMENT '',\n              `visible` tinyint(1) DEFAULT '1' COMMENT '',\n              `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '',\n              `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '',\n              PRIMARY KEY (`id`)\n            ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='';"
    db.session.execute(sql)

def init_menus():
    if False:
        for i in range(10):
            print('nop')
    sql = u"INSERT INTO `menus` VALUES\n            (1,'首页','index',0,'module',10000,'10',0,'wl-icon-main','/',1,'2017-06-11 23:11:38','2018-11-03 09:31:51'),\n            (2,'空间管理','space',0,'module',10001,'50',0,'wl-icon-space-set','/space/index',1,'2017-06-11 23:11:38','2018-11-01 07:37:23'),\n            (3,'用户管理','user',0,'module',10002,'40',0,'wl-icon-user-set','/user/index',1,'2017-06-11 23:11:52','2018-12-05 19:50:43'),\n            (4,'项目中心','project',0,'module',10003,'30',0,'wl-icon-project-set','',1,'2017-06-11 23:12:45','2018-12-05 19:45:43'),\n            (5,'部署管理','deploy',0,'module',10101,'10',0,'wl-icon-deploy-set','/deploy/index',1,'2017-06-11 23:13:51','2018-11-04 23:57:19'),\n            (6,'环境管理','group',4,'controller',10102,'50',0,'leaf','/environment/index',1,'2017-06-11 23:14:11','2018-11-03 09:31:41'),\n            (7,'服务器管理','role',4,'controller',10103,'40',0,'leaf','/server/index',1,'2017-06-11 23:14:44','2018-11-03 09:31:41'),\n            (8,'项目管理','environment',4,'controller',10201,'30',0,'leaf','/project/index',1,'2017-06-11 23:15:30','2018-12-05 19:45:12');"
    db.session.execute(sql)

def create_projects():
    if False:
        return 10
    sql = u"CREATE TABLE `projects` (\n              `id` int(10) NOT NULL AUTO_INCREMENT COMMENT '',\n              `user_id` int(10) NOT NULL COMMENT '',\n              `name` varchar(100) DEFAULT 'master' COMMENT '',\n              `environment_id` int(1) NOT NULL COMMENT '',\n              `space_id` int(10) NOT NULL DEFAULT '0' COMMENT '',\n              `status` tinyint(1) DEFAULT '1' COMMENT '',\n              `master` varchar(100) NOT NULL DEFAULT '' COMMENT '',\n              `version` varchar(40) DEFAULT '' COMMENT '',\n              `excludes` text COMMENT '',\n              `target_user` varchar(50) NOT NULL COMMENT '',\n              `target_port` int(3) NOT NULL DEFAULT '22' COMMENT '',\n              `target_root` varchar(200) NOT NULL COMMENT '',\n              `target_releases` varchar(200) NOT NULL COMMENT '',\n              `server_ids` text COMMENT '',\n              `task_vars` text COMMENT '',\n              `prev_deploy` text COMMENT '',\n              `post_deploy` text COMMENT '',\n              `prev_release` text COMMENT '',\n              `post_release` text COMMENT '',\n              `keep_version_num` int(3) NOT NULL DEFAULT '20' COMMENT '',\n              `repo_url` varchar(200) DEFAULT '' COMMENT '',\n              `repo_username` varchar(50) DEFAULT '' COMMENT '',\n              `repo_password` varchar(50) DEFAULT '' COMMENT '',\n              `repo_mode` varchar(50) DEFAULT 'branch' COMMENT '',\n              `repo_type` varchar(10) DEFAULT 'git' COMMENT '',\n              `notice_type` varchar(10) NOT NULL DEFAULT '' COMMENT '',\n              `notice_hook` text NOT NULL COMMENT '',\n              `task_audit` tinyint(1) DEFAULT '0' COMMENT '',\n              `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '',\n              `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '',\n              PRIMARY KEY (`id`)\n            ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='';"
    db.session.execute(sql)

def create_records():
    if False:
        return 10
    sql = u"CREATE TABLE `records` (\n              `id` bigint(10) NOT NULL AUTO_INCREMENT COMMENT '',\n              `stage` varchar(20) DEFAULT NULL COMMENT '',\n              `sequence` int(10) DEFAULT NULL COMMENT '',\n              `user_id` int(21) unsigned NOT NULL COMMENT '',\n              `task_id` bigint(11) NOT NULL COMMENT '',\n              `status` int(3) NOT NULL COMMENT '',\n              `host` varchar(200) DEFAULT '' COMMENT '',\n              `user` varchar(200) DEFAULT '' COMMENT '',\n              `command` text COMMENT '',\n              `success` LONGTEXT COMMENT '',\n              `error` LONGTEXT COMMENT '',\n              `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '',\n              `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '',\n              PRIMARY KEY (`id`)\n            ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='';"
    db.session.execute(sql)

def create_servers():
    if False:
        return 10
    sql = u"CREATE TABLE `servers` (\n              `id` int(10) NOT NULL AUTO_INCREMENT COMMENT '',\n              `name` varchar(100) DEFAULT '' COMMENT '',\n              `host` varchar(100) NOT NULL COMMENT '',\n              `port` int(1) DEFAULT '22' COMMENT '',\n              `status` tinyint(1) DEFAULT '1' COMMENT '',\n              `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '',\n              `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '',\n              PRIMARY KEY (`id`)\n            ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='';"
    db.session.execute(sql)

def create_spaces():
    if False:
        return 10
    sql = u"CREATE TABLE `spaces` (\n              `id` int(10) NOT NULL AUTO_INCREMENT COMMENT '',\n              `user_id` int(10) NOT NULL COMMENT '',\n              `name` varchar(100) NOT NULL COMMENT '',\n              `status` tinyint(1) DEFAULT '1' COMMENT '',\n              `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '',\n              `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '',\n              PRIMARY KEY (`id`)\n            ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='';"
    db.session.execute(sql)

def init_spaces():
    if False:
        return 10
    sql = u"INSERT INTO `spaces` VALUES\n            (1,2,'Demo空间',1,'2018-09-17 22:09:37','2018-11-18 00:09:58');"
    db.session.execute(sql)

def create_tasks():
    if False:
        i = 10
        return i + 15
    sql = u"CREATE TABLE `tasks` (\n              `id` int(10) NOT NULL AUTO_INCREMENT COMMENT '',\n              `name` varchar(100) NOT NULL COMMENT '',\n              `user_id` bigint(21) unsigned NOT NULL COMMENT '',\n              `project_id` int(11) NOT NULL COMMENT '',\n              `action` int(1) DEFAULT '0' COMMENT '',\n              `status` tinyint(1) NOT NULL COMMENT '',\n              `link_id` varchar(100) DEFAULT '' COMMENT '',\n              `ex_link_id` varchar(100) DEFAULT '' COMMENT '',\n              `servers` text COMMENT '',\n              `commit_id` varchar(40) DEFAULT '' COMMENT '',\n              `branch` varchar(100) DEFAULT 'master' COMMENT '',\n              `tag` varchar(100) DEFAULT '' COMMENT '',\n              `file_transmission_mode` smallint(3) NOT NULL DEFAULT '1' COMMENT '',\n              `file_list` LONGTEXT COMMENT '',\n              `enable_rollback` int(1) NOT NULL DEFAULT '1' COMMENT '',\n              `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '',\n              `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '',\n              PRIMARY KEY (`id`)\n            ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='';"
    db.session.execute(sql)

def create_users():
    if False:
        while True:
            i = 10
    sql = u"CREATE TABLE `users` (\n              `id` int(11) NOT NULL AUTO_INCREMENT,\n              `username` varchar(50) NOT NULL COMMENT '',\n              `is_email_verified` tinyint(1) NOT NULL DEFAULT '0' COMMENT '',\n              `email` varchar(50) NOT NULL COMMENT '',\n              `password` varchar(100) NOT NULL COMMENT '',\n              `password_hash` varchar(50) DEFAULT NULL COMMENT '',\n              `avatar` varchar(100) DEFAULT 'default.jpg' COMMENT '',\n              `role` varchar(10) NOT NULL DEFAULT '' COMMENT '',\n              `status` tinyint(1) NOT NULL DEFAULT '1' COMMENT '',\n              `last_space` int(11) NOT NULL DEFAULT '0',\n              `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '',\n              `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '',\n              PRIMARY KEY (`id`)\n            ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='';"
    db.session.execute(sql)

def init_users():
    if False:
        while True:
            i = 10
    sql = u"INSERT INTO `users` VALUES\n            (1,'Super',1,'super@walle-web.io','pbkdf2:sha256:50000$AyRSJVSn$448c69b93158b30b9e3625d340b48dbdbce1186fcf30fc72663a9361ffec339b','','','SUPER',1,0,'2017-03-17 09:03:09','2018-11-24 17:01:23'),\n            (2,'Owner',1,'owner@walle-web.io','pbkdf2:sha256:50000$AyRSJVSn$448c69b93158b30b9e3625d340b48dbdbce1186fcf30fc72663a9361ffec339b','','','',1,1,'2017-03-20 19:05:44','2018-11-24 17:01:23'),\n            (3,'Master',1,'master@walle-web.io','pbkdf2:sha256:50000$AyRSJVSn$448c69b93158b30b9e3625d340b48dbdbce1186fcf30fc72663a9361ffec339b','','','',1,1,'2017-04-13 15:03:57','2018-11-24 10:22:37'),\n            (4,'Developer',1,'developer@walle-web.io','pbkdf2:sha256:50000$AyRSJVSn$448c69b93158b30b9e3625d340b48dbdbce1186fcf30fc72663a9361ffec339b','','','',1,1,'2017-05-11 22:33:35','2018-12-05 19:37:47'),\n            (5,'Reporter',1,'reporter@walle-web.io','pbkdf2:sha256:50000$AyRSJVSn$448c69b93158b30b9e3625d340b48dbdbce1186fcf30fc72663a9361ffec339b','','','',1,1,'2017-05-11 23:39:11','2018-11-23 07:40:55')"
    db.session.execute(sql)

def create_members():
    if False:
        return 10
    sql = u"CREATE TABLE `members` (\n              `id` int(10) NOT NULL AUTO_INCREMENT COMMENT '',\n              `user_id` int(10) DEFAULT '0' COMMENT '',\n              `source_id` int(10) DEFAULT '0' COMMENT '',\n              `source_type` varchar(10) DEFAULT '' COMMENT '',\n              `access_level` varchar(10) DEFAULT '10' COMMENT '',\n              `status` tinyint(1) DEFAULT '1' COMMENT '',\n              `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '',\n              `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '',\n              PRIMARY KEY (`id`)\n            ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='';"
    db.session.execute(sql)

def init_members():
    if False:
        print('Hello World!')
    sql = u"INSERT INTO `members` VALUES\n            (null,2,1,'group','OWNER',1,'2018-12-09 00:35:59','2018-12-09 00:35:59'),\n            (null,3,1,'group','MASTER',1,'2018-12-09 00:35:59','2018-12-09 00:35:59'),\n            (null,4,1,'group','DEVELOPER',1,'2018-12-09 00:35:59','2018-12-09 00:35:59'),\n            (null,5,1,'group','REPORTER',1,'2018-12-09 00:35:59','2018-12-09 00:35:59');"
    db.session.execute(sql)

def downgrade():
    if False:
        return 10
    op.drop_table('members')
    op.drop_table('users')
    op.drop_table('tasks')
    op.drop_table('spaces')
    op.drop_table('servers')
    op.drop_table('records')
    op.drop_table('projects')
    op.drop_table('menus')
    op.drop_table('environments')