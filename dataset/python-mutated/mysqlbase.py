import time
import mysql.connector

class MySQLMixin(object):
    maxlimit = 18446744073709551615

    @property
    def dbcur(self):
        if False:
            print('Hello World!')
        try:
            if self.conn.unread_result:
                self.conn.get_rows()
                if hasattr(self.conn, 'free_result'):
                    self.conn.free_result()
            return self.conn.cursor()
        except (mysql.connector.OperationalError, mysql.connector.InterfaceError):
            self.conn.ping(reconnect=True)
            self.conn.database = self.database_name
            return self.conn.cursor()

class SplitTableMixin(object):
    UPDATE_PROJECTS_TIME = 10 * 60

    def _tablename(self, project):
        if False:
            while True:
                i = 10
        if self.__tablename__:
            return '%s_%s' % (self.__tablename__, project)
        else:
            return project

    @property
    def projects(self):
        if False:
            while True:
                i = 10
        if time.time() - getattr(self, '_last_update_projects', 0) > self.UPDATE_PROJECTS_TIME:
            self._list_project()
        return self._projects

    @projects.setter
    def projects(self, value):
        if False:
            while True:
                i = 10
        self._projects = value

    def _list_project(self):
        if False:
            while True:
                i = 10
        self._last_update_projects = time.time()
        self.projects = set()
        if self.__tablename__:
            prefix = '%s_' % self.__tablename__
        else:
            prefix = ''
        for (project,) in self._execute('show tables;'):
            if project.startswith(prefix):
                project = project[len(prefix):]
                self.projects.add(project)

    def drop(self, project):
        if False:
            while True:
                i = 10
        if project not in self.projects:
            self._list_project()
        if project not in self.projects:
            return
        tablename = self._tablename(project)
        self._execute('DROP TABLE %s' % self.escape(tablename))
        self._list_project()