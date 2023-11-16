"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from persepolis.scripts.useful_tools import determineConfigFolder
from persepolis.scripts import logger
from time import sleep
import traceback
import sqlite3
import random
import ast
import os
config_folder = determineConfigFolder()
persepolis_tmp = os.path.join(config_folder, 'persepolis_tmp')

class TempDB:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_db_connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.temp_db_cursor = self.temp_db_connection.cursor()
        self.lock = False

    def lockCursor(self):
        if False:
            return 10
        while self.lock:
            rand_float = random.uniform(0, 0.5)
            sleep(rand_float)
        self.lock = True

    def createTables(self):
        if False:
            return 10
        self.lockCursor()
        self.temp_db_cursor.execute('CREATE TABLE IF NOT EXISTS single_db_table(\n                                                                                ID INTEGER,\n                                                                                gid TEXT PRIMARY KEY,\n                                                                                status TEXT,\n                                                                                shutdown TEXT\n                                                                                )')
        self.temp_db_cursor.execute('CREATE TABLE IF NOT EXISTS queue_db_table(\n                                                                                ID INTEGER,\n                                                                                category TEXT PRIMARY KEY,\n                                                                                shutdown TEXT\n                                                                                )')
        self.temp_db_connection.commit()
        self.lock = False

    def insertInSingleTable(self, gid):
        if False:
            i = 10
            return i + 15
        self.lockCursor()
        self.temp_db_cursor.execute("INSERT INTO single_db_table VALUES(\n                                                                NULL,\n                                                                '{}',\n                                                                'active',\n                                                                NULL)".format(gid))
        self.temp_db_connection.commit()
        self.lock = False

    def insertInQueueTable(self, category):
        if False:
            return 10
        self.lockCursor()
        self.temp_db_cursor.execute("INSERT INTO queue_db_table VALUES(\n                                                                NULL,\n                                                                '{}',\n                                                                NULL)".format(category))
        self.temp_db_connection.commit()
        self.lock = False

    def updateSingleTable(self, dict):
        if False:
            i = 10
            return i + 15
        self.lockCursor()
        keys_list = ['gid', 'shutdown', 'status']
        for key in keys_list:
            if key not in dict.keys():
                dict[key] = None
        self.temp_db_cursor.execute('UPDATE single_db_table SET shutdown = coalesce(:shutdown, shutdown),\n                                                                status = coalesce(:status, status)\n                                                                WHERE gid = :gid', dict)
        self.temp_db_connection.commit()
        self.lock = False

    def updateQueueTable(self, dict):
        if False:
            for i in range(10):
                print('nop')
        self.lockCursor()
        keys_list = ['category', 'shutdown']
        for key in keys_list:
            if key not in dict.keys():
                dict[key] = None
        self.temp_db_cursor.execute('UPDATE queue_db_table SET shutdown = coalesce(:shutdown, shutdown)\n                                                                WHERE category = :category', dict)
        self.temp_db_connection.commit()
        self.lock = False

    def returnActiveGids(self):
        if False:
            for i in range(10):
                print('nop')
        self.lockCursor()
        self.temp_db_cursor.execute("SELECT gid FROM single_db_table WHERE status = 'active'")
        list = self.temp_db_cursor.fetchall()
        self.lock = False
        gid_list = []
        for tuple in list:
            gid = tuple[0]
            gid_list.append(gid)
        return gid_list

    def returnGid(self, gid):
        if False:
            return 10
        self.lockCursor()
        self.temp_db_cursor.execute("SELECT shutdown, status FROM single_db_table WHERE gid = '{}'".format(gid))
        list = self.temp_db_cursor.fetchall()
        self.lock = False
        tuple = list[0]
        dict = {'shutdown': str(tuple[0]), 'status': tuple[1]}
        return dict

    def returnCategory(self, category):
        if False:
            for i in range(10):
                print('nop')
        self.lockCursor()
        self.temp_db_cursor.execute("SELECT shutdown FROM queue_db_table WHERE category = '{}'".format(category))
        list = self.temp_db_cursor.fetchall()
        self.lock = False
        tuple = list[0]
        dict = {'shutdown': tuple[0]}
        return dict

    def resetDataBase(self):
        if False:
            return 10
        self.lockCursor()
        self.temp_db_cursor.execute('DELETE FROM single_db_table')
        self.temp_db_cursor.execute('DELETE FROM queue_db_table')
        self.lock = False

    def closeConnections(self):
        if False:
            while True:
                i = 10
        self.lockCursor()
        self.temp_db_cursor.close()
        self.temp_db_connection.close()
        self.lock = False

class PluginsDB:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        plugins_db_path = os.path.join(persepolis_tmp, 'plugins.db')
        self.plugins_db_connection = sqlite3.connect(plugins_db_path, check_same_thread=False)
        self.plugins_db_cursor = self.plugins_db_connection.cursor()
        self.lock = False

    def lockCursor(self):
        if False:
            for i in range(10):
                print('nop')
        while self.lock:
            rand_float = random.uniform(0, 0.5)
            sleep(rand_float)
        self.lock = True

    def createTables(self):
        if False:
            while True:
                i = 10
        self.lockCursor()
        self.plugins_db_cursor.execute('CREATE TABLE IF NOT EXISTS plugins_db_table(\n                                                                                ID INTEGER PRIMARY KEY,\n                                                                                link TEXT,\n                                                                                referer TEXT,\n                                                                                load_cookies TEXT,\n                                                                                user_agent TEXT,\n                                                                                header TEXT,\n                                                                                out TEXT,\n                                                                                status TEXT\n                                                                                )')
        self.plugins_db_connection.commit()
        self.lock = False

    def insertInPluginsTable(self, list):
        if False:
            for i in range(10):
                print('nop')
        self.lockCursor()
        for dict in list:
            self.plugins_db_cursor.execute("INSERT INTO plugins_db_table VALUES(\n                                                                        NULL,\n                                                                        :link,\n                                                                        :referer,\n                                                                        :load_cookies,\n                                                                        :user_agent,\n                                                                        :header,\n                                                                        :out,\n                                                                        'new'\n                                                                            )", dict)
        self.plugins_db_connection.commit()
        self.lock = False

    def returnNewLinks(self):
        if False:
            return 10
        self.lockCursor()
        self.plugins_db_cursor.execute("SELECT link, referer, load_cookies, user_agent, header, out\n                                            FROM plugins_db_table\n                                            WHERE status = 'new'")
        list = self.plugins_db_cursor.fetchall()
        self.plugins_db_cursor.execute("UPDATE plugins_db_table SET status = 'old'\n                                            WHERE status = 'new'")
        self.plugins_db_connection.commit()
        self.lock = False
        new_list = []
        for tuple in list:
            dict = {'link': tuple[0], 'referer': tuple[1], 'load_cookies': tuple[2], 'user_agent': tuple[3], 'header': tuple[4], 'out': tuple[5]}
            new_list.append(dict)
        return new_list

    def deleteOldLinks(self):
        if False:
            i = 10
            return i + 15
        self.lockCursor()
        self.plugins_db_cursor.execute("DELETE FROM plugins_db_table WHERE status = 'old'")
        self.plugins_db_connection.commit()
        self.lock = False

    def closeConnections(self):
        if False:
            for i in range(10):
                print('nop')
        self.lockCursor()
        self.plugins_db_cursor.close()
        self.plugins_db_connection.close()
        self.lock = False

class PersepolisDB:

    def __init__(self):
        if False:
            while True:
                i = 10
        persepolis_db_path = os.path.join(config_folder, 'persepolis.db')
        self.persepolis_db_connection = sqlite3.connect(persepolis_db_path, check_same_thread=False)
        self.persepolis_db_connection.execute('pragma foreign_keys=ON')
        self.persepolis_db_cursor = self.persepolis_db_connection.cursor()
        self.lock = False

    def lockCursor(self):
        if False:
            while True:
                i = 10
        while self.lock:
            rand_float = random.uniform(0, 0.5)
            sleep(rand_float)
        self.lock = True

    def createTables(self):
        if False:
            i = 10
            return i + 15
        self.lockCursor()
        self.persepolis_db_cursor.execute('CREATE TABLE IF NOT EXISTS category_db_table(\n                                                                category TEXT PRIMARY KEY,\n                                                                start_time_enable TEXT,\n                                                                start_time TEXT,\n                                                                end_time_enable TEXT,\n                                                                end_time TEXT,\n                                                                reverse TEXT,\n                                                                limit_enable TEXT,\n                                                                limit_value TEXT,\n                                                                after_download TEXT,\n                                                                gid_list TEXT\n                                                                            )')
        self.persepolis_db_cursor.execute('CREATE TABLE IF NOT EXISTS download_db_table(\n                                                                                    file_name TEXT,\n                                                                                    status TEXT,\n                                                                                    size TEXT,\n                                                                                    downloaded_size TEXT,\n                                                                                    percent TEXT,\n                                                                                    connections TEXT,\n                                                                                    rate TEXT,\n                                                                                    estimate_time_left TEXT,\n                                                                                    gid TEXT PRIMARY KEY,\n                                                                                    link TEXT,\n                                                                                    first_try_date TEXT,\n                                                                                    last_try_date TEXT,\n                                                                                    category TEXT,\n                                                                                    FOREIGN KEY(category) REFERENCES category_db_table(category)\n                                                                                    ON UPDATE CASCADE\n                                                                                    ON DELETE CASCADE\n                                                                                         )')
        self.persepolis_db_cursor.execute('CREATE TABLE IF NOT EXISTS addlink_db_table(\n                                                                                ID INTEGER PRIMARY KEY,\n                                                                                gid TEXT,\n                                                                                out TEXT,\n                                                                                start_time TEXT,\n                                                                                end_time TEXT,\n                                                                                link TEXT,\n                                                                                ip TEXT,\n                                                                                port TEXT,\n                                                                                proxy_user TEXT,\n                                                                                proxy_passwd TEXT,\n                                                                                download_user TEXT,\n                                                                                download_passwd TEXT,\n                                                                                connections TEXT,\n                                                                                limit_value TEXT,\n                                                                                download_path TEXT,\n                                                                                referer TEXT,\n                                                                                load_cookies TEXT,\n                                                                                user_agent TEXT,\n                                                                                header TEXT,\n                                                                                after_download TEXT,\n                                                                                FOREIGN KEY(gid) REFERENCES download_db_table(gid) \n                                                                                ON UPDATE CASCADE \n                                                                                ON DELETE CASCADE \n                                                                                    )')
        self.persepolis_db_cursor.execute('CREATE TABLE IF NOT EXISTS video_finder_db_table(\n                                                                                ID INTEGER PRIMARY KEY,\n                                                                                video_gid TEXT,\n                                                                                audio_gid TEXT,\n                                                                                video_completed TEXT,\n                                                                                audio_completed TEXT,\n                                                                                muxing_status TEXT,\n                                                                                checking TEXT,\n                                                                                download_path TEXT,\n                                                                                FOREIGN KEY(video_gid) REFERENCES download_db_table(gid)\n                                                                                ON DELETE CASCADE,\n                                                                                FOREIGN KEY(audio_gid) REFERENCES download_db_table(gid)\n                                                                                ON DELETE CASCADE\n                                                                                    )')
        self.persepolis_db_connection.commit()
        self.lock = False
        answer = self.searchCategoryInCategoryTable('All Downloads')
        if not answer:
            all_downloads_dict = {'category': 'All Downloads', 'start_time_enable': 'no', 'start_time': '0:0', 'end_time_enable': 'no', 'end_time': '0:0', 'reverse': 'no', 'limit_enable': 'no', 'limit_value': '0K', 'after_download': 'no', 'gid_list': '[]'}
            single_downloads_dict = {'category': 'Single Downloads', 'start_time_enable': 'no', 'start_time': '0:0', 'end_time_enable': 'no', 'end_time': '0:0', 'reverse': 'no', 'limit_enable': 'no', 'limit_value': '0K', 'after_download': 'no', 'gid_list': '[]'}
            self.insertInCategoryTable(all_downloads_dict)
            self.insertInCategoryTable(single_downloads_dict)
        answer = self.searchCategoryInCategoryTable('Scheduled Downloads')
        if not answer:
            scheduled_downloads_dict = {'category': 'Scheduled Downloads', 'start_time_enable': 'no', 'start_time': '0:0', 'end_time_enable': 'no', 'end_time': '0:0', 'reverse': 'no', 'limit_enable': 'no', 'limit_value': '0K', 'after_download': 'no', 'gid_list': '[]'}
            self.insertInCategoryTable(scheduled_downloads_dict)

    def insertInCategoryTable(self, dict):
        if False:
            return 10
        self.lockCursor()
        self.persepolis_db_cursor.execute('INSERT INTO category_db_table VALUES(\n                                                                            :category,\n                                                                            :start_time_enable,\n                                                                            :start_time,\n                                                                            :end_time_enable,\n                                                                            :end_time,\n                                                                            :reverse,\n                                                                            :limit_enable,\n                                                                            :limit_value,\n                                                                            :after_download,\n                                                                            :gid_list\n                                                                            )', dict)
        self.persepolis_db_connection.commit()
        self.lock = False

    def insertInDownloadTable(self, list):
        if False:
            while True:
                i = 10
        self.lockCursor()
        for dict in list:
            self.persepolis_db_cursor.execute('INSERT INTO download_db_table VALUES(\n                                                                            :file_name,\n                                                                            :status,\n                                                                            :size,\n                                                                            :downloaded_size,\n                                                                            :percent,\n                                                                            :connections,\n                                                                            :rate,\n                                                                            :estimate_time_left,\n                                                                            :gid,\n                                                                            :link,\n                                                                            :first_try_date,\n                                                                            :last_try_date,\n                                                                            :category\n                                                                            )', dict)
        self.persepolis_db_connection.commit()
        self.lock = False
        if len(list) != 0:
            category = dict['category']
            category_dict = self.searchCategoryInCategoryTable(category)
            all_downloads_dict = self.searchCategoryInCategoryTable('All Downloads')
            category_gid_list = category_dict['gid_list']
            all_downloads_gid_list = all_downloads_dict['gid_list']
            for dict in list:
                gid = dict['gid']
                category_gid_list.append(gid)
                all_downloads_gid_list.append(gid)
            self.updateCategoryTable([all_downloads_dict])
            self.updateCategoryTable([category_dict])

    def insertInAddLinkTable(self, list):
        if False:
            print('Hello World!')
        self.lockCursor()
        for dict in list:
            self.persepolis_db_cursor.execute('INSERT INTO addlink_db_table VALUES(NULL,\n                                                                                :gid,\n                                                                                :out,\n                                                                                :start_time,\n                                                                                :end_time,\n                                                                                :link,\n                                                                                :ip,\n                                                                                :port,\n                                                                                :proxy_user,\n                                                                                :proxy_passwd,\n                                                                                :download_user,\n                                                                                :download_passwd,\n                                                                                :connections,\n                                                                                :limit_value,\n                                                                                :download_path,\n                                                                                :referer,\n                                                                                :load_cookies,\n                                                                                :user_agent,\n                                                                                :header,\n                                                                                NULL\n                                                                                )', dict)
        self.persepolis_db_connection.commit()
        self.lock = False

    def insertInVideoFinderTable(self, list):
        if False:
            for i in range(10):
                print('nop')
        self.lockCursor()
        for dictionary in list:
            self.persepolis_db_cursor.execute('INSERT INTO video_finder_db_table VALUES(NULL,\n                                                                                :video_gid,\n                                                                                :audio_gid,\n                                                                                :video_completed,\n                                                                                :audio_completed,\n                                                                                :muxing_status,\n                                                                                :checking,\n                                                                                :download_path\n                                                                                )', dictionary)
        self.persepolis_db_connection.commit()
        self.lock = False

    def searchGidInVideoFinderTable(self, gid):
        if False:
            print('Hello World!')
        self.lockCursor()
        self.persepolis_db_cursor.execute("SELECT * FROM video_finder_db_table WHERE audio_gid = '{}' OR video_gid = '{}'".format(str(gid), str(gid)))
        result_list = self.persepolis_db_cursor.fetchall()
        self.lock = False
        if result_list:
            tuple = result_list[0]
        else:
            return None
        dictionary = {'video_gid': tuple[1], 'audio_gid': tuple[2], 'video_completed': tuple[3], 'audio_completed': tuple[4], 'muxing_status': tuple[5], 'checking': tuple[6], 'download_path': tuple[7]}
        return dictionary

    def searchGidInDownloadTable(self, gid):
        if False:
            while True:
                i = 10
        self.lockCursor()
        self.persepolis_db_cursor.execute("SELECT * FROM download_db_table WHERE gid = '{}'".format(str(gid)))
        list = self.persepolis_db_cursor.fetchall()
        self.lock = False
        if list:
            tuple = list[0]
        else:
            return None
        dict = {'file_name': tuple[0], 'status': tuple[1], 'size': tuple[2], 'downloaded_size': tuple[3], 'percent': tuple[4], 'connections': tuple[5], 'rate': tuple[6], 'estimate_time_left': tuple[7], 'gid': tuple[8], 'link': tuple[9], 'first_try_date': tuple[10], 'last_try_date': tuple[11], 'category': tuple[12]}
        return dict

    def returnItemsInDownloadTable(self, category=None):
        if False:
            for i in range(10):
                print('nop')
        self.lockCursor()
        if category:
            self.persepolis_db_cursor.execute("SELECT * FROM download_db_table WHERE category = '{}'".format(category))
        else:
            self.persepolis_db_cursor.execute('SELECT * FROM download_db_table')
        rows = self.persepolis_db_cursor.fetchall()
        self.lock = False
        downloads_dict = {}
        for tuple in rows:
            dict = {'file_name': tuple[0], 'status': tuple[1], 'size': tuple[2], 'downloaded_size': tuple[3], 'percent': tuple[4], 'connections': tuple[5], 'rate': tuple[6], 'estimate_time_left': tuple[7], 'gid': tuple[8], 'link': tuple[9], 'first_try_date': tuple[10], 'last_try_date': tuple[11], 'category': tuple[12]}
            downloads_dict[tuple[8]] = dict
        return downloads_dict

    def searchLinkInAddLinkTable(self, link):
        if False:
            while True:
                i = 10
        self.lockCursor()
        self.persepolis_db_cursor.execute('SELECT * FROM addlink_db_table WHERE link = (?)', (link,))
        list = self.persepolis_db_cursor.fetchall()
        self.lock = False
        if list:
            return True
        else:
            return False

    def searchGidInAddLinkTable(self, gid):
        if False:
            return 10
        self.lockCursor()
        self.persepolis_db_cursor.execute("SELECT * FROM addlink_db_table WHERE gid = '{}'".format(str(gid)))
        list = self.persepolis_db_cursor.fetchall()
        self.lock = False
        if list:
            tuple = list[0]
        else:
            return None
        dict = {'gid': tuple[1], 'out': tuple[2], 'start_time': tuple[3], 'end_time': tuple[4], 'link': tuple[5], 'ip': tuple[6], 'port': tuple[7], 'proxy_user': tuple[8], 'proxy_passwd': tuple[9], 'download_user': tuple[10], 'download_passwd': tuple[11], 'connections': tuple[12], 'limit_value': tuple[13], 'download_path': tuple[14], 'referer': tuple[15], 'load_cookies': tuple[16], 'user_agent': tuple[17], 'header': tuple[18], 'after_download': tuple[19]}
        return dict

    def returnItemsInAddLinkTable(self, category=None):
        if False:
            return 10
        self.lockCursor()
        if category:
            self.persepolis_db_cursor.execute("SELECT * FROM addlink_db_table WHERE category = '{}'".format(category))
        else:
            self.persepolis_db_cursor.execute('SELECT * FROM addlink_db_table')
        rows = self.persepolis_db_cursor.fetchall()
        self.lock = False
        addlink_dict = {}
        for tuple in rows:
            dict = {'gid': tuple[1], 'out': tuple[2], 'start_time': tuple[3], 'end_time': tuple[4], 'link': tuple[5], 'ip': tuple[6], 'port': tuple[7], 'proxy_user': tuple[8], 'proxy_passwd': tuple[9], 'download_user': tuple[10], 'download_passwd': tuple[11], 'connections': tuple[12], 'limit_value': tuple[13], 'download_path': tuple[13], 'referer': tuple[14], 'load_cookies': tuple[15], 'user_agent': tuple[16], 'header': tuple[17], 'after_download': tuple[18]}
            addlink_dict[tuple[1]] = dict
        return addlink_dict

    def updateDownloadTable(self, list):
        if False:
            return 10
        self.lockCursor()
        keys_list = ['file_name', 'status', 'size', 'downloaded_size', 'percent', 'connections', 'rate', 'estimate_time_left', 'gid', 'link', 'first_try_date', 'last_try_date', 'category']
        for dict in list:
            for key in keys_list:
                if key not in dict.keys():
                    dict[key] = None
            self.persepolis_db_cursor.execute('UPDATE download_db_table SET   file_name = coalesce(:file_name, file_name),\n                                                                                    status = coalesce(:status, status),\n                                                                                    size = coalesce(:size, size),\n                                                                                    downloaded_size = coalesce(:downloaded_size, downloaded_size),\n                                                                                    percent = coalesce(:percent, percent),\n                                                                                    connections = coalesce(:connections, connections),\n                                                                                    rate = coalesce(:rate, rate),\n                                                                                    estimate_time_left = coalesce(:estimate_time_left, estimate_time_left),\n                                                                                    link = coalesce(:link, link),\n                                                                                    first_try_date = coalesce(:first_try_date, first_try_date),\n                                                                                    last_try_date = coalesce(:last_try_date, last_try_date),\n                                                                                    category = coalesce(:category, category)\n                                                                                    WHERE gid = :gid', dict)
        self.persepolis_db_connection.commit()
        self.lock = False

    def updateCategoryTable(self, list):
        if False:
            for i in range(10):
                print('nop')
        self.lockCursor()
        keys_list = ['category', 'start_time_enable', 'start_time', 'end_time_enable', 'end_time', 'reverse', 'limit_enable', 'limit_value', 'after_download', 'gid_list']
        for dict in list:
            if 'gid_list' in dict.keys():
                dict['gid_list'] = str(dict['gid_list'])
            for key in keys_list:
                if key not in dict.keys():
                    dict[key] = None
            self.persepolis_db_cursor.execute('UPDATE category_db_table SET   start_time_enable = coalesce(:start_time_enable, start_time_enable),\n                                                                                    start_time = coalesce(:start_time, start_time),\n                                                                                    end_time_enable = coalesce(:end_time_enable, end_time_enable),\n                                                                                    end_time = coalesce(:end_time, end_time),\n                                                                                    reverse = coalesce(:reverse, reverse),\n                                                                                    limit_enable = coalesce(:limit_enable, limit_enable),\n                                                                                    limit_value = coalesce(:limit_value, limit_value),\n                                                                                    after_download = coalesce(:after_download, after_download),\n                                                                                    gid_list = coalesce(:gid_list, gid_list)\n                                                                                    WHERE category = :category', dict)
        self.persepolis_db_connection.commit()
        self.lock = False

    def updateAddLinkTable(self, list):
        if False:
            i = 10
            return i + 15
        self.lockCursor()
        keys_list = ['gid', 'out', 'start_time', 'end_time', 'link', 'ip', 'port', 'proxy_user', 'proxy_passwd', 'download_user', 'download_passwd', 'connections', 'limit_value', 'download_path', 'referer', 'load_cookies', 'user_agent', 'header', 'after_download']
        for dict in list:
            for key in keys_list:
                if key not in dict.keys():
                    dict[key] = None
            self.persepolis_db_cursor.execute('UPDATE addlink_db_table SET out = coalesce(:out, out),\n                                                                                start_time = coalesce(:start_time, start_time),\n                                                                                end_time = coalesce(:end_time, end_time),\n                                                                                link = coalesce(:link, link),\n                                                                                ip = coalesce(:ip, ip),\n                                                                                port = coalesce(:port, port),\n                                                                                proxy_user = coalesce(:proxy_user, proxy_user),\n                                                                                proxy_passwd = coalesce(:proxy_passwd, proxy_passwd),\n                                                                                download_user = coalesce(:download_user, download_user),\n                                                                                download_passwd = coalesce(:download_passwd, download_passwd),\n                                                                                connections = coalesce(:connections, connections),\n                                                                                limit_value = coalesce(:limit_value, limit_value),\n                                                                                download_path = coalesce(:download_path, download_path),\n                                                                                referer = coalesce(:referer, referer),\n                                                                                load_cookies = coalesce(:load_cookies, load_cookies),\n                                                                                user_agent = coalesce(:user_agent, user_agent),\n                                                                                header = coalesce(:header, header),\n                                                                                after_download = coalesce(:after_download , after_download)\n                                                                                WHERE gid = :gid', dict)
        self.persepolis_db_connection.commit()
        self.lock = False

    def updateVideoFinderTable(self, list):
        if False:
            i = 10
            return i + 15
        self.lockCursor()
        keys_list = ['video_gid', 'audio_gid', 'video_completed', 'audio_completed', 'muxing_status', 'checking']
        for dictionary in list:
            for key in keys_list:
                if key not in dictionary.keys():
                    dictionary[key] = None
            if dictionary['video_gid']:
                self.persepolis_db_cursor.execute('UPDATE video_finder_db_table SET video_completed = coalesce(:video_completed, video_completed),\n                                                                                audio_completed = coalesce(:audio_completed, audio_completed),\n                                                                                muxing_status = coalesce(:muxing_status, muxing_status),\n                                                                                checking = coalesce(:checking, checking),\n                                                                                download_path = coalesce(:download_path, download_path)\n                                                                                WHERE video_gid = :video_gid', dictionary)
            elif dictionary['audio_gid']:
                self.persepolis_db_cursor.execute('UPDATE video_finder_db_table SET video_completed = coalesce(:video_completed, video_completed),\n                                                                                audio_completed = coalesce(:audio_completed, audio_completed),\n                                                                                muxing_status = coalesce(:muxing_status, muxing_status),\n                                                                                checking = coalesce(:checking, checking),\n                                                                                download_path = coalesce(:download_path, download_path)\n                                                                                WHERE audio_gid = :audio_gid', dictionary)
        self.persepolis_db_connection.commit()
        self.lock = False

    def setDefaultGidInAddlinkTable(self, gid, start_time=False, end_time=False, after_download=False):
        if False:
            print('Hello World!')
        self.lockCursor()
        if start_time:
            self.persepolis_db_cursor.execute("UPDATE addlink_db_table SET start_time = NULL\n                                                                        WHERE gid = '{}' ".format(gid))
        if end_time:
            self.persepolis_db_cursor.execute("UPDATE addlink_db_table SET end_time = NULL\n                                                                        WHERE gid = '{}' ".format(gid))
        if after_download:
            self.persepolis_db_cursor.execute("UPDATE addlink_db_table SET after_download = NULL\n                                                                        WHERE gid = '{}' ".format(gid))
        self.persepolis_db_connection.commit()
        self.lock = False

    def searchCategoryInCategoryTable(self, category):
        if False:
            print('Hello World!')
        self.lockCursor()
        self.persepolis_db_cursor.execute("SELECT * FROM category_db_table WHERE category = '{}'".format(str(category)))
        list = self.persepolis_db_cursor.fetchall()
        self.lock = False
        if list:
            tuple = list[0]
        else:
            return None
        gid_list = ast.literal_eval(tuple[9])
        dict = {'category': tuple[0], 'start_time_enable': tuple[1], 'start_time': tuple[2], 'end_time_enable': tuple[3], 'end_time': tuple[4], 'reverse': tuple[5], 'limit_enable': tuple[6], 'limit_value': tuple[7], 'after_download': tuple[8], 'gid_list': gid_list}
        return dict

    def categoriesList(self):
        if False:
            while True:
                i = 10
        self.lockCursor()
        self.persepolis_db_cursor.execute('SELECT category FROM category_db_table ORDER BY ROWID')
        rows = self.persepolis_db_cursor.fetchall()
        queues_list = []
        for tuple in rows:
            queues_list.append(tuple[0])
        self.lock = False
        return queues_list

    def setDBTablesToDefaultValue(self):
        if False:
            return 10
        self.lockCursor()
        self.persepolis_db_cursor.execute("UPDATE category_db_table SET start_time_enable = 'no', end_time_enable = 'no',\n                                        reverse = 'no', limit_enable = 'no', after_download = 'no'")
        self.persepolis_db_cursor.execute("UPDATE download_db_table SET status = 'stopped' \n                                        WHERE status NOT IN ('complete', 'error')")
        self.persepolis_db_cursor.execute('UPDATE addlink_db_table SET start_time = NULL,\n                                                                        end_time = NULL,\n                                                                        after_download = NULL\n                                                                                        ')
        self.persepolis_db_cursor.execute("UPDATE video_finder_db_table SET checking = 'no'")
        self.persepolis_db_connection.commit()
        self.lock = False

    def findActiveDownloads(self, category=None):
        if False:
            return 10
        self.lockCursor()
        if category:
            self.persepolis_db_cursor.execute("SELECT gid FROM download_db_table WHERE (category = '{}') AND (status = 'downloading' OR status = 'waiting' \n                                            OR status = 'scheduled' OR status = 'paused')".format(str(category)))
        else:
            self.persepolis_db_cursor.execute("SELECT gid FROM download_db_table WHERE (status = 'downloading' OR status = 'waiting' \n                                            OR status = 'scheduled' OR status = 'paused')")
        result = self.persepolis_db_cursor.fetchall()
        gid_list = []
        for result_tuple in result:
            gid_list.append(result_tuple[0])
        self.lock = False
        return gid_list

    def returnDownloadingItems(self):
        if False:
            return 10
        self.lockCursor()
        self.persepolis_db_cursor.execute("SELECT gid FROM download_db_table WHERE (status = 'downloading' OR status = 'waiting')")
        result = self.persepolis_db_cursor.fetchall()
        gid_list = []
        for result_tuple in result:
            gid_list.append(result_tuple[0])
        self.lock = False
        return gid_list

    def returnPausedItems(self):
        if False:
            while True:
                i = 10
        self.lockCursor()
        self.persepolis_db_cursor.execute("SELECT gid FROM download_db_table WHERE (status = 'paused')")
        result = self.persepolis_db_cursor.fetchall()
        gid_list = []
        for result_tuple in result:
            gid_list.append(result_tuple[0])
        self.lock = False
        return gid_list

    def returnVideoFinderGids(self):
        if False:
            for i in range(10):
                print('nop')
        self.lockCursor()
        self.persepolis_db_cursor.execute('SELECT video_gid, audio_gid FROM video_finder_db_table')
        result = self.persepolis_db_cursor.fetchall()
        self.lock = False
        gid_list = []
        video_gid_list = []
        audio_gid_list = []
        for result_tuple in result:
            gid_list.append(result_tuple[0])
            video_gid_list.append(result_tuple[0])
            gid_list.append(result_tuple[1])
            audio_gid_list.append(result_tuple[1])
        return (gid_list, video_gid_list, audio_gid_list)

    def deleteCategory(self, category):
        if False:
            print('Hello World!')
        category_dict = self.searchCategoryInCategoryTable(category)
        all_downloads_dict = self.searchCategoryInCategoryTable('All Downloads')
        category_gid_list = category_dict['gid_list']
        all_downloads_gid_list = all_downloads_dict['gid_list']
        for gid in category_gid_list:
            all_downloads_gid_list.remove(gid)
        self.updateCategoryTable([all_downloads_dict])
        self.lockCursor()
        self.persepolis_db_cursor.execute("DELETE FROM category_db_table WHERE category = '{}'".format(str(category)))
        self.persepolis_db_connection.commit()
        self.lock = False

    def resetDataBase(self):
        if False:
            while True:
                i = 10
        all_downloads_dict = {'category': 'All Downloads', 'gid_list': []}
        single_downloads_dict = {'category': 'Single Downloads', 'gid_list': []}
        scheduled_downloads_dict = {'category': 'Scheduled Downloads', 'gid_list': []}
        self.updateCategoryTable([all_downloads_dict, single_downloads_dict, scheduled_downloads_dict])
        self.lockCursor()
        self.persepolis_db_cursor.execute("DELETE FROM category_db_table WHERE category NOT IN ('All Downloads', 'Single Downloads', 'Scheduled Downloads')")
        self.persepolis_db_cursor.execute('DELETE FROM download_db_table')
        self.persepolis_db_cursor.execute('DELETE FROM addlink_db_table')
        self.persepolis_db_connection.commit()
        self.lock = False

    def deleteItemInDownloadTable(self, gid, category):
        if False:
            print('Hello World!')
        self.lockCursor()
        self.persepolis_db_cursor.execute("DELETE FROM download_db_table WHERE gid = '{}'".format(str(gid)))
        self.persepolis_db_connection.commit()
        self.lock = False
        for category_name in (category, 'All Downloads'):
            category_dict = self.searchCategoryInCategoryTable(category_name)
            gid_list = category_dict['gid_list']
            if gid in gid_list:
                gid_list.remove(gid)
                video_finder_dictionary = self.searchGidInVideoFinderTable(gid)
                if video_finder_dictionary:
                    video_gid = video_finder_dictionary['video_gid']
                    audio_gid = video_finder_dictionary['audio_gid']
                    if gid == video_gid:
                        gid_list.remove(audio_gid)
                    else:
                        gid_list.remove(video_gid)
                self.updateCategoryTable([category_dict])

    def correctDataBase(self):
        if False:
            while True:
                i = 10
        self.lockCursor()
        for units in [['KB', 'KiB'], ['MB', 'MiB'], ['GB', 'GiB']]:
            dict = {'old_unit': units[0], 'new_unit': units[1]}
            self.persepolis_db_cursor.execute('UPDATE download_db_table \n                    SET size = replace(size, :old_unit, :new_unit)', dict)
            self.persepolis_db_cursor.execute('UPDATE download_db_table \n                    SET rate = replace(rate, :old_unit, :new_unit)', dict)
            self.persepolis_db_cursor.execute('UPDATE download_db_table \n                    SET downloaded_size = replace(downloaded_size, :old_unit, :new_unit)', dict)
        self.persepolis_db_connection.commit()
        self.lock = False

    def closeConnections(self):
        if False:
            for i in range(10):
                print('nop')
        self.lockCursor()
        self.persepolis_db_cursor.close()
        self.persepolis_db_connection.close()
        self.lock = False