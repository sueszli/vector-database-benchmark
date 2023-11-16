import os
import sqlite3
from .settings import Settings
SELECT_FROM_PROFILE_WHERE_NAME = 'SELECT * FROM profiles WHERE name = :name'
INSERT_INTO_PROFILE = 'INSERT INTO profiles (name) VALUES (?)'
SQL_CREATE_PROFILE_TABLE = '\n    CREATE TABLE IF NOT EXISTS `profiles` (\n        `id` INTEGER PRIMARY KEY AUTOINCREMENT,\n        `name` TEXT NOT NULL);'
SQL_CREATE_RECORD_ACTIVITY_TABLE = '\n    CREATE TABLE IF NOT EXISTS `recordActivity` (\n        `profile_id` INTEGER REFERENCES `profiles` (id),\n        `likes` SMALLINT UNSIGNED NOT NULL,\n        `comments` SMALLINT UNSIGNED NOT NULL,\n        `follows` SMALLINT UNSIGNED NOT NULL,\n        `unfollows` SMALLINT UNSIGNED NOT NULL,\n        `server_calls` INT UNSIGNED NOT NULL,\n        `created` DATETIME NOT NULL);'
SQL_CREATE_FOLLOW_RESTRICTION_TABLE = '\n    CREATE TABLE IF NOT EXISTS `followRestriction` (\n        `profile_id` INTEGER REFERENCES `profiles` (id),\n        `username` TEXT NOT NULL,\n        `times` TINYINT UNSIGNED NOT NULL);'
SQL_CREATE_SHARE_WITH_PODS_RESTRICTION_TABLE = '\n    CREATE TABLE IF NOT EXISTS `shareWithPodsRestriction` (\n        `profile_id` INTEGER REFERENCES `profiles` (id),\n        `postid` TEXT NOT NULL,\n        `times` TINYINT UNSIGNED NOT NULL);'
SQL_CREATE_COMMENT_RESTRICTION_TABLE = '\n    CREATE TABLE IF NOT EXISTS `commentRestriction` (\n        `profile_id` INTEGER REFERENCES `profiles` (id),\n        `postid` TEXT NOT NULL,\n        `times` TINYINT UNSIGNED NOT NULL);'
SQL_CREATE_ACCOUNTS_PROGRESS_TABLE = '\n    CREATE TABLE IF NOT EXISTS `accountsProgress` (\n        `profile_id` INTEGER NOT NULL,\n        `followers` INTEGER NOT NULL,\n        `following` INTEGER NOT NULL,\n        `total_posts` INTEGER NOT NULL,\n        `created` DATETIME NOT NULL,\n        `modified` DATETIME NOT NULL,\n        CONSTRAINT `fk_accountsProgress_profiles1`\n        FOREIGN KEY(`profile_id`) REFERENCES `profiles`(`id`));'

def get_database(make=False):
    if False:
        for i in range(10):
            print('nop')
    logger = Settings.logger
    credentials = Settings.profile
    (profile_id, name) = (credentials['id'], credentials['name'])
    address = validate_database_address()
    if not os.path.isfile(address) or make:
        create_database(address, logger, name)
    profile_id = get_profile(name, address, logger) if profile_id is None or make else profile_id
    return (address, profile_id)

def create_database(address, logger, name):
    if False:
        i = 10
        return i + 15
    try:
        connection = sqlite3.connect(address)
        with connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            create_tables(cursor, ['profiles', 'recordActivity', 'followRestriction', 'shareWithPodsRestriction', 'commentRestriction', 'accountsProgress'])
            connection.commit()
    except Exception as exc:
        logger.warning("Wah! Error occurred while getting a DB for '{}':\n\t{}".format(name, str(exc).encode('utf-8')))
    finally:
        if connection:
            connection.close()

def create_tables(cursor, tables):
    if False:
        i = 10
        return i + 15
    if 'profiles' in tables:
        cursor.execute(SQL_CREATE_PROFILE_TABLE)
    if 'recordActivity' in tables:
        cursor.execute(SQL_CREATE_RECORD_ACTIVITY_TABLE)
    if 'followRestriction' in tables:
        cursor.execute(SQL_CREATE_FOLLOW_RESTRICTION_TABLE)
    if 'shareWithPodsRestriction' in tables:
        cursor.execute(SQL_CREATE_SHARE_WITH_PODS_RESTRICTION_TABLE)
    if 'commentRestriction' in tables:
        cursor.execute(SQL_CREATE_COMMENT_RESTRICTION_TABLE)
    if 'accountsProgress' in tables:
        cursor.execute(SQL_CREATE_ACCOUNTS_PROGRESS_TABLE)

def verify_database_directories(address):
    if False:
        print('Hello World!')
    db_dir = os.path.dirname(address)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

def validate_database_address():
    if False:
        while True:
            i = 10
    address = Settings.database_location
    if not address.endswith('.db'):
        slash = '\\' if '\\' in address else '/'
        address = address if address.endswith(slash) else address + slash
        address += 'instapy.db'
        Settings.database_location = address
    verify_database_directories(address)
    return address

def get_profile(name, address, logger):
    if False:
        print('Hello World!')
    try:
        conn = sqlite3.connect(address)
        with conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            profile = select_profile_by_username(cursor, name)
            if profile is None:
                add_profile(conn, cursor, name)
                profile = select_profile_by_username(cursor, name)
    except Exception as exc:
        logger.warning("Heeh! Error occurred while getting a DB profile for '{}':\n\t{}".format(name, str(exc).encode('utf-8')))
    finally:
        if conn:
            conn.close()
    profile = dict(profile)
    profile_id = profile['id']
    Settings.profile['id'] = profile_id
    return profile_id

def add_profile(conn, cursor, name):
    if False:
        print('Hello World!')
    cursor.execute(INSERT_INTO_PROFILE, (name,))
    conn.commit()

def select_profile_by_username(cursor, name):
    if False:
        return 10
    cursor.execute(SELECT_FROM_PROFILE_WHERE_NAME, {'name': name})
    profile = cursor.fetchone()
    return profile