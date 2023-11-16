import random
import secrets
import string
from uuid import uuid4
import pytest
from db import db
from journalist_app import create_app
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from .helpers import bool_or_none, random_bool, random_bytes, random_chars, random_datetime, random_username
random.seed('ᕕ( ᐛ )ᕗ')

class Helper:

    @staticmethod
    def add_source():
        if False:
            i = 10
            return i + 15
        filesystem_id = random_chars(96) if random_bool() else None
        params = {'uuid': str(uuid4()), 'filesystem_id': filesystem_id, 'journalist_designation': random_chars(50), 'flagged': bool_or_none(), 'last_updated': random_datetime(nullable=True), 'pending': bool_or_none(), 'interaction_count': random.randint(0, 1000)}
        sql = '\n        INSERT INTO sources (\n            uuid,\n            filesystem_id,\n            journalist_designation,\n            flagged,\n            last_updated,\n            pending,\n            interaction_count\n        )\n        VALUES (\n            :uuid,\n            :filesystem_id,\n            :journalist_designation,\n            :flagged,\n            :last_updated,\n            :pending,\n            :interaction_count\n        )\n        '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def add_journalist():
        if False:
            while True:
                i = 10
        if random_bool():
            otp_secret = random_chars(16, string.ascii_uppercase + '234567')
        else:
            otp_secret = None
        is_totp = random_bool()
        if is_totp:
            hotp_counter = 0 if random_bool() else None
        else:
            hotp_counter = random.randint(0, 10000) if random_bool() else None
        last_token = random_chars(6, string.digits) if random_bool() else None
        params = {'uuid': str(uuid4()), 'username': random_username(), 'session_nonce': 0, 'pw_salt': random_bytes(1, 64, nullable=True), 'pw_hash': random_bytes(32, 64, nullable=True), 'is_admin': bool_or_none(), 'otp_secret': otp_secret, 'is_totp': is_totp, 'hotp_counter': hotp_counter, 'last_token': last_token, 'created_on': random_datetime(nullable=True), 'last_access': random_datetime(nullable=True)}
        sql = '\n        INSERT INTO journalists (\n            uuid,\n            username,\n            session_nonce,\n            pw_salt,\n            pw_hash,\n            is_admin,\n            otp_secret,\n            is_totp,\n            hotp_counter,\n            last_token,\n            created_on,\n            last_access\n        )\n        VALUES (\n            :uuid,\n            :username,\n            :session_nonce,\n            :pw_salt,\n            :pw_hash,\n            :is_admin,\n            :otp_secret,\n            :is_totp,\n            :hotp_counter,\n            :last_token,\n            :created_on,\n            :last_access\n        );\n        '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def add_reply(journalist_id, source_id):
        if False:
            while True:
                i = 10
        params = {'uuid': str(uuid4()), 'journalist_id': journalist_id, 'source_id': source_id, 'filename': random_chars(50), 'size': random.randint(0, 1024 * 1024 * 500), 'deleted_by_source': 0}
        sql = '\n        INSERT INTO replies (uuid, journalist_id, source_id, filename, size, deleted_by_source)\n        VALUES (:uuid, :journalist_id, :source_id, :filename, :size, :deleted_by_source)\n        '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def add_message(source_id):
        if False:
            return 10
        params = {'uuid': str(uuid4()), 'source_id': source_id, 'filename': random_chars(50) + '-msg.gpg', 'size': random.randint(0, 1024 * 1024 * 500), 'downloaded': secrets.choice([True, False])}
        sql = '\n        INSERT INTO submissions (uuid, source_id, filename, size, downloaded)\n        VALUES (:uuid, :source_id, :filename, :size, :downloaded)\n        '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def add_file(source_id):
        if False:
            print('Hello World!')
        params = {'uuid': str(uuid4()), 'source_id': source_id, 'filename': random_chars(50) + '-doc.gz.gpg', 'size': random.randint(0, 1024 * 1024 * 500), 'downloaded': secrets.choice([True, False]), 'checksum': 'sha256:' + random_chars(64)}
        sql = '\n        INSERT INTO submissions (uuid, source_id, filename, size, downloaded, checksum)\n        VALUES (:uuid, :source_id, :filename, :size, :downloaded, :checksum)\n        '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def mark_reply_as_seen(reply_id, journalist_id):
        if False:
            return 10
        params = {'reply_id': reply_id, 'journalist_id': journalist_id}
        sql = '\n        INSERT INTO seen_replies (reply_id, journalist_id)\n        VALUES (:reply_id, :journalist_id)\n        '
        try:
            db.engine.execute(text(sql), **params)
        except IntegrityError:
            pass

    @staticmethod
    def mark_file_as_seen(file_id, journalist_id):
        if False:
            return 10
        params = {'file_id': file_id, 'journalist_id': journalist_id}
        sql = '\n        INSERT INTO seen_files (file_id, journalist_id)\n        VALUES (:file_id, :journalist_id)\n        '
        try:
            db.engine.execute(text(sql), **params)
        except IntegrityError:
            pass

    @staticmethod
    def mark_message_as_seen(message_id, journalist_id):
        if False:
            i = 10
            return i + 15
        params = {'message_id': message_id, 'journalist_id': journalist_id}
        sql = '\n        INSERT INTO seen_messages (message_id, journalist_id)\n        VALUES (:message_id, :journalist_id)\n        '
        try:
            db.engine.execute(text(sql), **params)
        except IntegrityError:
            pass

class UpgradeTester(Helper):
    """
    This migration verifies that the seen_files, seen_messages, and seen_replies association tables
    now exist, and that the data migration completes successfully.
    """
    JOURNO_NUM = 20
    SOURCE_NUM = 20

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            print('Hello World!')
        with self.app.app_context():
            for _ in range(self.SOURCE_NUM):
                self.add_source()
            for _ in range(self.JOURNO_NUM):
                self.add_journalist()
            for i in range(self.SOURCE_NUM):
                for _ in range(random.randint(1, 3)):
                    self.add_message(i)
                for _ in range(random.randint(0, 2)):
                    self.add_file(i)
            for i in range(30):
                selected_journo = random.randint(0, self.JOURNO_NUM)
                selected_source = random.randint(0, self.SOURCE_NUM)
                self.add_reply(selected_journo, selected_source)

    def check_upgrade(self):
        if False:
            i = 10
            return i + 15
        with self.app.app_context():
            sql = 'SELECT * FROM submissions'
            submissions = db.engine.execute(text(sql)).fetchall()
            sql = 'SELECT * FROM replies'
            replies = db.engine.execute(text(sql)).fetchall()
            for submission in submissions:
                if submission.filename.endswith('-doc.gz.gpg') and secrets.choice([0, 1]):
                    selected_journo_id = random.randint(0, self.JOURNO_NUM)
                    self.mark_file_as_seen(submission.id, selected_journo_id)
                elif secrets.choice([0, 1]):
                    selected_journo_id = random.randint(0, self.JOURNO_NUM)
                    self.mark_message_as_seen(submission.id, selected_journo_id)
            for reply in replies:
                if secrets.choice([0, 1]):
                    selected_journo_id = random.randint(0, self.JOURNO_NUM)
                    self.mark_reply_as_seen(reply.id, selected_journo_id)
            params = {'reply_id': 100, 'journalist_id': 100}
            sql = '\n            INSERT INTO seen_replies (reply_id, journalist_id)\n            VALUES (:reply_id, :journalist_id);\n            '
            db.engine.execute(text(sql), **params)
            with pytest.raises(IntegrityError):
                db.engine.execute(text(sql), **params)
            params = {'message_id': 100, 'journalist_id': 100}
            sql = '\n            INSERT INTO seen_messages (message_id, journalist_id)\n            VALUES (:message_id, :journalist_id);\n            '
            db.engine.execute(text(sql), **params)
            with pytest.raises(IntegrityError):
                db.engine.execute(text(sql), **params)
            params = {'file_id': 101, 'journalist_id': 100}
            sql = '\n            INSERT INTO seen_files (file_id, journalist_id)\n            VALUES (:file_id, :journalist_id);\n            '
            db.engine.execute(text(sql), **params)
            with pytest.raises(IntegrityError):
                db.engine.execute(text(sql), **params)

class DowngradeTester(Helper):
    """
    This migration verifies that the seen_files, seen_messages, and seen_replies association tables
    are removed.
    """
    JOURNO_NUM = 20
    SOURCE_NUM = 20

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            while True:
                i = 10
        with self.app.app_context():
            for _ in range(self.SOURCE_NUM):
                self.add_source()
            for _ in range(self.JOURNO_NUM):
                self.add_journalist()
            for i in range(self.SOURCE_NUM):
                for _ in range(random.randint(1, 3)):
                    self.add_message(i)
                for _ in range(random.randint(0, 2)):
                    self.add_file(i)
            for i in range(30):
                selected_journo = random.randint(0, self.JOURNO_NUM)
                selected_source = random.randint(0, self.SOURCE_NUM)
                self.add_reply(selected_journo, selected_source)
            sql = 'SELECT * FROM submissions'
            submissions = db.engine.execute(text(sql)).fetchall()
            for submission in submissions:
                if submission.filename.endswith('-doc.gz.gpg') and secrets.choice([0, 1]):
                    selected_journo_id = random.randint(0, self.JOURNO_NUM)
                    self.mark_file_as_seen(submission.id, selected_journo_id)
                elif secrets.choice([0, 1]):
                    selected_journo_id = random.randint(0, self.JOURNO_NUM)
                    self.mark_message_as_seen(submission.id, selected_journo_id)
            sql = 'SELECT * FROM replies'
            replies = db.engine.execute(text(sql)).fetchall()
            for reply in replies:
                if secrets.choice([0, 1]):
                    selected_journo_id = random.randint(0, self.JOURNO_NUM)
                    self.mark_reply_as_seen(reply.id, selected_journo_id)
            for submission in submissions:
                if submission.filename.endswith('-doc.gz.gpg') and secrets.choice([0, 1]):
                    selected_journo_id = random.randint(0, self.JOURNO_NUM)
                    self.mark_file_as_seen(submission.id, selected_journo_id)
                elif secrets.choice([0, 1]):
                    selected_journo_id = random.randint(0, self.JOURNO_NUM)
                    self.mark_message_as_seen(submission.id, selected_journo_id)
            for reply in replies:
                if secrets.choice([0, 1]):
                    selected_journo_id = random.randint(0, self.JOURNO_NUM)
                    self.mark_reply_as_seen(reply.id, selected_journo_id)

    def check_downgrade(self):
        if False:
            for i in range(10):
                print('nop')
        with self.app.app_context():
            params = {'table_name': 'seen_files'}
            sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name;"
            seen_files_exists = db.engine.execute(text(sql), **params).fetchall()
            assert not seen_files_exists
            params = {'table_name': 'seen_messages'}
            sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name;"
            seen_messages_exists = db.engine.execute(text(sql), **params).fetchall()
            assert not seen_messages_exists
            params = {'table_name': 'seen_replies'}
            sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name;"
            seen_replies_exists = db.engine.execute(text(sql), **params).fetchall()
            assert not seen_replies_exists