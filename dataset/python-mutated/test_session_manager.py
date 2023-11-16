from datetime import datetime, timedelta, timezone
from unittest import mock
import pytest
from db import db
from passphrases import PassphraseGenerator
from source_app.session_manager import SessionManager, UserHasBeenDeleted, UserNotLoggedIn, UserSessionExpired
from source_user import create_source_user

class TestSessionManager:

    def test_log_user_in(self, source_app, app_storage):
        if False:
            return 10
        passphrase = PassphraseGenerator.get_default().generate_passphrase()
        source_user = create_source_user(db_session=db.session, source_passphrase=passphrase, source_app_storage=app_storage)
        with source_app.test_request_context():
            SessionManager.log_user_in(db_session=db.session, supplied_passphrase=passphrase)
            assert SessionManager.is_user_logged_in(db_session=db.session)
            logged_in_user = SessionManager.get_logged_in_user(db_session=db.session)
            assert logged_in_user.db_record_id == source_user.db_record_id

    def test_log_user_out(self, source_app, app_storage):
        if False:
            i = 10
            return i + 15
        passphrase = PassphraseGenerator.get_default().generate_passphrase()
        create_source_user(db_session=db.session, source_passphrase=passphrase, source_app_storage=app_storage)
        with source_app.test_request_context():
            SessionManager.log_user_in(db_session=db.session, supplied_passphrase=passphrase)
            SessionManager.log_user_out()
            assert not SessionManager.is_user_logged_in(db_session=db.session)
            with pytest.raises(UserNotLoggedIn):
                SessionManager.get_logged_in_user(db_session=db.session)

    def test_get_logged_in_user_but_session_expired(self, source_app, app_storage):
        if False:
            for i in range(10):
                print('nop')
        passphrase = PassphraseGenerator.get_default().generate_passphrase()
        create_source_user(db_session=db.session, source_passphrase=passphrase, source_app_storage=app_storage)
        with source_app.test_request_context():
            SessionManager.log_user_in(db_session=db.session, supplied_passphrase=passphrase)
            with mock.patch('source_app.session_manager.datetime') as mock_datetime:
                six_hours_later = datetime.now(timezone.utc) + timedelta(hours=6)
                mock_datetime.now.return_value = six_hours_later
                with pytest.raises(UserSessionExpired):
                    SessionManager.get_logged_in_user(db_session=db.session)

    def test_get_logged_in_user_but_user_deleted(self, source_app, app_storage):
        if False:
            print('Hello World!')
        passphrase = PassphraseGenerator.get_default().generate_passphrase()
        source_user = create_source_user(db_session=db.session, source_passphrase=passphrase, source_app_storage=app_storage)
        with source_app.test_request_context():
            SessionManager.log_user_in(db_session=db.session, supplied_passphrase=passphrase)
            source_in_db = source_user.get_db_record()
            source_in_db.deleted_at = datetime.utcnow()
            db.session.commit()
            with pytest.raises(UserHasBeenDeleted):
                SessionManager.get_logged_in_user(db_session=db.session)