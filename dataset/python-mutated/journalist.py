import sys
from encryption import EncryptionManager, GpgKeyNotFoundError
from execution import asynchronous
from journalist_app import create_app
from models import Source
from sdconfig import SecureDropConfig
from startup import validate_journalist_key
config = SecureDropConfig.get_current()
app = create_app(config)

@asynchronous
def prime_keycache() -> None:
    if False:
        return 10
    'Pre-load the source public keys into Redis.'
    with app.app_context():
        encryption_mgr = EncryptionManager.get_default()
        for source in Source.query.filter_by(pending=False, deleted_at=None).all():
            try:
                encryption_mgr.get_source_public_key(source.filesystem_id)
            except GpgKeyNotFoundError:
                pass
if not validate_journalist_key(app):
    sys.exit(1)
prime_keycache()
if __name__ == '__main__':
    debug = getattr(config, 'env', 'prod') != 'prod'
    app.run(debug=debug, host='0.0.0.0', port=8081)