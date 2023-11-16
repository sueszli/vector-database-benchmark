from bottles.backend.logger import Logger
logging = Logger()

class GSettingsStub:

    @staticmethod
    def get_boolean(key: str) -> bool:
        if False:
            while True:
                i = 10
        logging.warning(f'Stub GSettings key {key}=False')
        return False