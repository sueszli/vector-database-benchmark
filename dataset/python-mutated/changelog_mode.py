from pyflink.java_gateway import get_gateway
__all__ = ['ChangelogMode']

class ChangelogMode(object):
    """
    The set of changes contained in a changelog.
    """

    def __init__(self, j_changelog_mode):
        if False:
            print('Hello World!')
        self._j_changelog_mode = j_changelog_mode

    @staticmethod
    def insert_only():
        if False:
            for i in range(10):
                print('nop')
        gateway = get_gateway()
        return ChangelogMode(gateway.jvm.org.apache.flink.table.connector.ChangelogMode.insertOnly())

    @staticmethod
    def upsert():
        if False:
            while True:
                i = 10
        gateway = get_gateway()
        return ChangelogMode(gateway.jvm.org.apache.flink.table.connector.ChangelogMode.upsert())

    @staticmethod
    def all():
        if False:
            for i in range(10):
                print('nop')
        gateway = get_gateway()
        return ChangelogMode(gateway.jvm.org.apache.flink.table.connector.ChangelogMode.all())