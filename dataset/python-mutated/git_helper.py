import plumbum
git = plumbum.local['git']

class GitHelper:

    @staticmethod
    def commit_hash() -> str:
        if False:
            for i in range(10):
                print('nop')
        return git['rev-parse', 'HEAD']().strip()

    @staticmethod
    def commit_hash_tag() -> str:
        if False:
            for i in range(10):
                print('nop')
        return GitHelper.commit_hash()[:12]

    @staticmethod
    def commit_message() -> str:
        if False:
            return 10
        return git['log', -1, '--pretty=%B']().strip()
if __name__ == '__main__':
    print('Git hash:', GitHelper.commit_hash())
    print('Git message:', GitHelper.commit_message())