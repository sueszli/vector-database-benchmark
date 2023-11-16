import subprocess

class BashMock:

    @staticmethod
    def tag_latest_release(tag):
        if False:
            i = 10
            return i + 15
        bash_command = f'./scripts/tag_latest_release.sh {tag} --dry-run'
        result = subprocess.run(bash_command, shell=True, capture_output=True, text=True, env={'TEST_ENV': 'true'})
        return result

    @staticmethod
    def docker_build_push(tag, branch):
        if False:
            i = 10
            return i + 15
        bash_command = f'./scripts/docker_build_push.sh {tag}'
        result = subprocess.run(bash_command, shell=True, capture_output=True, text=True, env={'TEST_ENV': 'true', 'GITHUB_REF': f'refs/heads/{branch}'})
        return result