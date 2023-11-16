"""
Module used for detecting whether SAMCLI is executed in a CI/CD environment.
"""
import os
from enum import Enum, auto
from typing import Callable, Dict, Mapping, Optional, Union

class CICDPlatform(Enum):
    Jenkins = auto()
    GitLab = auto()
    GitHubAction = auto()
    TravisCI = auto()
    CircleCI = auto()
    AWSCodeBuild = auto()
    TeamCity = auto()
    Bamboo = auto()
    Buddy = auto()
    CodeShip = auto()
    Semaphore = auto()
    Appveyor = auto()
    Other = auto()

def _is_codeship(environ: Mapping) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Use environ to determine whether it is running in CodeShip.\n    According to the doc,\n    https://docs.cloudbees.com/docs/cloudbees-codeship/latest/basic-builds-and-configuration/set-environment-variables\n    > CI_NAME                # Always CodeShip. Ex: codeship\n\n    to handle both "CodeShip" and "codeship," here the string is converted to lower case first.\n\n    Parameters\n    ----------\n    environ\n\n    Returns\n    -------\n    bool\n        whether the env is CodeShip\n    '
    ci_name: str = environ.get('CI_NAME', '').lower()
    return ci_name == 'codeship'

def _is_jenkins(environ: Mapping) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Use environ to determine whether it is running in Jenkins.\n    According to the doc,\n    https://www.jenkins.io/doc/book/pipeline/jenkinsfile/#working-with-your-jenkinsfile\n    > BUILD_TAG\n    >   String of jenkins-${JOB_NAME}-${BUILD_NUMBER}.\n    > ...\n    > JENKINS_URL\n    >   Full URL of Jenkins, such as https://example.com:port/jenkins/\n    >   (NOTE: only available if Jenkins URL set in "System Configuration")\n\n    Here firstly check JENKINS_URL\'s presence, if not, then fallback to check BUILD_TAG starts with "jenkins"\n    '
    return 'JENKINS_URL' in environ or environ.get('BUILD_TAG', '').startswith('jenkins-')
_ENV_VAR_OR_CALLABLE_BY_PLATFORM: Dict[CICDPlatform, Union[str, Callable[[Mapping], bool]]] = {CICDPlatform.Jenkins: _is_jenkins, CICDPlatform.GitLab: 'GITLAB_CI', CICDPlatform.GitHubAction: 'GITHUB_ACTION', CICDPlatform.TravisCI: 'TRAVIS', CICDPlatform.CircleCI: 'CIRCLECI', CICDPlatform.AWSCodeBuild: 'CODEBUILD_BUILD_ID', CICDPlatform.TeamCity: 'TEAMCITY_VERSION', CICDPlatform.Bamboo: 'bamboo_buildNumber', CICDPlatform.Buddy: 'BUDDY', CICDPlatform.CodeShip: _is_codeship, CICDPlatform.Semaphore: 'SEMAPHORE', CICDPlatform.Appveyor: 'APPVEYOR', CICDPlatform.Other: 'CI'}

def _is_cicd_platform(cicd_platform: CICDPlatform, environ: Mapping) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Check whether sam-cli run in a particular CI/CD platform based on certain environment variables.\n\n    Parameters\n    ----------\n    cicd_platform\n        an enum CICDPlatform object indicating which CI/CD platform  to check against.\n    environ\n        the mapping to look for environment variables, for example, os.environ.\n\n    Returns\n    -------\n    bool\n        A boolean indicating whether there are environment variables matching the cicd_platform.\n    '
    env_var_or_callable = _ENV_VAR_OR_CALLABLE_BY_PLATFORM[cicd_platform]
    if isinstance(env_var_or_callable, str):
        return env_var_or_callable in environ
    return env_var_or_callable(environ)

class CICDDetector:
    _cicd_platform: Optional[CICDPlatform]

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self._cicd_platform: Optional[CICDPlatform] = next((cicd_platform for cicd_platform in CICDPlatform if _is_cicd_platform(cicd_platform, os.environ)))
        except StopIteration:
            self._cicd_platform = None

    def platform(self) -> Optional[CICDPlatform]:
        if False:
            print('Hello World!')
        '\n        Identify which CICD platform SAM CLI is running in.\n        Returns\n        -------\n        CICDPlatform\n            an optional CICDPlatform enum indicating the CICD platform.\n        '
        return self._cicd_platform