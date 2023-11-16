import os
from glob import glob
from amazon_kclpy import kcl
from localstack.utils.files import save_file

def get_dir_of_file(f):
    if False:
        i = 10
        return i + 15
    return os.path.dirname(os.path.abspath(f))

def get_kcl_dir():
    if False:
        return 10
    return get_dir_of_file(kcl.__file__)

def get_kcl_jar_path():
    if False:
        while True:
            i = 10
    jars = ':'.join(glob(os.path.join(get_kcl_dir(), 'jars', '*jar')))
    return jars

def get_kcl_classpath(properties=None, paths=None):
    if False:
        return 10
    '\n    Generates a classpath that includes the location of the kcl jars, the\n    properties file and the optional paths.\n\n    :type properties: str\n    :param properties: Path to properties file.\n\n    :type paths: list\n    :param paths: List of strings. The paths that will be prepended to the classpath.\n\n    :rtype: str\n    :return: A java class path that will allow your properties to be\n             found and the MultiLangDaemon and its deps and\n        any custom paths you provided.\n    '
    if paths is None:
        paths = []
    paths = [os.path.abspath(p) for p in paths]
    paths.append(get_kcl_jar_path())
    if properties:
        dir_of_file = get_dir_of_file(properties)
        paths.append(dir_of_file)
    dir_name = os.path.dirname(os.path.realpath(__file__))
    paths.insert(0, os.path.realpath(os.path.join(dir_name, 'java')))
    return ':'.join([p for p in paths if p != ''])

def get_kcl_app_command(java, multi_lang_daemon_class, properties, paths=None):
    if False:
        print('Hello World!')
    '\n    Generates a command to run the MultiLangDaemon.\n\n    :type java: str\n    :param java: Path to java\n\n    :type multi_lang_daemon_class: str\n    :param multi_lang_daemon_class: Name of multi language daemon class, e.g.\n            com.amazonaws.services.kinesis.multilang.MultiLangDaemon\n\n    :type properties: str\n    :param properties: Optional properties file to be included in the classpath.\n\n    :type paths: list\n    :param paths: List of strings. Additional paths to prepend to the classpath.\n\n    :rtype: str\n    :return: A command that will run the MultiLangDaemon with your\n             properties and custom paths and java.\n    '
    if paths is None:
        paths = []
    logging_config = os.path.join(get_dir_of_file(__file__), 'java', 'logging.properties')
    sys_props = f'-Djava.util.logging.config.file="{logging_config}" -Daws.cborEnabled=false'
    return '{java} -cp {cp} {sys_props} {daemon} {props}'.format(java=java, cp=get_kcl_classpath(properties, paths), daemon=multi_lang_daemon_class, props=os.path.basename(properties), sys_props=sys_props)

def create_config_file(config_file, executableName, streamName, applicationName, region_name, credentialsProvider=None, **kwargs):
    if False:
        i = 10
        return i + 15
    if not credentialsProvider:
        credentialsProvider = 'DefaultAWSCredentialsProviderChain'
    content = f'\n        executableName = {executableName}\n        streamName = {streamName}\n        applicationName = {applicationName}\n        AWSCredentialsProvider = {credentialsProvider}\n        kinesisCredentialsProvider = {credentialsProvider}\n        dynamoDBCredentialsProvider = {credentialsProvider}\n        cloudWatchCredentialsProvider = {credentialsProvider}\n        processingLanguage = python/3.10\n        shardSyncIntervalMillis = 2000\n        parentShardPollIntervalMillis = 2000\n        idleTimeBetweenReadsInMillis = 1000\n        timeoutInSeconds = 60\n        regionName = {region_name}\n    '
    for (key, value) in kwargs.items():
        content += f'\n{key} = {value}'
    content = content.replace('    ', '')
    save_file(config_file, content)