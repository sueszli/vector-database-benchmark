""" For creating virtual environments.

This can create empty virtualenv, but also populate them with packages from
the reports.
"""
import os
from nuitka.Tracing import tools_logger
from nuitka.TreeXML import fromFile
from nuitka.utils.FileOperations import openTextFile
from .Virtualenv import withVirtualenv

def createEnvironmentFromReport(environment_folder, report_filename):
    if False:
        return 10
    if os.path.exists(environment_folder):
        tools_logger.sysexit('Error, environment folder must not exist already.')
    containing_folder = os.path.dirname(environment_folder)
    if not os.path.exists(containing_folder):
        tools_logger.sysexit("Error, environment folder must be in existing folder, not '%s'." % containing_folder)
    if not os.path.isfile(report_filename):
        tools_logger.sysexit("Error, no such report file '%s'." % report_filename)
    with openTextFile(report_filename, 'r', encoding='utf8') as report_file:
        root = fromFile(report_file, use_lxml=True)
    requirements_filename = os.path.join(environment_folder, 'requirements.txt')
    with withVirtualenv(os.path.basename(environment_folder), base_dir=containing_folder, style='blue', delete=False) as venv:
        with openTextFile(requirements_filename, 'w', encoding='utf8') as requirements_file:
            for node in root.xpath('distributions/distribution'):
                requirements_file.write('%s==%s\n' % (node.attrib['name'], node.attrib['version']))
        venv.runCommand('pip install -r requirements.txt', style='blue')