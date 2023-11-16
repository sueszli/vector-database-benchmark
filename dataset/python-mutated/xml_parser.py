import re
import uuid
from datetime import datetime
from defusedxml import ElementTree
from dojo.models import Finding
from dojo.models import Endpoint
XML_NAMESPACE = {'x': 'https://www.veracode.com/schema/reports/export/1.0'}

class VeracodeXMLParser(object):
    """This parser is written for Veracode Detailed XML reports, version 1.5.

    Version is annotated in the report, `detailedreport/@report_format_version`.
    see https://help.veracode.com/r/t_download_XML_report
    """
    vc_severity_mapping = {1: 'Info', 2: 'Low', 3: 'Medium', 4: 'High', 5: 'Critical'}

    def get_findings(self, filename, test):
        if False:
            return 10
        root = ElementTree.parse(filename).getroot()
        app_id = root.attrib['app_id']
        report_date = datetime.strptime(root.attrib['last_update_time'], '%Y-%m-%d %H:%M:%S %Z')
        dupes = dict()
        for category_node in root.findall('x:severity/x:category', namespaces=XML_NAMESPACE):
            mitigation_text = ''
            mitigation_text += category_node.find('x:recommendations/x:para', namespaces=XML_NAMESPACE).get('text') + '\n\n'
            mitigation_text += ''.join(list(map(lambda x: '    * ' + x.get('text') + '\n', category_node.findall('x:recommendations/x:para/x:bulletitem', namespaces=XML_NAMESPACE))))
            for flaw_node in category_node.findall('x:cwe/x:staticflaws/x:flaw', namespaces=XML_NAMESPACE):
                dupe_key = flaw_node.attrib['issueid']
                if dupe_key not in dupes:
                    dupes[dupe_key] = self.__xml_static_flaw_to_finding(app_id, flaw_node, mitigation_text, test)
            for flaw_node in category_node.findall('x:cwe/x:dynamicflaws/x:flaw', namespaces=XML_NAMESPACE):
                dupe_key = flaw_node.attrib['issueid']
                if dupe_key not in dupes:
                    dupes[dupe_key] = self.__xml_dynamic_flaw_to_finding(app_id, flaw_node, mitigation_text, test)
        for component in root.findall('x:software_composition_analysis/x:vulnerable_components/x:component', namespaces=XML_NAMESPACE):
            _library = component.attrib['library']
            if 'library_id' in component.attrib and component.attrib['library_id'].startswith('maven:'):
                split_library_id = component.attrib['library_id'].split(':')
                if len(split_library_id) > 2:
                    _library = split_library_id[2]
            _vendor = component.attrib['vendor']
            _version = component.attrib['version']
            for vulnerability in component.findall('x:vulnerabilities/x:vulnerability', namespaces=XML_NAMESPACE):
                dupes[str(uuid.uuid4())] = self.__xml_sca_flaw_to_finding(test, report_date, _vendor, _library, _version, vulnerability)
        return list(dupes.values())

    @classmethod
    def __xml_flaw_to_unique_id(cls, app_id, xml_node):
        if False:
            print('Hello World!')
        issue_id = xml_node.attrib['issueid']
        return 'app-' + app_id + '_issue-' + issue_id

    @classmethod
    def __xml_flaw_to_severity(cls, xml_node):
        if False:
            print('Hello World!')
        return cls.vc_severity_mapping.get(int(xml_node.attrib['severity']), 'Info')

    @classmethod
    def __xml_flaw_to_finding(cls, app_id, xml_node, mitigation_text, test):
        if False:
            while True:
                i = 10
        finding = Finding()
        finding.test = test
        finding.mitigation = mitigation_text
        finding.static_finding = True
        finding.dynamic_finding = False
        finding.unique_id_from_tool = cls.__xml_flaw_to_unique_id(app_id, xml_node)
        finding.severity = cls.__xml_flaw_to_severity(xml_node)
        finding.cwe = int(xml_node.attrib['cweid'])
        finding.title = xml_node.attrib['categoryname']
        finding.impact = 'CIA Impact: ' + xml_node.attrib['cia_impact'].upper()
        _description = xml_node.attrib['description'].replace('. ', '.\n')
        finding.description = _description
        _references = 'None'
        if 'References:' in _description:
            _references = _description[_description.index('References:') + 13:].replace(')  ', ')\n')
        finding.references = _references + '\n\nVulnerable Module: ' + xml_node.attrib['module'] + '\nType: ' + xml_node.attrib['type'] + '\nVeracode issue ID: ' + xml_node.attrib['issueid']
        _date_found = test.target_start
        if 'date_first_occurrence' in xml_node.attrib:
            _date_found = datetime.strptime(xml_node.attrib['date_first_occurrence'], '%Y-%m-%d %H:%M:%S %Z')
        finding.date = _date_found
        _is_mitigated = False
        _mitigated_date = None
        if 'mitigation_status' in xml_node.attrib and xml_node.attrib['mitigation_status'].lower() == 'accepted':
            if 'remediation_status' in xml_node.attrib and xml_node.attrib['remediation_status'].lower() == 'fixed':
                _is_mitigated = True
            else:
                for mitigation in xml_node.findall('x:mitigations/x:mitigation', namespaces=XML_NAMESPACE):
                    _is_mitigated = True
                    _mitigated_date = datetime.strptime(mitigation.attrib['date'], '%Y-%m-%d %H:%M:%S %Z')
        finding.is_mitigated = _is_mitigated
        finding.mitigated = _mitigated_date
        finding.active = not _is_mitigated
        _false_positive = False
        if _is_mitigated:
            _remediation_status = xml_node.attrib['remediation_status'].lower()
            if 'false positive' in _remediation_status or 'falsepositive' in _remediation_status:
                _false_positive = True
        finding.false_p = _false_positive
        return finding

    @classmethod
    def __xml_static_flaw_to_finding(cls, app_id, xml_node, mitigation_text, test):
        if False:
            print('Hello World!')
        finding = cls.__xml_flaw_to_finding(app_id, xml_node, mitigation_text, test)
        finding.static_finding = True
        finding.dynamic_finding = False
        _line_number = xml_node.attrib['line']
        _functionrelativelocation = xml_node.attrib['functionrelativelocation']
        if _line_number is not None and _line_number.isdigit() and (_functionrelativelocation is not None) and _functionrelativelocation.isdigit():
            finding.line = int(_line_number) + int(_functionrelativelocation)
            finding.sast_source_line = finding.line
        _source_file = xml_node.attrib.get('sourcefile')
        _sourcefilepath = xml_node.attrib.get('sourcefilepath')
        finding.file_path = _sourcefilepath + _source_file
        finding.sast_source_file_path = _sourcefilepath + _source_file
        _sast_source_obj = xml_node.attrib.get('functionprototype')
        if isinstance(_sast_source_obj, str):
            finding.sast_source_object = _sast_source_obj if _sast_source_obj else None
        finding.unsaved_tags = ['sast']
        return finding

    @classmethod
    def __xml_dynamic_flaw_to_finding(cls, app_id, xml_node, mitigation_text, test):
        if False:
            for i in range(10):
                print('nop')
        finding = cls.__xml_flaw_to_finding(app_id, xml_node, mitigation_text, test)
        finding.static_finding = False
        finding.dynamic_finding = True
        url_host = xml_node.attrib.get('url')
        finding.unsaved_endpoints = [Endpoint.from_uri(url_host)]
        finding.unsaved_tags = ['dast']
        return finding

    @staticmethod
    def _get_cwe(val):
        if False:
            while True:
                i = 10
        cweSearch = re.search('CWE-(\\d+)', val, re.IGNORECASE)
        if cweSearch:
            return int(cweSearch.group(1))
        else:
            return None

    @classmethod
    def __xml_sca_flaw_to_finding(cls, test, report_date, vendor, library, version, xml_node):
        if False:
            i = 10
            return i + 15
        finding = Finding()
        finding.test = test
        finding.static_finding = True
        finding.dynamic_finding = False
        cvss_score = float(xml_node.attrib['cvss_score'])
        finding.cvssv3_score = cvss_score
        finding.severity = cls.__xml_flaw_to_severity(xml_node)
        finding.unsaved_vulnerability_ids = [xml_node.attrib['cve_id']]
        finding.cwe = cls._get_cwe(xml_node.attrib['cwe_id'])
        finding.title = 'Vulnerable component: {0}:{1}'.format(library, version)
        finding.component_name = library
        finding.component_version = version
        finding.date = report_date
        _description = 'This library has known vulnerabilities.\n'
        _description += '**CVE:** {0} ({1})\nCVS Score: {2} ({3})\nSummary: \n>{4}\n\n-----\n\n'.format(xml_node.attrib['cve_id'], xml_node.attrib.get('first_found_date'), xml_node.attrib['cvss_score'], cls.vc_severity_mapping.get(int(xml_node.attrib['severity']), 'Info'), xml_node.attrib['cve_summary'])
        finding.description = _description
        finding.unsaved_tags = ['sca']
        _is_mitigated = False
        _mitigated_date = None
        if 'mitigation' in xml_node.attrib and xml_node.attrib['mitigation'].lower() == 'true':
            for mitigation in xml_node.findall('x:mitigations/x:mitigation', namespaces=XML_NAMESPACE):
                _is_mitigated = True
                _mitigated_date = datetime.strptime(mitigation.attrib['date'], '%Y-%m-%d %H:%M:%S %Z')
        finding.is_mitigated = _is_mitigated
        finding.mitigated = _mitigated_date
        finding.active = not _is_mitigated
        return finding