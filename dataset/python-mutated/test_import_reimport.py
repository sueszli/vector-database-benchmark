import datetime
from django.urls import reverse
from dojo.models import Test_Type, User, Test, Finding
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient
from django.test.client import Client
from django.utils import timezone
from .dojo_test_case import DojoAPITestCase, get_unit_tests_path
from .test_utils import assertTestImportModelsCreated
from django.test import override_settings
import logging
logger = logging.getLogger(__name__)
ENGAGEMENT_NAME_DEFAULT = 'Engagement 1'
PRODUCT_NAME_DEFAULT = 'Product A'
PRODUCT_TYPE_NAME_DEFAULT = 'Type type'

class ImportReimportMixin(object):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.scans_path = '/scans/'
        self.zap_sample0_filename = self.scans_path + 'zap/0_zap_sample.xml'
        self.zap_sample1_filename = self.scans_path + 'zap/1_zap_sample_0_and_new_absent.xml'
        self.zap_sample2_filename = self.scans_path + 'zap/2_zap_sample_0_and_new_endpoint.xml'
        self.zap_sample3_filename = self.scans_path + 'zap/3_zap_sampl_0_and_different_severities.xml'
        self.anchore_file_name = self.scans_path + 'anchore_engine/one_vuln_many_files.json'
        self.scan_type_anchore = 'Anchore Engine Scan'
        self.acunetix_file_name = self.scans_path + 'acunetix/one_finding.xml'
        self.scan_type_acunetix = 'Acunetix Scan'
        self.gitlab_dep_scan_components_filename = f'{self.scans_path}gitlab_dep_scan/gl-dependency-scanning-report-many-vuln_v15.json'
        self.scan_type_gtlab_dep_scan = 'GitLab Dependency Scanning Report'
        self.sonarqube_file_name1 = self.scans_path + 'sonarqube/sonar-6-findings.html'
        self.sonarqube_file_name2 = self.scans_path + 'sonarqube/sonar-6-findings-1-unique_id_changed.html'
        self.scan_type_sonarqube_detailed = 'SonarQube Scan detailed'
        self.veracode_many_findings = self.scans_path + 'veracode/many_findings.xml'
        self.veracode_same_hash_code_different_unique_id = self.scans_path + 'veracode/many_findings_same_hash_code_different_unique_id.xml'
        self.veracode_same_unique_id_different_hash_code = self.scans_path + 'veracode/many_findings_same_unique_id_different_hash_code.xml'
        self.veracode_different_hash_code_different_unique_id = self.scans_path + 'veracode/many_findings_different_hash_code_different_unique_id.xml'
        self.veracode_mitigated_findings = self.scans_path + 'veracode/mitigated_finding.xml'
        self.scan_type_veracode = 'Veracode Scan'
        self.clair_few_findings = self.scans_path + 'clair/few_vuln.json'
        self.clair_empty = self.scans_path + 'clair/empty.json'
        self.scan_type_clair = 'Clair Scan'
        self.generic_filename_with_file = self.scans_path + 'generic/test_with_image.json'
        self.aws_prowler_file_name = self.scans_path + 'aws_prowler/many_vuln.json'
        self.aws_prowler_file_name_plus_one = self.scans_path + 'aws_prowler/many_vuln_plus_one.json'
        self.scan_type_aws_prowler = 'AWS Prowler Scan'
        self.nuclei_empty = self.scans_path + 'nuclei/empty.jsonl'
        self.gitlab_dast_file_name = f'{self.scans_path}gitlab_dast/gitlab_dast_one_vul_v15.json'
        self.scan_type_gitlab_dast = 'GitLab DAST Report'
        self.anchore_grype_file_name = self.scans_path + 'anchore_grype/check_all_fields.json'
        self.anchore_grype_scan_type = 'Anchore Grype'

    def test_zap_scan_base_active_verified(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('importing original zap xml report')
        endpoint_count_before = self.db_endpoint_count()
        endpoint_status_count_before_active = self.db_endpoint_status_count(mitigated=False)
        endpoint_status_count_before_mitigated = self.db_endpoint_status_count(mitigated=True)
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.import_scan_with_params(self.zap_sample0_filename)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(4, findings)
        self.assertEqual(endpoint_count_before + 2, self.db_endpoint_count())
        self.assertEqual(endpoint_status_count_before_active + 7, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated, self.db_endpoint_status_count(mitigated=True))
        self.assertEqual(notes_count_before, self.db_notes_count())
        return test_id

    def test_zap_scan_base_not_active_not_verified(self):
        if False:
            while True:
                i = 10
        logger.debug('importing original zap xml report')
        endpoint_count_before = self.db_endpoint_count()
        endpoint_status_count_before_active = self.db_endpoint_status_count(mitigated=False)
        endpoint_status_count_before_mitigated = self.db_endpoint_status_count(mitigated=True)
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.import_scan_with_params(self.zap_sample0_filename, active=False, verified=False)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id, active=False, verified=False)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(4, findings)
        self.assertEqual(endpoint_count_before + 2, self.db_endpoint_count())
        self.assertEqual(endpoint_status_count_before_active + 7, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated, self.db_endpoint_status_count(mitigated=True))
        self.assertEqual(notes_count_before, self.db_notes_count())
        return test_id

    def test_import_default_scan_date_parser_not_sets_date(self):
        if False:
            while True:
                i = 10
        logger.debug('importing zap xml report with date set by parser')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.import_scan_with_params(self.zap_sample0_filename, active=False, verified=False)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id, active=False, verified=False)
        self.log_finding_summary_json_api(findings)
        date = findings['results'][0]['date']
        self.assertEqual(date, str(timezone.localtime(timezone.now()).date()))
        return test_id

    def test_import_default_scan_date_parser_sets_date(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('importing original acunetix xml report')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=1, created=1):
            import0 = self.import_scan_with_params(self.acunetix_file_name, scan_type=self.scan_type_acunetix, active=False, verified=False)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id, active=False, verified=False)
        self.log_finding_summary_json_api(findings)
        date = findings['results'][0]['date']
        self.assertEqual(date, '2018-09-24')
        return test_id

    def test_import_set_scan_date_parser_not_sets_date(self):
        if False:
            i = 10
            return i + 15
        logger.debug('importing original zap xml report')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.import_scan_with_params(self.zap_sample0_filename, active=False, verified=False, scan_date='2006-12-26')
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id, active=False, verified=False)
        self.log_finding_summary_json_api(findings)
        date = findings['results'][0]['date']
        self.assertEqual(date, '2006-12-26')
        return test_id

    def test_import_set_scan_date_parser_sets_date(self):
        if False:
            i = 10
            return i + 15
        logger.debug('importing acunetix xml report with date set by parser')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=1, created=1):
            import0 = self.import_scan_with_params(self.acunetix_file_name, scan_type=self.scan_type_acunetix, active=False, verified=False, scan_date='2006-12-26')
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id, active=False, verified=False)
        self.log_finding_summary_json_api(findings)
        date = findings['results'][0]['date']
        self.assertEqual(date, '2006-12-26')
        return test_id

    def test_import_reimport_no_scan_date_parser_no_date(self):
        if False:
            while True:
                i = 10
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        test_id = import0['test']
        reimport0 = self.reimport_scan_with_params(test_id, self.zap_sample1_filename)
        test_id = reimport0['test']
        findings = self.get_test_findings_api(test_id)
        self.assert_finding_count_json(5, findings)
        self.assertEqual(findings['results'][4]['date'], str(timezone.localtime(timezone.now()).date()))

    def test_import_reimport_scan_date_parser_no_date(self):
        if False:
            while True:
                i = 10
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        test_id = import0['test']
        reimport0 = self.reimport_scan_with_params(test_id, self.zap_sample1_filename, scan_date='2020-02-02')
        test_id = reimport0['test']
        findings = self.get_test_findings_api(test_id)
        self.assert_finding_count_json(5, findings)
        self.assertEqual(findings['results'][4]['date'], '2020-02-02')

    def test_import_reimport_no_scan_date_parser_date(self):
        if False:
            for i in range(10):
                print('nop')
        import0 = self.import_scan_with_params(self.aws_prowler_file_name, scan_type=self.scan_type_aws_prowler)
        test_id = import0['test']
        reimport0 = self.reimport_scan_with_params(test_id, self.aws_prowler_file_name_plus_one, scan_type=self.scan_type_aws_prowler)
        test_id = reimport0['test']
        findings = self.get_test_findings_api(test_id)
        self.assert_finding_count_json(5, findings)
        self.log_finding_summary_json_api(findings)
        self.assertEqual(findings['results'][2]['date'], '2021-08-23')

    def test_import_reimport_scan_date_parser_date(self):
        if False:
            i = 10
            return i + 15
        import0 = self.import_scan_with_params(self.aws_prowler_file_name, scan_type=self.scan_type_aws_prowler)
        test_id = import0['test']
        reimport0 = self.reimport_scan_with_params(test_id, self.aws_prowler_file_name_plus_one, scan_type=self.scan_type_aws_prowler, scan_date='2020-02-02')
        test_id = reimport0['test']
        findings = self.get_test_findings_api(test_id)
        self.assert_finding_count_json(5, findings)
        self.log_finding_summary_json_api(findings)
        self.assertEqual(findings['results'][2]['date'], '2020-02-02')

    def test_sonar_detailed_scan_base_active_verified(self):
        if False:
            i = 10
            return i + 15
        logger.debug('importing original sonar report')
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, imports=1, affected_findings=6, created=6):
            import0 = self.import_scan_with_params(self.sonarqube_file_name1, scan_type=self.scan_type_sonarqube_detailed)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(6, findings)
        self.assertEqual(notes_count_before, self.db_notes_count())
        return test_id

    def test_veracode_scan_base_active_verified(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('importing original veracode report')
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.import_scan_with_params(self.veracode_many_findings, scan_type=self.scan_type_veracode)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(4, findings)
        self.assertEqual(notes_count_before, self.db_notes_count())
        return test_id

    def test_import_veracode_reimport_veracode_active_verified_mitigated(self):
        if False:
            print('Hello World!')
        logger.debug('reimporting exact same original veracode mitigated xml report again')
        import_veracode_many_findings = self.import_scan_with_params(self.veracode_mitigated_findings, scan_type=self.scan_type_veracode, verified=True, forceActive=True, forceVerified=True)
        test_id = import_veracode_many_findings['test']
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=1, created=0, closed=1, reactivated=0, untouched=0):
            reimport_veracode_mitigated_findings = self.reimport_scan_with_params(test_id, self.veracode_mitigated_findings, scan_type=self.scan_type_veracode)
        test_id = reimport_veracode_mitigated_findings['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(1, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(0, findings)
        self.assertEqual(notes_count_before, self.db_notes_count() - 1)
        mitigated_findings = self.get_test_findings_api(test_id, is_mitigated=True)
        self.assert_finding_count_json(1, mitigated_findings)

    def test_import_0_reimport_0_active_verified(self):
        if False:
            i = 10
            return i + 15
        logger.debug('reimporting exact same original zap xml report again')
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        test_id = import0['test']
        endpoint_count_before = self.db_endpoint_count()
        endpoint_status_count_before_active = self.db_endpoint_status_count(mitigated=False)
        endpoint_status_count_before_mitigated = self.db_endpoint_status_count(mitigated=True)
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, untouched=4):
            reimport0 = self.reimport_scan_with_params(test_id, self.zap_sample0_filename)
        test_id = reimport0['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id)
        self.assert_finding_count_json(4, findings)
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(endpoint_status_count_before_active, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated, self.db_endpoint_status_count(mitigated=True))
        self.assertEqual(notes_count_before, self.db_notes_count())

    def test_import_0_reimport_0_active_not_verified(self):
        if False:
            while True:
                i = 10
        logger.debug('reimporting exact same original zap xml report again, verified=False')
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        test_id = import0['test']
        endpoint_count_before = self.db_endpoint_count()
        endpoint_status_count_before_active = self.db_endpoint_status_count(mitigated=False)
        endpoint_status_count_before_mitigated = self.db_endpoint_status_count(mitigated=True)
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, untouched=4):
            reimport0 = self.reimport_scan_with_params(test_id, self.zap_sample0_filename, verified=False)
        test_id = reimport0['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(4, findings)
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(endpoint_status_count_before_active, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated, self.db_endpoint_status_count(mitigated=True))
        self.assertEqual(notes_count_before, self.db_notes_count())

    def test_import_sonar1_reimport_sonar1_active_not_verified(self):
        if False:
            return 10
        logger.debug('reimporting exact same original sonar report again, verified=False')
        importsonar1 = self.import_scan_with_params(self.sonarqube_file_name1, scan_type=self.scan_type_sonarqube_detailed)
        test_id = importsonar1['test']
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, untouched=6):
            reimportsonar1 = self.reimport_scan_with_params(test_id, self.sonarqube_file_name1, scan_type=self.scan_type_sonarqube_detailed, verified=False)
        test_id = reimportsonar1['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(6, findings)
        self.assertEqual(notes_count_before, self.db_notes_count())

    def test_import_veracode_reimport_veracode_active_not_verified(self):
        if False:
            print('Hello World!')
        logger.debug('reimporting exact same original veracode report again, verified=False')
        import_veracode_many_findings = self.import_scan_with_params(self.veracode_many_findings, scan_type=self.scan_type_veracode)
        test_id = import_veracode_many_findings['test']
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, untouched=4):
            reimport_veracode_many_findings = self.reimport_scan_with_params(test_id, self.veracode_many_findings, scan_type=self.scan_type_veracode, verified=False)
        test_id = reimport_veracode_many_findings['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(4, findings)
        self.assertEqual(notes_count_before, self.db_notes_count())

    def test_import_sonar1_reimport_sonar2(self):
        if False:
            print('Hello World!')
        logger.debug('reimporting same findings except one with a different unique_id_from_tool')
        importsonar1 = self.import_scan_with_params(self.sonarqube_file_name1, scan_type=self.scan_type_sonarqube_detailed)
        test_id = importsonar1['test']
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=2, created=1, closed=1, untouched=5):
            reimportsonar1 = self.reimport_scan_with_params(test_id, self.sonarqube_file_name2, scan_type=self.scan_type_sonarqube_detailed, verified=False)
        test_id = reimportsonar1['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0, findings)
        findings = self.get_test_findings_api(test_id, is_mitigated=True)
        self.assert_finding_count_json(1, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(7, findings)
        self.assertEqual(notes_count_before + 1, self.db_notes_count())

    def test_import_veracode_reimport_veracode_same_hash_code_different_unique_id(self):
        if False:
            i = 10
            return i + 15
        logger.debug('reimporting report with one finding having same hash_code but different unique_id_from_tool, verified=False')
        import_veracode_many_findings = self.import_scan_with_params(self.veracode_many_findings, scan_type=self.scan_type_veracode)
        test_id = import_veracode_many_findings['test']
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, untouched=4):
            reimport_veracode_many_findings = self.reimport_scan_with_params(test_id, self.veracode_same_hash_code_different_unique_id, scan_type=self.scan_type_veracode, verified=False)
        test_id = reimport_veracode_many_findings['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(4, findings)
        self.assertEqual(notes_count_before, self.db_notes_count())

    def test_import_veracode_reimport_veracode_same_unique_id_different_hash_code(self):
        if False:
            while True:
                i = 10
        logger.debug('reimporting report with one finding having same unique_id_from_tool but different hash_code, verified=False')
        import_veracode_many_findings = self.import_scan_with_params(self.veracode_many_findings, scan_type=self.scan_type_veracode)
        test_id = import_veracode_many_findings['test']
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, untouched=4):
            reimport_veracode_many_findings = self.reimport_scan_with_params(test_id, self.veracode_same_unique_id_different_hash_code, scan_type=self.scan_type_veracode, verified=False)
        test_id = reimport_veracode_many_findings['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(4, findings)
        self.assertEqual(notes_count_before, self.db_notes_count())

    def test_import_veracode_reimport_veracode_different_hash_code_different_unique_id(self):
        if False:
            print('Hello World!')
        logger.debug('reimporting report with one finding having different hash_code and different unique_id_from_tool, verified=False')
        import_veracode_many_findings = self.import_scan_with_params(self.veracode_many_findings, scan_type=self.scan_type_veracode)
        test_id = import_veracode_many_findings['test']
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=2, created=1, closed=1, untouched=3):
            reimport_veracode_many_findings = self.reimport_scan_with_params(test_id, self.veracode_different_hash_code_different_unique_id, scan_type=self.scan_type_veracode, verified=False)
        test_id = reimport_veracode_many_findings['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(5, findings)
        self.assertEqual(notes_count_before + 1, self.db_notes_count())

    def test_import_0_reimport_1_active_not_verified(self):
        if False:
            while True:
                i = 10
        logger.debug('reimporting updated zap xml report, 1 new finding and 1 no longer present, verified=False')
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        finding_count_before = self.db_finding_count()
        endpoint_count_before = self.db_endpoint_count()
        endpoint_status_count_before_active = self.db_endpoint_status_count(mitigated=False)
        endpoint_status_count_before_mitigated = self.db_endpoint_status_count(mitigated=True)
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=2, created=1, closed=1, untouched=3):
            reimport1 = self.reimport_scan_with_params(test_id, self.zap_sample1_filename, verified=False)
        test_id = reimport1['test']
        self.assertEqual(test_id, test_id)
        test = self.get_test_api(test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id)
        self.assert_finding_count_json(4 + 1, findings)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(5, findings)
        self.assertEqual(finding_count_before + 1, self.db_finding_count())
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(endpoint_status_count_before_active + 2 - 3, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated + 2, self.db_endpoint_status_count(mitigated=True))
        self.assertEqual(notes_count_before + 1, self.db_notes_count())

    def test_import_0_reimport_1_active_verified_reimport_0_active_verified(self):
        if False:
            while True:
                i = 10
        logger.debug('reimporting updated zap xml report, 1 new finding and 1 no longer present, verified=True and then 0 again')
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        finding_count_before = self.db_finding_count()
        endpoint_count_before = self.db_endpoint_count()
        endpoint_status_count_before_active = self.db_endpoint_status_count(mitigated=False)
        endpoint_status_count_before_mitigated = self.db_endpoint_status_count(mitigated=True)
        notes_count_before = self.db_notes_count()
        reimport1 = self.reimport_scan_with_params(test_id, self.zap_sample1_filename)
        self.assertEqual(endpoint_status_count_before_active - 3 + 2, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated + 2, self.db_endpoint_status_count(mitigated=True))
        endpoint_status_count_before_active = self.db_endpoint_status_count(mitigated=False)
        endpoint_status_count_before_mitigated = self.db_endpoint_status_count(mitigated=True)
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=2, closed=1, reactivated=1, untouched=3):
            reimport0 = self.reimport_scan_with_params(test_id, self.zap_sample0_filename)
        test_id = reimport1['test']
        self.assertEqual(test_id, test_id)
        test = self.get_test_api(test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id)
        self.assert_finding_count_json(4 + 1, findings)
        zap1_ok = False
        zap4_ok = False
        for finding in findings['results']:
            if 'Zap1' in finding['title']:
                self.assertTrue(finding['active'])
                zap1_ok = True
            if 'Zap4' in finding['title']:
                self.assertFalse(finding['active'])
                zap4_ok = True
        self.assertTrue(zap1_ok)
        self.assertTrue(zap4_ok)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(5, findings)
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(endpoint_status_count_before_active + 3 - 2, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated - 3 + 2, self.db_endpoint_status_count(mitigated=True))
        self.assertEqual(notes_count_before + 2 + 1, self.db_notes_count())

    def test_import_0_reimport_2_extra_endpoint(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('reimporting exact same original zap xml report again, with an extra endpoint for zap1')
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        finding_count_before = self.db_finding_count()
        endpoint_count_before = self.db_endpoint_count()
        endpoint_status_count_before_active = self.db_endpoint_status_count(mitigated=False)
        endpoint_status_count_before_mitigated = self.db_endpoint_status_count(mitigated=True)
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=0, untouched=4):
            reimport2 = self.reimport_scan_with_params(test_id, self.zap_sample2_filename)
        test_id = reimport2['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id)
        self.assert_finding_count_json(4, findings)
        self.assertEqual(endpoint_count_before + 1, self.db_endpoint_count())
        self.assertEqual(endpoint_status_count_before_active + 1, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated, self.db_endpoint_status_count(mitigated=True))
        self.assertEqual(notes_count_before, self.db_notes_count())
        self.assertEqual(finding_count_before, self.db_finding_count())

    def test_import_0_reimport_2_extra_endpoint_reimport_0(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('reimporting exact same original zap xml report again, with an extra endpoint for zap1')
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=0, untouched=4):
            reimport2 = self.reimport_scan_with_params(test_id, self.zap_sample2_filename)
        test_id = reimport2['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        finding_count_before = self.db_finding_count()
        endpoint_count_before = self.db_endpoint_count()
        endpoint_status_count_before_active = self.db_endpoint_status_count(mitigated=False)
        endpoint_status_count_before_mitigated = self.db_endpoint_status_count(mitigated=True)
        notes_count_before = self.db_notes_count()
        reimport0 = self.reimport_scan_with_params(test_id, self.zap_sample0_filename)
        test_id = reimport0['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id)
        self.assert_finding_count_json(4, findings)
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(endpoint_status_count_before_active - 1, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated + 1, self.db_endpoint_status_count(mitigated=True))
        self.assertEqual(notes_count_before, self.db_notes_count())
        self.assertEqual(finding_count_before, self.db_finding_count())

    def test_import_0_reimport_3_active_verified(self):
        if False:
            print('Hello World!')
        logger.debug('reimporting updated zap xml report, with different severities for zap2 and zap5')
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        finding_count_before = self.db_finding_count()
        endpoint_count_before = self.db_endpoint_count()
        endpoint_status_count_before_active = self.db_endpoint_status_count(mitigated=False)
        endpoint_status_count_before_mitigated = self.db_endpoint_status_count(mitigated=True)
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=4, created=2, closed=2, untouched=2):
            reimport1 = self.reimport_scan_with_params(test_id, self.zap_sample3_filename)
        test_id = reimport1['test']
        self.assertEqual(test_id, test_id)
        test = self.get_test_api(test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(4 + 2, findings)
        zap2_ok = False
        zap5_ok = False
        for finding in findings['results']:
            if 'Zap2' in finding['title']:
                self.assertTrue(finding['active'] or finding['severity'] == 'Low')
                self.assertTrue(not finding['active'] or finding['severity'] == 'Medium')
                zap2_ok = True
            if 'Zap5' in finding['title']:
                self.assertTrue(finding['active'] or finding['severity'] == 'Low')
                self.assertTrue(not finding['active'] or finding['severity'] == 'Medium')
                zap5_ok = True
        self.assertTrue(zap2_ok)
        self.assertTrue(zap5_ok)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0 + 0, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(4 + 2, findings)
        self.assertEqual(finding_count_before + 2, self.db_finding_count())
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(endpoint_status_count_before_active + 3 + 3 - 3 - 3, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated + 2 + 2, self.db_endpoint_status_count(mitigated=True))
        self.assertEqual(notes_count_before + 2, self.db_notes_count())

    def test_import_reimport_without_closing_old_findings(self):
        if False:
            while True:
                i = 10
        logger.debug('reimporting updated zap xml report and keep old findings open')
        import1 = self.import_scan_with_params(self.zap_sample1_filename)
        test_id = import1['test']
        findings = self.get_test_findings_api(test_id)
        self.assert_finding_count_json(4, findings)
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=1, created=1, untouched=3):
            reimport1 = self.reimport_scan_with_params(test_id, self.zap_sample2_filename, close_old_findings=False)
        test_id = reimport1['test']
        self.assertEqual(test_id, test_id)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(5, findings)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0, findings)
        mitigated = 0
        not_mitigated = 0
        for finding in findings['results']:
            logger.debug(finding)
            if finding['is_mitigated']:
                mitigated += 1
            else:
                not_mitigated += 1
        self.assertEqual(mitigated, 0)
        self.assertEqual(not_mitigated, 0)

    def test_import_0_reimport_0_anchore_file_path(self):
        if False:
            return 10
        import0 = self.import_scan_with_params(self.anchore_file_name, scan_type=self.scan_type_anchore)
        test_id = import0['test']
        active_findings_before = self.get_test_findings_api(test_id, active=True)
        self.log_finding_summary_json_api(active_findings_before)
        active_findings_count_before = active_findings_before['count']
        notes_count_before = self.db_notes_count()
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=0, untouched=4):
            reimport0 = self.reimport_scan_with_params(test_id, self.anchore_file_name, scan_type=self.scan_type_anchore)
        active_findings_after = self.get_test_findings_api(test_id, active=True)
        self.log_finding_summary_json_api(active_findings_after)
        self.assert_finding_count_json(active_findings_count_before, active_findings_after)
        self.assertEqual(notes_count_before, self.db_notes_count())

    def test_import_reimport_keep_false_positive_and_out_of_scope(self):
        if False:
            print('Hello World!')
        logger.debug('importing zap0 with 4 findings, manually setting 3 findings to active=False, reimporting zap0 must return only 1 finding active=True')
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        test_id = import0['test']
        test_api_response = self.get_test_api(test_id)
        product_api_response = self.get_engagement_api(test_api_response['engagement'])
        product_id = product_api_response['product']
        self.patch_product_api(product_id, {'enable_simple_risk_acceptance': True})
        active_findings_before = self.get_test_findings_api(test_id, active=True)
        self.assert_finding_count_json(4, active_findings_before)
        for finding in active_findings_before['results']:
            if 'Zap1' in finding['title']:
                self.patch_finding_api(finding['id'], {'active': False, 'verified': False, 'false_p': True, 'out_of_scope': False, 'risk_accepted': False, 'is_mitigated': True})
            elif 'Zap2' in finding['title']:
                self.patch_finding_api(finding['id'], {'active': False, 'verified': False, 'false_p': False, 'out_of_scope': True, 'risk_accepted': False, 'is_mitigated': True})
            elif 'Zap3' in finding['title']:
                self.patch_finding_api(finding['id'], {'active': False, 'verified': False, 'false_p': False, 'out_of_scope': False, 'risk_accepted': True, 'is_mitigated': True})
        active_findings_before = self.get_test_findings_api(test_id, active=True)
        self.assert_finding_count_json(1, active_findings_before)
        for finding in active_findings_before['results']:
            if 'Zap5' in finding['title']:
                self.delete_finding_api(finding['id'])
        active_findings_before = self.get_test_findings_api(test_id, active=True)
        self.assert_finding_count_json(0, active_findings_before)
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=1, created=1):
            reimport0 = self.reimport_scan_with_params(test_id, self.zap_sample0_filename)
        self.assertEqual(reimport0['test'], test_id)
        active_findings_after = self.get_test_findings_api(test_id, active=True)
        self.assert_finding_count_json(1, active_findings_after)
        active_findings_after = self.get_test_findings_api(test_id, active=False)
        self.assert_finding_count_json(3, active_findings_after)
        for finding in active_findings_after['results']:
            if 'Zap1' in finding['title']:
                self.assertFalse(finding['active'])
                self.assertFalse(finding['verified'])
                self.assertTrue(finding['false_p'])
                self.assertFalse(finding['out_of_scope'])
                self.assertFalse(finding['risk_accepted'])
                self.assertTrue(finding['is_mitigated'])
            elif 'Zap2' in finding['title']:
                self.assertFalse(finding['active'])
                self.assertFalse(finding['verified'])
                self.assertFalse(finding['false_p'])
                self.assertTrue(finding['out_of_scope'])
                self.assertFalse(finding['risk_accepted'])
                self.assertTrue(finding['is_mitigated'])
            elif 'Zap3' in finding['title']:
                self.assertFalse(finding['active'])
                self.assertFalse(finding['verified'])
                self.assertFalse(finding['false_p'])
                self.assertFalse(finding['out_of_scope'])
                self.assertTrue(finding['risk_accepted'])
                self.assertTrue(finding['is_mitigated'])
            elif 'Zap5' in finding['title']:
                self.assertTrue(finding['active'])
                self.assertTrue(finding['verified'])
                self.assertFalse(finding['false_p'])
                self.assertFalse(finding['out_of_scope'])
                self.assertFalse(finding['risk_accepted'])
                self.assertFalse(finding['is_mitigated'])

    def test_import_6_reimport_6_gitlab_dep_scan_component_name_and_version(self):
        if False:
            print('Hello World!')
        import0 = self.import_scan_with_params(self.gitlab_dep_scan_components_filename, scan_type=self.scan_type_gtlab_dep_scan, minimum_severity='Info')
        test_id = import0['test']
        active_findings_before = self.get_test_findings_api(test_id, active=True)
        self.assert_finding_count_json(6, active_findings_before)
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=0, created=0, untouched=6):
            reimport0 = self.reimport_scan_with_params(test_id, self.gitlab_dep_scan_components_filename, scan_type=self.scan_type_gtlab_dep_scan, minimum_severity='Info')
        active_findings_after = self.get_test_findings_api(test_id, active=True)
        self.assert_finding_count_json(6, active_findings_after)
        count = 0
        for finding in active_findings_after['results']:
            if 'v0.0.0-20190219172222-a4c6cb3142f2' == finding['component_version']:
                self.assertEqual('CVE-2020-29652: Nil Pointer Dereference', finding['title'])
                self.assertEqual('CVE-2020-29652', finding['vulnerability_ids'][0]['vulnerability_id'])
                self.assertEqual('golang.org/x/crypto', finding['component_name'])
                count = count + 1
            elif 'v0.0.0-20190308221718-c2843e01d9a2' == finding['component_version']:
                self.assertEqual('CVE-2020-29652: Nil Pointer Dereference', finding['title'])
                self.assertEqual('CVE-2020-29652', finding['vulnerability_ids'][0]['vulnerability_id'])
                self.assertEqual('golang.org/x/crypto', finding['component_name'])
                count = count + 1
            elif 'v0.0.0-20200302210943-78000ba7a073' == finding['component_version']:
                self.assertEqual('CVE-2020-29652: Nil Pointer Dereference', finding['title'])
                self.assertEqual('CVE-2020-29652', finding['vulnerability_ids'][0]['vulnerability_id'])
                self.assertEqual('golang.org/x/crypto', finding['component_name'])
                count = count + 1
            elif 'v0.3.0' == finding['component_version']:
                self.assertEqual('CVE-2020-14040: Loop With Unreachable Exit Condition (Infinite Loop)', finding['title'])
                self.assertEqual('CVE-2020-14040', finding['vulnerability_ids'][0]['vulnerability_id'])
                self.assertEqual('golang.org/x/text', finding['component_name'])
                count = count + 1
            elif 'v0.3.2' == finding['component_version']:
                self.assertEqual('CVE-2020-14040: Loop With Unreachable Exit Condition (Infinite Loop)', finding['title'])
                self.assertEqual('CVE-2020-14040', finding['vulnerability_ids'][0]['vulnerability_id'])
                self.assertEqual('golang.org/x/text', finding['component_name'])
                count = count + 1
        self.assertEqual(5, count)

    def test_import_param_close_old_findings_with_additional_endpoint(self):
        if False:
            i = 10
            return i + 15
        logger.debug('importing clair report with additional endpoint')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.import_scan_with_params(self.clair_few_findings, scan_type=self.scan_type_clair, close_old_findings=True, endpoint_to_add=1)
        test_id = import0['test']
        test = self.get_test(test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(4, findings)
        engagement_findings = Finding.objects.filter(test__engagement_id=1, test__test_type=test.test_type, active=True, is_mitigated=False)
        self.assertEqual(engagement_findings.count(), 4)
        for finding in engagement_findings:
            self.assertEqual(finding.endpoints.count(), 1)
            self.assertEqual(finding.endpoints.first().id, 1)
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, closed=4):
            self.import_scan_with_params(self.clair_empty, scan_type=self.scan_type_clair, close_old_findings=True, endpoint_to_add=1)
        engagement_findings_count = Finding.objects.filter(test__engagement_id=1, test__test_type=test.test_type, active=True, is_mitigated=False).count()
        self.assertEqual(engagement_findings_count, 0)

    def test_import_param_close_old_findings_with_same_service(self):
        if False:
            print('Hello World!')
        logger.debug('importing clair report with same service')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.import_scan_with_params(self.clair_few_findings, scan_type=self.scan_type_clair, close_old_findings=True, service='service_1')
        test_id = import0['test']
        test = self.get_test(test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(4, findings)
        engagement_findings = Finding.objects.filter(test__engagement_id=1, test__test_type=test.test_type, active=True, is_mitigated=False)
        self.assertEqual(engagement_findings.count(), 4)
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, closed=4):
            self.import_scan_with_params(self.clair_empty, scan_type=self.scan_type_clair, close_old_findings=True, service='service_1')
        engagement_findings_count = Finding.objects.filter(test__engagement_id=1, test__test_type=test.test_type, active=True, is_mitigated=False).count()
        self.assertEqual(engagement_findings_count, 0)

    def test_import_param_close_old_findings_with_different_services(self):
        if False:
            while True:
                i = 10
        logger.debug('importing clair report with different services')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.import_scan_with_params(self.clair_few_findings, scan_type=self.scan_type_clair, close_old_findings=True, service='service_1')
        test_id = import0['test']
        test = self.get_test(test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(4, findings)
        engagement_findings = Finding.objects.filter(test__engagement_id=1, test__test_type=test.test_type, active=True, is_mitigated=False)
        self.assertEqual(engagement_findings.count(), 4)
        with assertTestImportModelsCreated(self, imports=1, affected_findings=0, closed=0):
            self.import_scan_with_params(self.clair_empty, scan_type=self.scan_type_clair, close_old_findings=True, service='service_2')
        engagement_findings_count = Finding.objects.filter(test__engagement_id=1, test__test_type=test.test_type, active=True, is_mitigated=False).count()
        self.assertEqual(engagement_findings_count, 4)

    def test_import_param_close_old_findings_with_and_without_service_1(self):
        if False:
            print('Hello World!')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.import_scan_with_params(self.clair_few_findings, scan_type=self.scan_type_clair, close_old_findings=True, service='service_1')
        test_id = import0['test']
        test = self.get_test(test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(4, findings)
        engagement_findings = Finding.objects.filter(test__engagement_id=1, test__test_type=test.test_type, active=True, is_mitigated=False)
        self.assertEqual(engagement_findings.count(), 4)
        with assertTestImportModelsCreated(self, imports=1, affected_findings=0, closed=0):
            self.import_scan_with_params(self.clair_empty, scan_type=self.scan_type_clair, close_old_findings=True, service=None)
        engagement_findings_count = Finding.objects.filter(test__engagement_id=1, test__test_type=test.test_type, active=True, is_mitigated=False).count()
        self.assertEqual(engagement_findings_count, 4)

    def test_import_param_close_old_findings_with_and_without_service_2(self):
        if False:
            return 10
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.import_scan_with_params(self.clair_few_findings, scan_type=self.scan_type_clair, close_old_findings=True, service=None)
        test_id = import0['test']
        test = self.get_test(test_id)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(4, findings)
        engagement_findings = Finding.objects.filter(test__engagement_id=1, test__test_type=test.test_type, active=True, is_mitigated=False)
        self.assertEqual(engagement_findings.count(), 4)
        with assertTestImportModelsCreated(self, imports=1, affected_findings=0, closed=0):
            self.import_scan_with_params(self.clair_empty, scan_type=self.scan_type_clair, close_old_findings=True, service='service_2')
        engagement_findings_count = Finding.objects.filter(test__engagement_id=1, test__test_type=test.test_type, active=True, is_mitigated=False).count()
        self.assertEqual(engagement_findings_count, 4)

    def test_import_reimport_generic(self):
        if False:
            return 10
        'This test do a basic import and re-import of a generic JSON report\n\n        This test is useful because some features are only activated in generic JSON format\n        '
        import0 = self.import_scan_with_params(self.generic_filename_with_file, scan_type='Generic Findings Import')
        test_id = import0['test']
        with assertTestImportModelsCreated(self, reimports=1, untouched=1):
            reimport0 = self.reimport_scan_with_params(test_id, self.generic_filename_with_file, scan_type='Generic Findings Import')
        test_id2 = reimport0['test']
        self.assertEqual(test_id, test_id2)
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        findings = self.get_test_findings_api(test_id, verified=True)
        self.assert_finding_count_json(0, findings)
        findings = self.get_test_findings_api(test_id, verified=False)
        self.assert_finding_count_json(1, findings)

    def test_import_nuclei_emptyc(self):
        if False:
            i = 10
            return i + 15
        'This test do a basic import of Nuclei report with no vulnerability\n\n        This test is useful because Nuclei use jsonl for his format so it can generate empty files.\n        It tests the condition limit of loading an empty file.\n        '
        import0 = self.import_scan_with_params(self.nuclei_empty, scan_type='Nuclei Scan')
        test_id = import0['test']
        reimport0 = self.reimport_scan_with_params(test_id, self.nuclei_empty, scan_type='Nuclei Scan')
        test_id2 = reimport0['test']
        self.assertEqual(test_id, test_id2)

    def test_import_reimport_endpoint_where_eps_date_is_different(self):
        if False:
            print('Hello World!')
        endpoint_count_before = self.db_endpoint_count()
        endpoint_status_count_before_active = self.db_endpoint_status_count(mitigated=False)
        endpoint_status_count_before_mitigated = self.db_endpoint_status_count(mitigated=True)
        with assertTestImportModelsCreated(self, imports=1, affected_findings=1, created=1):
            import0 = self.import_scan_with_params(self.gitlab_dast_file_name, self.scan_type_gitlab_dast, active=True, verified=True)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(1, findings)
        test = self.get_test_api(test_id)['id']
        finding = Finding.objects.filter(test__engagement_id=1, test=test).first()
        self.assertEqual(finding.status_finding.count(), 1)
        original_date = finding.status_finding.first().date
        self.assertEqual(endpoint_count_before + 1, self.db_endpoint_count())
        self.assertEqual(endpoint_status_count_before_active + 1, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated, self.db_endpoint_status_count(mitigated=True))
        reimport0 = self.reimport_scan_with_params(test_id, self.gitlab_dast_file_name, scan_type=self.scan_type_gitlab_dast)
        test_id = reimport0['test']
        findings = self.get_test_findings_api(test_id)
        self.log_finding_summary_json_api(findings)
        self.assert_finding_count_json(1, findings)
        finding = Finding.objects.filter(test__engagement_id=1, test=test).first()
        self.assertEqual(finding.status_finding.count(), 1)
        reimported_date = finding.status_finding.first().date
        self.assertEqual(original_date, reimported_date)
        self.assertEqual(endpoint_count_before + 1, self.db_endpoint_count())
        self.assertEqual(endpoint_status_count_before_active + 1, self.db_endpoint_status_count(mitigated=False))
        self.assertEqual(endpoint_status_count_before_mitigated, self.db_endpoint_status_count(mitigated=True))

    def test_import_reimport_vulnerability_ids(self):
        if False:
            return 10
        import0 = self.import_scan_with_params(self.anchore_grype_file_name, scan_type=self.anchore_grype_scan_type)
        test_id = import0['test']
        test = Test.objects.get(id=test_id)
        findings = Finding.objects.filter(test=test)
        self.assertEqual(4, len(findings))
        self.assertEqual('GHSA-v6rh-hp5x-86rv', findings[3].cve)
        self.assertEqual(2, len(findings[3].vulnerability_ids))
        self.assertEqual('GHSA-v6rh-hp5x-86rv', findings[3].vulnerability_ids[0])
        self.assertEqual('CVE-2021-44420', findings[3].vulnerability_ids[1])
        test_type = Test_Type.objects.get(name=self.anchore_grype_scan_type)
        reimport_test = Test(engagement=test.engagement, test_type=test_type, scan_type=self.anchore_grype_scan_type, target_start=datetime.datetime.now(), target_end=datetime.datetime.now())
        reimport_test.save()
        reimport0 = self.reimport_scan_with_params(reimport_test.id, self.anchore_grype_file_name, scan_type=self.anchore_grype_scan_type)
        findings = Finding.objects.filter(test=reimport_test)
        self.assertEqual(4, len(findings))
        self.assertEqual('GHSA-v6rh-hp5x-86rv', findings[3].cve)
        self.assertEqual(2, len(findings[3].vulnerability_ids))
        self.assertEqual('GHSA-v6rh-hp5x-86rv', findings[3].vulnerability_ids[0])
        self.assertEqual('CVE-2021-44420', findings[3].vulnerability_ids[1])

class ImportReimportTestAPI(DojoAPITestCase, ImportReimportMixin):
    fixtures = ['dojo_testdata.json']

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        ImportReimportMixin.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def setUp(self):
        if False:
            return 10
        testuser = User.objects.get(username='admin')
        token = Token.objects.get(user=testuser)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + token.key)

    def test_import_0_reimport_1_active_verified_reimport_0_active_verified_statistics(self):
        if False:
            while True:
                i = 10
        logger.debug('reimporting updated zap xml report, 1 new finding and 1 no longer present, verified=True and then 0 again')
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        self.assertEqual(import0['statistics'], {'after': {'info': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'low': {'active': 3, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 3}, 'medium': {'active': 1, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 1}, 'high': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'critical': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'total': {'active': 4, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 4}}})
        test_id = import0['test']
        reimport1 = self.reimport_scan_with_params(test_id, self.zap_sample1_filename)
        self.assertEqual(reimport1['statistics'], {'after': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 3, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 1, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 4, 'verified': 0}, 'medium': {'active': 1, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}, 'total': {'active': 4, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 1, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 5, 'verified': 0}}, 'before': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 3, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 3, 'verified': 0}, 'medium': {'active': 1, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}, 'total': {'active': 4, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 4, 'verified': 0}}, 'delta': {'closed': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 1, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}, 'medium': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'total': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 1, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}}, 'created': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 1, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}, 'medium': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'total': {'active': 1, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}}, 'left untouched': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 2, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 2, 'verified': 0}, 'medium': {'active': 1, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}, 'total': {'active': 3, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 3, 'verified': 0}}, 'reactivated': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'medium': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'total': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}}}})
        with assertTestImportModelsCreated(self, reimports=1, affected_findings=2, closed=1, reactivated=1, untouched=3):
            reimport0 = self.reimport_scan_with_params(test_id, self.zap_sample0_filename)
        self.assertEqual(reimport0['statistics'], {'after': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 3, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 1, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 4, 'verified': 0}, 'medium': {'active': 1, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}, 'total': {'active': 4, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 1, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 5, 'verified': 0}}, 'before': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 3, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 1, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 4, 'verified': 0}, 'medium': {'active': 1, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}, 'total': {'active': 4, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 1, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 5, 'verified': 0}}, 'delta': {'closed': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 1, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}, 'medium': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'total': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 1, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}}, 'created': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'medium': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'total': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}}, 'left untouched': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 2, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 2, 'verified': 0}, 'medium': {'active': 1, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}, 'total': {'active': 3, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 3, 'verified': 0}}, 'reactivated': {'critical': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'high': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'info': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'low': {'active': 1, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}, 'medium': {'active': 0, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 0, 'verified': 0}, 'total': {'active': 1, 'duplicate': 0, 'false_p': 0, 'is_mitigated': 0, 'out_of_scope': 0, 'risk_accepted': 0, 'total': 1, 'verified': 0}}}})

    @override_settings(TRACK_IMPORT_HISTORY=False)
    def test_import_0_reimport_1_active_verified_reimport_0_active_verified_statistics_no_history(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('reimporting updated zap xml report, 1 new finding and 1 no longer present, verified=True and then 0 again')
        import0 = self.import_scan_with_params(self.zap_sample0_filename)
        self.assertEqual(import0['statistics'], {'after': {'info': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'low': {'active': 3, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 3}, 'medium': {'active': 1, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 1}, 'high': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'critical': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'total': {'active': 4, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 4}}})
        test_id = import0['test']
        reimport1 = self.reimport_scan_with_params(test_id, self.zap_sample1_filename)
        self.assertEqual(reimport1['statistics'], {'before': {'info': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'low': {'active': 3, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 3}, 'medium': {'active': 1, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 1}, 'high': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'critical': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'total': {'active': 4, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 4}}, 'after': {'info': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'low': {'active': 3, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 1, 'risk_accepted': 0, 'total': 4}, 'medium': {'active': 1, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 1}, 'high': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'critical': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'total': {'active': 4, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 1, 'risk_accepted': 0, 'total': 5}}})
        with assertTestImportModelsCreated(self, reimports=0, affected_findings=0, closed=0, reactivated=0, untouched=0):
            reimport0 = self.reimport_scan_with_params(test_id, self.zap_sample0_filename)
        self.assertEqual(reimport0['statistics'], {'before': {'info': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'low': {'active': 3, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 1, 'risk_accepted': 0, 'total': 4}, 'medium': {'active': 1, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 1}, 'high': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'critical': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'total': {'active': 4, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 1, 'risk_accepted': 0, 'total': 5}}, 'after': {'info': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'low': {'active': 3, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 1, 'risk_accepted': 0, 'total': 4}, 'medium': {'active': 1, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 1}, 'high': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'critical': {'active': 0, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 0, 'risk_accepted': 0, 'total': 0}, 'total': {'active': 4, 'verified': 0, 'duplicate': 0, 'false_p': 0, 'out_of_scope': 0, 'is_mitigated': 1, 'risk_accepted': 0, 'total': 5}}})

    def test_reimport_default_scan_date_parser_not_sets_date(self):
        if False:
            return 10
        logger.debug('importing zap xml report with date set by parser')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.reimport_scan_with_params(None, self.zap_sample0_filename, active=False, verified=False, product_name=PRODUCT_NAME_DEFAULT, engagement=None, engagement_name=ENGAGEMENT_NAME_DEFAULT, product_type_name=PRODUCT_TYPE_NAME_DEFAULT, auto_create_context=True)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id, active=False, verified=False)
        self.log_finding_summary_json_api(findings)
        date = findings['results'][0]['date']
        self.assertEqual(date, str(timezone.localtime(timezone.now()).date()))
        return test_id

    def test_reimport_default_scan_date_parser_sets_date(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('importing original acunetix xml report')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=1, created=1):
            import0 = self.reimport_scan_with_params(None, self.acunetix_file_name, scan_type=self.scan_type_acunetix, active=False, verified=False, product_name=PRODUCT_NAME_DEFAULT, engagement=None, engagement_name=ENGAGEMENT_NAME_DEFAULT, product_type_name=PRODUCT_TYPE_NAME_DEFAULT, auto_create_context=True)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id, active=False, verified=False)
        self.log_finding_summary_json_api(findings)
        date = findings['results'][0]['date']
        self.assertEqual(date, '2018-09-24')
        return test_id

    def test_reimport_set_scan_date_parser_not_sets_date(self):
        if False:
            i = 10
            return i + 15
        logger.debug('importing original zap xml report')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=4, created=4):
            import0 = self.reimport_scan_with_params(None, self.zap_sample0_filename, active=False, verified=False, scan_date='2006-12-26', product_name=PRODUCT_NAME_DEFAULT, engagement=None, engagement_name=ENGAGEMENT_NAME_DEFAULT, product_type_name=PRODUCT_TYPE_NAME_DEFAULT, auto_create_context=True)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id, active=False, verified=False)
        self.log_finding_summary_json_api(findings)
        date = findings['results'][0]['date']
        self.assertEqual(date, '2006-12-26')
        return test_id

    def test_reimport_set_scan_date_parser_sets_date(self):
        if False:
            print('Hello World!')
        logger.debug('importing acunetix xml report with date set by parser')
        with assertTestImportModelsCreated(self, imports=1, affected_findings=1, created=1):
            import0 = self.reimport_scan_with_params(None, self.acunetix_file_name, scan_type=self.scan_type_acunetix, active=False, verified=False, scan_date='2006-12-26', product_name=PRODUCT_NAME_DEFAULT, engagement=None, engagement_name=ENGAGEMENT_NAME_DEFAULT, product_type_name=PRODUCT_TYPE_NAME_DEFAULT, auto_create_context=True)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id, active=False, verified=False)
        self.log_finding_summary_json_api(findings)
        date = findings['results'][0]['date']
        self.assertEqual(date, '2006-12-26')
        return test_id

class ImportReimportTestUI(DojoAPITestCase, ImportReimportMixin):
    fixtures = ['dojo_testdata.json']
    client_ui = Client()

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        ImportReimportMixin.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def setUp(self):
        if False:
            while True:
                i = 10
        testuser = User.objects.get(username='admin')
        token = Token.objects.get(user=testuser)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + token.key)
        self.client_ui = Client()
        self.client_ui.force_login(self.get_test_admin())

    def import_scan_with_params(self, *args, **kwargs):
        if False:
            return 10
        return self.import_scan_with_params_ui(*args, **kwargs)

    def reimport_scan_with_params(self, *args, **kwargs):
        if False:
            return 10
        return self.reimport_scan_with_params_ui(*args, **kwargs)

    def import_scan_ui(self, engagement, payload):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('import_scan payload %s', payload)
        response = self.client_ui.post(reverse('import_scan_results', args=(engagement,)), payload)
        test = Test.objects.get(id=response.url.split('/')[-1])
        self.assertEqual(302, response.status_code, response.content[:1000])
        return {'test': test.id}

    def reimport_scan_ui(self, test, payload):
        if False:
            print('Hello World!')
        response = self.client_ui.post(reverse('re_import_scan_results', args=(test,)), payload)
        self.assertEqual(302, response.status_code, response.content[:1000])
        test = Test.objects.get(id=response.url.split('/')[-1])
        return {'test': test.id}

    def import_scan_with_params_ui(self, filename, scan_type='ZAP Scan', engagement=1, minimum_severity='Low', active=True, verified=False, push_to_jira=None, endpoint_to_add=None, tags=None, close_old_findings=False, scan_date=None, service=None, forceActive=False, forceVerified=False):
        if False:
            while True:
                i = 10
        activePayload = 'not_specified'
        if forceActive:
            activePayload = 'force_to_true'
        elif not active:
            activePayload = 'force_to_false'
        verifiedPayload = 'not_specified'
        if forceVerified:
            verifiedPayload = 'force_to_true'
        elif not verified:
            verifiedPayload = 'force_to_false'
        payload = {'minimum_severity': minimum_severity, 'active': activePayload, 'verified': verifiedPayload, 'scan_type': scan_type, 'file': open(get_unit_tests_path() + filename), 'environment': 1, 'version': '1.0.1', 'close_old_findings': close_old_findings}
        if push_to_jira is not None:
            payload['push_to_jira'] = push_to_jira
        if endpoint_to_add is not None:
            payload['endpoints'] = [endpoint_to_add]
        if tags is not None:
            payload['tags'] = tags
        if scan_date is not None:
            payload['scan_date'] = scan_date
        if service is not None:
            payload['service'] = service
        return self.import_scan_ui(engagement, payload)

    def reimport_scan_with_params_ui(self, test_id, filename, scan_type='ZAP Scan', minimum_severity='Low', active=True, verified=False, push_to_jira=None, tags=None, close_old_findings=True, scan_date=None):
        if False:
            i = 10
            return i + 15
        activePayload = 'force_to_true'
        if not active:
            activePayload = 'force_to_false'
        verifiedPayload = 'force_to_true'
        if not verified:
            verifiedPayload = 'force_to_false'
        payload = {'minimum_severity': minimum_severity, 'active': activePayload, 'verified': verifiedPayload, 'scan_type': scan_type, 'file': open(get_unit_tests_path() + filename), 'version': '1.0.1', 'close_old_findings': close_old_findings}
        if push_to_jira is not None:
            payload['push_to_jira'] = push_to_jira
        if tags is not None:
            payload['tags'] = tags
        if scan_date is not None:
            payload['scan_date'] = scan_date
        return self.reimport_scan_ui(test_id, payload)