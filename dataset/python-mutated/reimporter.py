import base64
import logging
import dojo.finding.helper as finding_helper
import dojo.jira_link.helper as jira_helper
import dojo.notifications.helper as notifications_helper
from dojo.decorators import dojo_async_task
from dojo.celery import app
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core import serializers
from django.core.files.base import ContentFile
from django.utils import timezone
from dojo.importers import utils as importer_utils
from dojo.importers.reimporter import utils as reimporter_utils
from dojo.models import BurpRawRequestResponse, FileUpload, Finding, Notes, Test_Import
from dojo.tools.factory import get_parser
from dojo.utils import get_current_user, is_finding_groups_enabled
from django.db.models import Q
logger = logging.getLogger(__name__)
deduplicationLogger = logging.getLogger('dojo.specific-loggers.deduplication')

class DojoDefaultReImporter(object):

    @dojo_async_task
    @app.task(ignore_result=False)
    def process_parsed_findings(self, test, parsed_findings, scan_type, user, active=None, verified=None, minimum_severity=None, endpoints_to_add=None, push_to_jira=None, group_by=None, now=timezone.now(), service=None, scan_date=None, do_not_reactivate=False, create_finding_groups_for_all_findings=True, **kwargs):
        if False:
            while True:
                i = 10
        items = parsed_findings
        original_items = list(test.finding_set.all())
        new_items = []
        mitigated_count = 0
        finding_count = 0
        finding_added_count = 0
        reactivated_count = 0
        reactivated_items = []
        unchanged_count = 0
        unchanged_items = []
        logger.debug('starting reimport of %i items.', len(items) if items else 0)
        deduplication_algorithm = test.deduplication_algorithm
        i = 0
        group_names_to_findings_dict = {}
        logger.debug('STEP 1: looping over findings from the reimported report and trying to match them to existing findings')
        deduplicationLogger.debug('Algorithm used for matching new findings to existing findings: %s', deduplication_algorithm)
        for item in items:
            if item.severity.lower().startswith('info') and item.severity != 'Info':
                item.severity = 'Info'
            item.numerical_severity = Finding.get_numerical_severity(item.severity)
            if minimum_severity and Finding.SEVERITIES[item.severity] > Finding.SEVERITIES[minimum_severity]:
                continue
            component_name = item.component_name if hasattr(item, 'component_name') else None
            component_version = item.component_version if hasattr(item, 'component_version') else None
            if not hasattr(item, 'test'):
                item.test = test
            if service:
                item.service = service
            if item.dynamic_finding:
                for e in item.unsaved_endpoints:
                    try:
                        e.clean()
                    except ValidationError as err:
                        logger.warning("DefectDojo is storing broken endpoint because cleaning wasn't successful: {}".format(err))
            item.hash_code = item.compute_hash_code()
            deduplicationLogger.debug("item's hash_code: %s", item.hash_code)
            findings = reimporter_utils.match_new_finding_to_existing_finding(item, test, deduplication_algorithm)
            deduplicationLogger.debug('found %i findings matching with current new finding', len(findings))
            if findings:
                finding = findings[0]
                if finding.false_p or finding.out_of_scope or finding.risk_accepted:
                    logger.debug('%i: skipping existing finding (it is marked as false positive:%s and/or out of scope:%s or is a risk accepted:%s): %i:%s:%s:%s', i, finding.false_p, finding.out_of_scope, finding.risk_accepted, finding.id, finding, finding.component_name, finding.component_version)
                    if finding.false_p == item.false_p and finding.out_of_scope == item.out_of_scope and (finding.risk_accepted == item.risk_accepted):
                        unchanged_items.append(finding)
                        unchanged_count += 1
                        continue
                elif finding.is_mitigated:
                    if item.is_mitigated:
                        unchanged_items.append(finding)
                        unchanged_count += 1
                        if item.mitigated:
                            logger.debug('item mitigated time: ' + str(item.mitigated.timestamp()))
                            logger.debug('finding mitigated time: ' + str(finding.mitigated.timestamp()))
                            if item.mitigated.timestamp() == finding.mitigated.timestamp():
                                logger.debug('New imported finding and already existing finding have the same mitigation date, will skip as they are the same.')
                                continue
                            if item.mitigated.timestamp() != finding.mitigated.timestamp():
                                logger.debug('New imported finding and already existing finding are both mitigated but have different dates, not taking action')
                                continue
                        else:
                            continue
                    else:
                        if not do_not_reactivate:
                            logger.debug('%i: reactivating: %i:%s:%s:%s', i, finding.id, finding, finding.component_name, finding.component_version)
                            finding.mitigated = None
                            finding.is_mitigated = False
                            finding.mitigated_by = None
                            finding.active = True
                            if verified is not None:
                                finding.verified = verified
                        if do_not_reactivate:
                            logger.debug("%i: skipping reactivating by user's choice do_not_reactivate: %i:%s:%s:%s", i, finding.id, finding, finding.component_name, finding.component_version)
                            existing_note = finding.notes.filter(entry='Finding has skipped reactivation from %s re-upload with user decision do_not_reactivate.' % scan_type, author=user)
                            if len(existing_note) == 0:
                                note = Notes(entry='Finding has skipped reactivation from %s re-upload with user decision do_not_reactivate.' % scan_type, author=user)
                                note.save()
                                finding.notes.add(note)
                                finding.save(dedupe_option=False)
                            continue
                    finding.component_name = finding.component_name if finding.component_name else component_name
                    finding.component_version = finding.component_version if finding.component_version else component_version
                    finding.save(dedupe_option=False)
                    note = Notes(entry='Re-activated by %s re-upload.' % scan_type, author=user)
                    note.save()
                    endpoint_statuses = finding.status_finding.exclude(Q(false_positive=True) | Q(out_of_scope=True) | Q(risk_accepted=True))
                    reimporter_utils.chunk_endpoints_and_reactivate(endpoint_statuses)
                    finding.notes.add(note)
                    reactivated_items.append(finding)
                    reactivated_count += 1
                else:
                    logger.debug('%i: updating existing finding: %i:%s:%s:%s', i, finding.id, finding, finding.component_name, finding.component_version)
                    if not (finding.mitigated and finding.is_mitigated):
                        logger.debug('Reimported item matches a finding that is currently open.')
                        if item.is_mitigated:
                            logger.debug('Reimported mitigated item matches a finding that is currently open, closing.')
                            logger.debug('%i: closing: %i:%s:%s:%s', i, finding.id, finding, finding.component_name, finding.component_version)
                            finding.mitigated = item.mitigated
                            finding.is_mitigated = True
                            finding.mitigated_by = item.mitigated_by
                            finding.active = False
                            if verified is not None:
                                finding.verified = verified
                        elif item.risk_accepted or item.false_p or item.out_of_scope:
                            logger.debug('Reimported mitigated item matches a finding that is currently open, closing.')
                            logger.debug('%i: closing: %i:%s:%s:%s', i, finding.id, finding, finding.component_name, finding.component_version)
                            finding.risk_accepted = item.risk_accepted
                            finding.false_p = item.false_p
                            finding.out_of_scope = item.out_of_scope
                            finding.active = False
                            if verified is not None:
                                finding.verified = verified
                        else:
                            unchanged_items.append(finding)
                            unchanged_count += 1
                    if component_name is not None and (not finding.component_name) or (component_version is not None and (not finding.component_version)):
                        finding.component_name = finding.component_name if finding.component_name else component_name
                        finding.component_version = finding.component_version if finding.component_version else component_version
                        finding.save(dedupe_option=False)
                if finding.dynamic_finding:
                    logger.debug('Re-import found an existing dynamic finding for this new finding. Checking the status of endpoints')
                    reimporter_utils.update_endpoint_status(finding, item, user)
            else:
                item.reporter = user
                item.last_reviewed = timezone.now()
                item.last_reviewed_by = user
                if active is not None:
                    item.active = active
                if verified is not None:
                    item.verified = verified
                if scan_date:
                    item.date = scan_date.date()
                item.save(dedupe_option=False)
                logger.debug('%i: reimport created new finding as no existing finding match: %i:%s:%s:%s', i, item.id, item, item.component_name, item.component_version)
                if is_finding_groups_enabled() and group_by:
                    name = finding_helper.get_group_by_group_name(item, group_by)
                    if name is not None:
                        if name in group_names_to_findings_dict:
                            group_names_to_findings_dict[name].append(item)
                        else:
                            group_names_to_findings_dict[name] = [item]
                finding_added_count += 1
                new_items.append(item)
                finding = item
                if hasattr(item, 'unsaved_req_resp'):
                    for req_resp in item.unsaved_req_resp:
                        burp_rr = BurpRawRequestResponse(finding=finding, burpRequestBase64=base64.b64encode(req_resp['req'].encode('utf-8')), burpResponseBase64=base64.b64encode(req_resp['resp'].encode('utf-8')))
                        burp_rr.clean()
                        burp_rr.save()
                if item.unsaved_request and item.unsaved_response:
                    burp_rr = BurpRawRequestResponse(finding=finding, burpRequestBase64=base64.b64encode(item.unsaved_request.encode()), burpResponseBase64=base64.b64encode(item.unsaved_response.encode()))
                    burp_rr.clean()
                    burp_rr.save()
            if finding:
                finding_count += 1
                importer_utils.chunk_endpoints_and_disperse(finding, test, item.unsaved_endpoints)
                if endpoints_to_add:
                    importer_utils.chunk_endpoints_and_disperse(finding, test, endpoints_to_add)
                if item.unsaved_tags:
                    finding.tags = item.unsaved_tags
                if item.unsaved_files:
                    for unsaved_file in item.unsaved_files:
                        data = base64.b64decode(unsaved_file.get('data'))
                        title = unsaved_file.get('title', '<No title>')
                        (file_upload, file_upload_created) = FileUpload.objects.get_or_create(title=title)
                        file_upload.file.save(title, ContentFile(data))
                        file_upload.save()
                        finding.files.add(file_upload)
                if finding.unsaved_vulnerability_ids:
                    importer_utils.handle_vulnerability_ids(finding)
                finding.component_name = finding.component_name if finding.component_name else component_name
                finding.component_version = finding.component_version if finding.component_version else component_version
                if is_finding_groups_enabled() and group_by:
                    finding.save()
                else:
                    finding.save(push_to_jira=push_to_jira)
        to_mitigate = set(original_items) - set(reactivated_items) - set(unchanged_items)
        untouched = set(unchanged_items) - set(to_mitigate) - set(new_items)
        for (group_name, findings) in group_names_to_findings_dict.items():
            finding_helper.add_findings_to_auto_group(group_name, findings, group_by, create_finding_groups_for_all_findings, **kwargs)
            if push_to_jira:
                if findings[0].finding_group is not None:
                    jira_helper.push_to_jira(findings[0].finding_group)
                else:
                    jira_helper.push_to_jira(findings[0])
        if is_finding_groups_enabled() and push_to_jira:
            for finding_group in set([finding.finding_group for finding in reactivated_items + unchanged_items if finding.finding_group is not None and (not finding.is_mitigated)]):
                jira_helper.push_to_jira(finding_group)
        sync = kwargs.get('sync', False)
        if not sync:
            serialized_new_items = [serializers.serialize('json', [finding]) for finding in new_items]
            serialized_reactivated_items = [serializers.serialize('json', [finding]) for finding in reactivated_items]
            serialized_to_mitigate = [serializers.serialize('json', [finding]) for finding in to_mitigate]
            serialized_untouched = [serializers.serialize('json', [finding]) for finding in untouched]
            return (serialized_new_items, serialized_reactivated_items, serialized_to_mitigate, serialized_untouched)
        return (new_items, reactivated_items, to_mitigate, untouched)

    def close_old_findings(self, test, to_mitigate, scan_date_time, user, push_to_jira=None):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('IMPORT_SCAN: Closing findings no longer present in scan report')
        mitigated_findings = []
        for finding in to_mitigate:
            if not finding.mitigated or not finding.is_mitigated:
                logger.debug('mitigating finding: %i:%s', finding.id, finding)
                finding.mitigated = scan_date_time
                finding.is_mitigated = True
                finding.mitigated_by = user
                finding.active = False
                endpoint_status = finding.status_finding.all()
                reimporter_utils.mitigate_endpoint_status(endpoint_status, user, kwuser=user, sync=True)
                if is_finding_groups_enabled() and finding.finding_group:
                    finding.save(dedupe_option=False)
                else:
                    finding.save(push_to_jira=push_to_jira, dedupe_option=False)
                note = Notes(entry='Mitigated by %s re-upload.' % test.test_type, author=user)
                note.save()
                finding.notes.add(note)
                mitigated_findings.append(finding)
        if is_finding_groups_enabled() and push_to_jira:
            for finding_group in set([finding.finding_group for finding in to_mitigate if finding.finding_group is not None]):
                jira_helper.push_to_jira(finding_group)
        return mitigated_findings

    def reimport_scan(self, scan, scan_type, test, active=None, verified=None, tags=None, minimum_severity=None, user=None, endpoints_to_add=None, scan_date=None, version=None, branch_tag=None, build_id=None, commit_hash=None, push_to_jira=None, close_old_findings=True, group_by=None, api_scan_configuration=None, service=None, do_not_reactivate=False, create_finding_groups_for_all_findings=True):
        if False:
            i = 10
            return i + 15
        logger.debug(f'REIMPORT_SCAN: parameters: {locals()}')
        user = user or get_current_user()
        now = timezone.now()
        if api_scan_configuration:
            if api_scan_configuration.product != test.engagement.product:
                raise ValidationError('API Scan Configuration has to be from same product as the Test')
            if test.api_scan_configuration != api_scan_configuration:
                test.api_scan_configuration = api_scan_configuration
                test.save()
        parser = get_parser(scan_type)
        if hasattr(parser, 'get_tests'):
            logger.debug('REIMPORT_SCAN parser v2: Create parse findings')
            try:
                tests = parser.get_tests(scan_type, scan)
            except ValueError as e:
                logger.warning(e)
                raise ValidationError(e)
            parsed_findings = []
            for test_raw in tests:
                parsed_findings.extend(test_raw.findings)
        else:
            logger.debug('REIMPORT_SCAN: Parse findings')
            try:
                parsed_findings = parser.get_findings(scan, test)
            except ValueError as e:
                logger.warning(e)
                raise ValidationError(e)
        logger.debug('REIMPORT_SCAN: Processing findings')
        new_findings = []
        reactivated_findings = []
        findings_to_mitigate = []
        untouched_findings = []
        if settings.ASYNC_FINDING_IMPORT:
            chunk_list = importer_utils.chunk_list(parsed_findings)
            results_list = []
            for findings_list in chunk_list:
                result = self.process_parsed_findings(test, findings_list, scan_type, user, active=active, verified=verified, minimum_severity=minimum_severity, endpoints_to_add=endpoints_to_add, push_to_jira=push_to_jira, group_by=group_by, now=now, service=service, scan_date=scan_date, sync=False, do_not_reactivate=do_not_reactivate, create_finding_groups_for_all_findings=create_finding_groups_for_all_findings)
                results_list += [result]
            logger.debug('REIMPORT_SCAN: Collecting Findings')
            for results in results_list:
                (serial_new_findings, serial_reactivated_findings, serial_findings_to_mitigate, serial_untouched_findings) = results.get()
                new_findings += [next(serializers.deserialize('json', finding)).object for finding in serial_new_findings]
                reactivated_findings += [next(serializers.deserialize('json', finding)).object for finding in serial_reactivated_findings]
                findings_to_mitigate += [next(serializers.deserialize('json', finding)).object for finding in serial_findings_to_mitigate]
                untouched_findings += [next(serializers.deserialize('json', finding)).object for finding in serial_untouched_findings]
            logger.debug('REIMPORT_SCAN: All Findings Collected')
            test.percent_complete = 50
            test.save()
            importer_utils.update_test_progress(test)
        else:
            (new_findings, reactivated_findings, findings_to_mitigate, untouched_findings) = self.process_parsed_findings(test, parsed_findings, scan_type, user, active=active, verified=verified, minimum_severity=minimum_severity, endpoints_to_add=endpoints_to_add, push_to_jira=push_to_jira, group_by=group_by, now=now, service=service, scan_date=scan_date, sync=True, do_not_reactivate=do_not_reactivate, create_finding_groups_for_all_findings=create_finding_groups_for_all_findings)
        closed_findings = []
        if close_old_findings:
            logger.debug('REIMPORT_SCAN: Closing findings no longer present in scan report')
            closed_findings = self.close_old_findings(test, findings_to_mitigate, scan_date, user=user, push_to_jira=push_to_jira)
        logger.debug('REIMPORT_SCAN: Updating test/engagement timestamps')
        importer_utils.update_timestamps(test, version, branch_tag, build_id, commit_hash, now, scan_date)
        logger.debug('REIMPORT_SCAN: Updating test tags')
        importer_utils.update_tags(test, tags)
        test_import = None
        if settings.TRACK_IMPORT_HISTORY:
            logger.debug('REIMPORT_SCAN: Updating Import History')
            test_import = importer_utils.update_import_history(Test_Import.REIMPORT_TYPE, active, verified, tags, minimum_severity, endpoints_to_add, version, branch_tag, build_id, commit_hash, push_to_jira, close_old_findings, test, new_findings, closed_findings, reactivated_findings, untouched_findings)
        logger.debug('REIMPORT_SCAN: Generating notifications')
        updated_count = len(closed_findings) + len(reactivated_findings) + len(new_findings)
        if updated_count > 0:
            notifications_helper.notify_scan_added(test, updated_count, new_findings=new_findings, findings_mitigated=closed_findings, findings_reactivated=reactivated_findings, findings_untouched=untouched_findings)
        logger.debug('REIMPORT_SCAN: Done')
        return (test, updated_count, len(new_findings), len(closed_findings), len(reactivated_findings), len(untouched_findings), test_import)