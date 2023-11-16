import logging
import re
import zipfile
from re import sub
from typing import List
from defusedxml.ElementTree import fromstring
from oletools import mraptor
from oletools.msodde import process_maybe_encrypted as msodde_process_maybe_encrypted
from oletools.olevba import VBA_Parser
from api_app.analyzers_manager.classes import FileAnalyzer
from api_app.analyzers_manager.models import MimeTypes
logger = logging.getLogger(__name__)
try:
    from XLMMacroDeobfuscator.deobfuscator import show_cells
    from XLMMacroDeobfuscator.xls_wrapper_2 import XLSWrapper2
except Exception as e:
    logger.exception(e)

class CannotDecryptException(Exception):
    pass

class DocInfo(FileAnalyzer):
    experimental: bool
    additional_passwords_to_check: list

    def config(self):
        if False:
            while True:
                i = 10
        super().config()
        self.olevba_results = {}
        self.vbaparser = None
        self.passwords_to_check = []
        self.passwords_to_check.extend(self.additional_passwords_to_check)

    def run(self):
        if False:
            print('Hello World!')
        results = {}
        try:
            self.vbaparser = VBA_Parser(self.filepath)
            self.manage_encrypted_doc()
            self.manage_xlm_macros()
            self.olevba_results['macro_found'] = self.vbaparser.detect_vba_macros()
            if self.olevba_results['macro_found']:
                vba_code_all_modules = ''
                macro_data = []
                for (v_filename, stream_path, vba_filename, vba_code) in self.vbaparser.extract_macros():
                    extracted_macro = {'filename': v_filename, 'ole_stream': stream_path, 'vba_filename': vba_filename, 'vba_code': vba_code}
                    macro_data.append(extracted_macro)
                    vba_code_all_modules += vba_code + '\n'
                self.olevba_results['macro_data'] = macro_data
                macro_raptor = mraptor.MacroRaptor(vba_code_all_modules)
                if macro_raptor:
                    macro_raptor.scan()
                    results['mraptor'] = 'suspicious' if macro_raptor.suspicious else 'ok'
                analyzer_results = self.vbaparser.analyze_macros()
                if analyzer_results:
                    analyze_macro_results = []
                    for (kw_type, keyword, description) in analyzer_results:
                        if kw_type != 'Hex String':
                            analyze_macro_result = {'type': kw_type, 'keyword': keyword, 'description': description}
                            analyze_macro_results.append(analyze_macro_result)
                    self.olevba_results['analyze_macro'] = analyze_macro_results
        except CannotDecryptException as e:
            logger.info(e)
        except Exception as e:
            error_message = f'job_id {self.job_id} vba parser failed. Error: {e}'
            logger.warning(error_message, stack_info=True)
            self.report.errors.append(error_message)
            self.report.save()
        finally:
            if self.vbaparser:
                self.vbaparser.close()
        results['olevba'] = self.olevba_results
        if self.file_mimetype != MimeTypes.ONE_NOTE.value:
            results['msodde'] = self.analyze_msodde()
        if self.file_mimetype in [MimeTypes.WORD1.value, MimeTypes.WORD2.value, MimeTypes.ZIP1.value, MimeTypes.ZIP2.value]:
            results['follina'] = self.analyze_for_follina_cve()
        return results

    def analyze_for_follina_cve(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        hits = []
        try:
            zipped = zipfile.ZipFile(self.filepath)
        except zipfile.BadZipFile:
            logger.info(f"file {self.filename} is not a zip file so wecant' do custom Follina Extraction")
        else:
            try:
                template = zipped.read('word/_rels/document.xml.rels')
            except KeyError:
                pass
            else:
                xml_root = fromstring(template)
                for xml_node in xml_root.iter():
                    target = xml_node.attrib.get('Target')
                    if target:
                        target = target.strip().lower()
                        hits += re.findall('mhtml:(https?://.*?)!', target)
        return hits

    def analyze_msodde(self):
        if False:
            print('Hello World!')
        try:
            msodde_result = msodde_process_maybe_encrypted(self.filepath, self.passwords_to_check)
        except Exception as e:
            error_message = f'job_id {self.job_id} msodde parser failed. Error: {e}'
            if 'Could not determine delimiter' in str(e) or self.filename.endswith('.exe'):
                logger.info(error_message, stack_info=True)
            else:
                logger.warning(error_message, stack_info=True)
            self.report.errors.append(error_message)
            self.report.save()
            msodde_result = f'Error: {e}'
        return msodde_result

    def manage_encrypted_doc(self):
        if False:
            for i in range(10):
                print('nop')
        self.olevba_results['is_encrypted'] = False
        if self.vbaparser.ole_file:
            is_encrypted = self.vbaparser.detect_is_encrypted()
            self.olevba_results['is_encrypted'] = is_encrypted
            if is_encrypted:
                common_pwd_to_check = []
                for num in range(10):
                    common_pwd_to_check.append(f'{num}{num}{num}{num}')
                filename_without_spaces_and_numbers = sub('[-_\\d\\s]', '', self.filename)
                filename_without_extension = sub('(\\..+)', '', filename_without_spaces_and_numbers)
                common_pwd_to_check.append(filename_without_extension)
                self.passwords_to_check.extend(common_pwd_to_check)
                decrypted_file_name = self.vbaparser.decrypt_file(self.passwords_to_check)
                self.olevba_results['additional_passwords_tried'] = self.passwords_to_check
                if decrypted_file_name:
                    self.vbaparser = VBA_Parser(decrypted_file_name)
                else:
                    self.olevba_results['cannot_decrypt'] = True
                    raise CannotDecryptException('cannot decrypt the file with the default password')

    def manage_xlm_macros(self):
        if False:
            print('Hello World!')
        self.olevba_results['xlm_macro'] = False
        if self.vbaparser.detect_xlm_macros():
            self.olevba_results['xlm_macro'] = True
            logger.debug('experimental XLM macro analysis start')
            parsed_file = b''
            try:
                excel_doc = XLSWrapper2(self.filepath)
                ae_list = ['auto_open', 'auto_close', 'auto_activate', 'auto_deactivate']
                self.olevba_results['xlm_macro_autoexec'] = []
                for ae in ae_list:
                    auto_exec_labels = excel_doc.get_defined_name(ae, full_match=False)
                    for label in auto_exec_labels:
                        self.olevba_results['xlm_macro_autoexec'].append(label[0])
                for i in show_cells(excel_doc):
                    rec_str = ''
                    if len(i) == 5:
                        if i[2] != 'None':
                            rec_str = '{:20}'.format(i[2])
                    if rec_str:
                        parsed_file += rec_str.encode()
                        parsed_file += b'\n'
            except Exception as e:
                logger.info(f'experimental XLM macro analysis failed. Exception: {e}')
            else:
                logger.debug(f'experimental XLM macro analysis succeeded. Binary to analyze: {parsed_file}')
                if parsed_file:
                    self.vbaparser = VBA_Parser(self.filename, data=parsed_file)