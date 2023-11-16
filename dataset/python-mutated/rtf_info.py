import re
from typing import List
from oletools.rtfobj import RtfObjParser
from api_app.analyzers_manager.classes import FileAnalyzer

class RTFInfo(FileAnalyzer):

    def analyze_for_follina_cve(self) -> List[str]:
        if False:
            print('Hello World!')
        content = self.read_file_bytes().decode('utf8', errors='ignore')
        return re.findall('objclass (https?://.*?)}', content)

    def run(self):
        if False:
            while True:
                i = 10
        results = {}
        rtfobj_results = {}
        binary = self.read_file_bytes()
        rtfp = RtfObjParser(binary)
        rtfp.parse()
        rtfobj_results['ole_objects'] = []
        for rtfobj in rtfp.objects:
            if rtfobj.is_ole:
                class_name = rtfobj.class_name.decode()
                ole_dict = {'format_id': rtfobj.format_id, 'class_name': class_name, 'ole_datasize': rtfobj.oledata_size}
                if rtfobj.is_package:
                    ole_dict['is_package'] = True
                    ole_dict['filename'] = rtfobj.filename
                    ole_dict['src_path'] = rtfobj.src_path
                    ole_dict['temp_path'] = rtfobj.temp_path
                    ole_dict['olepkgdata_md5'] = rtfobj.olepkgdata_md5
                else:
                    ole_dict['ole_md5'] = rtfobj.oledata_md5
                if rtfobj.clsid:
                    ole_dict['clsid_desc'] = rtfobj.clsid_desc
                    ole_dict['clsid_id'] = rtfobj.clsid
                rtfobj_results['ole_objects'].append(ole_dict)
                if class_name == 'OLE2Link':
                    rtfobj_results['exploit_ole2link_vuln'] = True
                elif class_name.lower() == 'equation.3':
                    rtfobj_results['exploit_equation_editor'] = True
        results['rtfobj'] = rtfobj_results
        results['follina'] = self.analyze_for_follina_cve()
        return results