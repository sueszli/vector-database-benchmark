"""
    www.icourse163.org 下所有免费课程的下载和解析
"""
import os
import re
if __package__ is None:
    import sys
    sys.path.append('..\\')
    sys.path.append('..\\..\\')
from Mooc.Mooc_Config import *
from Mooc.Mooc_Base import *
from Mooc.Mooc_Download import *
from Mooc.Mooc_Request import *
from Mooc.Mooc_Potplayer import *
from Mooc.Icourse163.Icourse163_Config import *
from Mooc.Icourse163.Icourse163_Base import *
__all__ = ['Icourse163_Mooc']

class Icourse163_Mooc(Icourse163_Base):
    course_url = 'https://www.icourse163.org/course/'
    infos_url = 'https://www.icourse163.org/dwr/call/plaincall/CourseBean.getMocTermDto.dwr'
    parse_url = 'https://www.icourse163.org/dwr/call/plaincall/CourseBean.getLessonUnitLearnVo.dwr'
    infos_data = {'callCount': '1', 'scriptSessionId': '${scriptSessionId}190', 'c0-scriptName': 'CourseBean', 'c0-methodName': 'getMocTermDto', 'c0-id': '0', 'c0-param0': None, 'c0-param1': 'number:0', 'c0-param2': 'boolean:true', 'batchId': '1543633161622'}
    parse_data = {'callCount': '1', 'scriptSessionId': '${scriptSessionId}190', 'c0-scriptName': 'CourseBean', 'c0-methodName': 'getLessonUnitLearnVo', 'httpSessionId': '5531d06316b34b9486a6891710115ebc', 'c0-id': '0', 'c0-param0': None, 'c0-param1': None, 'c0-param2': 'number:0', 'c0-param3': None, 'batchId': '1543633161622'}

    def __init__(self, mode=IS_SHD):
        if False:
            while True:
                i = 10
        super().__init__()
        self.mode = mode

    def _get_cid(self, url):
        if False:
            return 10
        self.cid = None
        match = courses_re['icourse163_mooc'].match(url)
        if match and match.group(4):
            self.cid = match.group(4)

    def _get_title(self):
        if False:
            print('Hello World!')
        if self.cid is None:
            return
        self.title = self.term_id = None
        url = self.course_url + self.cid
        text = request_get(url)
        match = re.search('termId : "(\\d+)"', text)
        if match:
            self.term_id = match.group(1)
        names = re.findall('name:"(.+)"', text)
        if names:
            title = '__'.join(names)
            self.title = winre.sub('', title)[:WIN_LENGTH]

    def _get_infos(self):
        if False:
            return 10
        if self.term_id is None:
            return
        self.infos = {}
        self.infos_data['c0-param0'] = 'number:' + self.term_id
        text = request_post(self.infos_url, self.infos_data, decoding='unicode_escape')
        chapters = re.findall('homeworks=\\w+;.+?id=(\\d+).+?name="((.|\\n)+?)";', text)
        for (i, chapter) in enumerate(chapters, 1):
            chapter_title = winre.sub('', '{' + str(i) + '}--' + chapter[1])[:WIN_LENGTH]
            self.infos[chapter_title] = {}
            lessons = re.findall('chapterId=' + chapter[0] + '.+?contentType=1.+?id=(\\d+).+?isTestChecked=false.+?name="((.|\\n)+?)".+?test', text)
            for (j, lesson) in enumerate(lessons, 1):
                lesson_title = winre.sub('', '{' + str(j) + '}--' + lesson[1])[:WIN_LENGTH]
                self.infos[chapter_title][lesson_title] = {}
                videos = re.findall('contentId=(\\d+).+contentType=(1).+id=(\\d+).+lessonId=' + lesson[0] + '.+name="(.+)"', text)
                pdfs = re.findall('contentId=(\\d+).+contentType=(3).+id=(\\d+).+lessonId=' + lesson[0] + '.+name="(.+)"', text)
                video_source = [{'params': video[:3], 'name': winre.sub('', '[{}.{}.{}]--{}'.format(i, j, k, video[3])).rstrip('.mp4')[:WIN_LENGTH]} for (k, video) in enumerate(videos, 1)]
                pdf_source = [{'params': pdf[:3], 'name': winre.sub('', '({}.{}.{})--{}'.format(i, j, k, pdf[3])).rstrip('.pdf')[:WIN_LENGTH]} for (k, pdf) in enumerate(pdfs, 1)]
                self.infos[chapter_title][lesson_title]['videos'] = video_source
                self.infos[chapter_title][lesson_title]['pdfs'] = pdf_source

    def _get_source_text(self, params):
        if False:
            i = 10
            return i + 15
        self.parse_data['c0-param0'] = params[0]
        self.parse_data['c0-param1'] = params[1]
        self.parse_data['c0-param3'] = params[2]
        text = request_post(self.parse_url, self.parse_data, decoding='unicode_escape')
        return text

    def _get_pdf_url(self, params):
        if False:
            print('Hello World!')
        text = self._get_source_text(params)
        pdf_match = re.search('textOrigUrl:"(.*?)"', text)
        pdf_url = None
        if pdf_match:
            pdf_url = pdf_match.group(1)
        return pdf_url

    def _get_video_url(self, params):
        if False:
            print('Hello World!')
        text = self._get_source_text(params)
        sub_match = re.search('name=".+";.*url="(.*?)"', text)
        video_url = sub_url = None
        if sub_match:
            sub_url = sub_match.group(1)
        resolutions = ['Shd', 'Hd', 'Sd']
        for (index, sp) in enumerate(resolutions, 1):
            video_match = re.search('(?P<ext>mp4)%sUrl="(?P<url>.*?\\.(?P=ext).*?)"' % sp, text)
            if video_match:
                (video_url, _) = video_match.group('url', 'ext')
                if index >= self.mode:
                    break
        return (video_url, sub_url)

    def _download(self):
        if False:
            print('Hello World!')
        print('\n{:^{}s}'.format(self.title, LEN_S))
        self.rootDir = rootDir = os.path.join(PATH, self.title)
        courseDir = os.path.join(rootDir, COURSENAME)
        if not os.path.exists(courseDir):
            os.makedirs(courseDir)
        Icourse163_Base.potplayer.init(rootDir)
        Icourse163_Base.potplayer.enable()
        for (i, chapter) in enumerate(self.infos, 1):
            print(chapter)
            chapterDir = os.path.join(courseDir, chapter)
            if not os.path.exists(chapterDir):
                os.mkdir(chapterDir)
            for (j, lesson) in enumerate(self.infos[chapter], 1):
                lessonDir = os.path.join(chapterDir, lesson)
                if not os.path.exists(lessonDir):
                    os.mkdir(lessonDir)
                print('  ' + lesson)
                sources = self.infos[chapter][lesson]
                for (k, pdf_source) in enumerate(sources['pdfs'], 1):
                    (params, pdf_name) = (pdf_source['params'], pdf_source['name'])
                    pdf_url = self._get_pdf_url(params)
                    if pdf_url:
                        self.download_pdf(pdf_url, pdf_name, lessonDir)
                if self.mode == ONLY_PDF:
                    continue
                for (k, video_source) in enumerate(sources['videos'], 1):
                    (params, name) = (video_source['params'], video_source['name'])
                    video_name = sub_name = name
                    (video_url, sub_url) = self._get_video_url(params)
                    if video_url:
                        self.download_video(video_url=video_url, video_name=video_name, video_dir=lessonDir)
                    if sub_url:
                        self.download_sub(sub_url, sub_name, lessonDir)

    def prepare(self, url):
        if False:
            print('Hello World!')
        self._get_cid(url)
        self._get_title()
        self._get_infos()

    def download(self):
        if False:
            return 10
        if self.cid and self.title and self.term_id and self.infos:
            self._download()
            return True
        return False

def main():
    if False:
        return 10
    url = 'https://www.icourse163.org/course/XHDX-1205600803'
    icourse163_mooc = Icourse163_Mooc()
    if icourse163_mooc.set_mode():
        icourse163_mooc.prepare(url)
        icourse163_mooc.download()
if __name__ == '__main__':
    main()