"""Converts a Selenium IDE recording that was exported as a
Python WebDriver unittest file into SeleniumBase Python file.
Also works with exported Katalon Recorder Selenium scripts:
    https://chrome.google.com/webstore/detail
        /katalon-recorder-selenium/ljdobmomdgdljniojadhoplhkpialdid
Usage:
        seleniumbase convert [PYTHON_WEBDRIVER_UNITTEST_FILE].py
Output:
        [NEW_FILE_SB].py  (adds "_SB" to the original file name)
                          (the original file is kept intact)"""
import codecs
import re
import sys
from seleniumbase.fixtures import js_utils

def main():
    if False:
        return 10
    expected_arg = '[A PYTHON_WEBDRIVER_UNITTEST_FILE exported from a Katalon/Selenium-IDE recording].py'
    num_args = len(sys.argv)
    if sys.argv[0].split('/')[-1] == 'seleniumbase' or sys.argv[0].split('\\')[-1] == 'seleniumbase':
        if num_args < 3 or num_args > 3:
            raise Exception('\n\n* INVALID RUN COMMAND! *  Usage:\n"seleniumbase convert %s"\n' % expected_arg)
    elif sys.argv[0].split('/')[-1] == 'sbase' or sys.argv[0].split('\\')[-1] == 'sbase':
        if num_args < 3 or num_args > 3:
            raise Exception('\n\n* INVALID RUN COMMAND! *  Usage:\n"sbase convert %s"\n' % expected_arg)
    elif num_args < 2 or num_args > 2:
        raise Exception('\n\n* INVALID RUN COMMAND! *  Usage:\n"python convert_ide.py %s"\n' % expected_arg)
    webdriver_python_file = sys.argv[num_args - 1]
    if not webdriver_python_file.endswith('.py'):
        raise Exception('\n\n`%s` is not a Python file!\n\nExpecting: %s\n' % (webdriver_python_file, expected_arg))
    seleniumbase_lines = []
    seleniumbase_lines.append('from seleniumbase import BaseCase')
    seleniumbase_lines.append('BaseCase.main(__name__, __file__)')
    seleniumbase_lines.append('')
    seleniumbase_lines.append('')
    ide_base_url = ''
    in_test_method = False
    uses_keys = False
    uses_select = False
    with open(webdriver_python_file, 'r', encoding='utf-8') as f:
        all_code = f.read()
    if 'def test_' not in all_code:
        raise Exception('\n\n`%s` is not a valid Python unittest.TestCase file!\n\nExpecting: %s\n\nDid you properly export your Katalon/Selenium-IDE recording as a Python WebDriver unittest file?\n' % (webdriver_python_file, expected_arg))
    code_lines = all_code.split('\n')
    for line in code_lines:
        data = re.findall('^\\s*# -\\*- coding: utf-8 -\\*-\\s*$', line)
        if data:
            continue
        data = re.findall('^class\\s\\S+\\(BaseCase\\):\\s*$', line)
        if data:
            seleniumbase_lines.append(line)
            continue
        data = re.findall('^class\\s\\S+\\(unittest\\.TestCase\\):\\s*$', line)
        if data:
            data = data[0].replace('unittest.TestCase', 'BaseCase')
            seleniumbase_lines.append(data)
            continue
        data = re.match('^\\s*self.base_url = "(\\S+)"\\s*$', line)
        if data:
            ide_base_url = data.group(1)
            continue
        data = re.match('^\\s*def\\s(\\S+)\\(self[,\\s\\S]*\\):\\s*$', line)
        if data:
            method_name = data.group(1)
            if method_name.startswith('test_'):
                in_test_method = True
                seleniumbase_lines.append(data.group())
            else:
                in_test_method = False
            continue
        if not in_test_method:
            continue
        if line.strip().startswith('#'):
            continue
        if len(line.strip()) == 0:
            continue
        if line.strip().endswith('.clear()'):
            continue
        data = re.findall('^\\s*driver = self.driver\\s*$', line)
        if data:
            continue
        data = re.match('^(\\s*)driver\\.get\\((self\\.base_url \\+ \\"/\\S*\\")\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            url = data.group(2)
            url = url.replace('self.base_url', '"%s"' % ide_base_url)
            if '/" + "/' in url:
                url = url.replace('/" + "/', '/')
            if "/' + '/" in url:
                url = url.replace("/' + '/", '/')
            command = '%sself.open(%s)' % (whitespace, url)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.get\\(\\"(\\S*)\\"\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            url = data.group(2)
            command = "%sself.open('%s')" % (whitespace, url)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_id\\(\\"(\\S+)\\"\\)\\.click\\(\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '#%s' % data.group(2).replace('#', '\\#')
            selector = selector.replace('[', '\\[').replace(']', '\\]')
            selector = selector.replace('.', '\\.')
            raw = ''
            if '\\[' in selector or '\\]' in selector or '\\.' in selector:
                raw = 'r'
            command = "%sself.click(%s'%s')" % (whitespace, raw, selector)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_id\\(\\"(\\S+)\\"\\)\\.submit\\(\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '#%s' % data.group(2).replace('#', '\\#')
            selector = selector.replace('[', '\\[').replace(']', '\\]')
            selector = selector.replace('.', '\\.')
            raw = ''
            if '\\[' in selector or '\\]' in selector or '\\.' in selector:
                raw = 'r'
            command = "%sself.submit(%s'%s')" % (whitespace, raw, selector)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_id\\(\\"(\\S+)\\"\\)\\.send_keys\\(\\"([\\S\\s]+)\\"\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '#%s' % data.group(2).replace('#', '\\#')
            selector = selector.replace('[', '\\[').replace(']', '\\]')
            selector = selector.replace('.', '\\.')
            raw = ''
            if '\\[' in selector or '\\]' in selector or '\\.' in selector:
                raw = 'r'
            text = data.group(3)
            command = "%sself.type(%s'%s', '%s')" % (whitespace, raw, selector, text)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_id\\(\\"(\\S+)\\"\\)\\.send_keys\\(Keys\\.([\\S]+)\\)\\s*$', line)
        if data:
            uses_keys = True
            whitespace = data.group(1)
            selector = '#%s' % data.group(2).replace('#', '\\#')
            selector = selector.replace('[', '\\[').replace(']', '\\]')
            selector = selector.replace('.', '\\.')
            raw = ''
            if '\\[' in selector or '\\]' in selector or '\\.' in selector:
                raw = 'r'
            key = 'Keys.%s' % data.group(3)
            command = "%sself.send_keys(%s'%s', %s)" % (whitespace, raw, selector, key)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_name\\(\\"(\\S+)\\"\\)\\.click\\(\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '[name="%s"]' % data.group(2)
            command = "%sself.click('%s')" % (whitespace, selector)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_name\\(\\"(\\S+)\\"\\)\\.submit\\(\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '[name="%s"]' % data.group(2)
            command = "%sself.submit('%s')" % (whitespace, selector)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_name\\(\\"(\\S+)\\"\\)\\.send_keys\\(\\"([\\S\\s]+)\\"\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '[name="%s"]' % data.group(2)
            text = data.group(3)
            command = "%sself.type('%s', '%s')" % (whitespace, selector, text)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_name\\(\\"(\\S+)\\"\\)\\.send_keys\\(Keys\\.([\\S]+)\\)\\s*$', line)
        if data:
            uses_keys = True
            whitespace = data.group(1)
            selector = '[name="%s"]' % data.group(2)
            key = 'Keys.%s' % data.group(3)
            command = "%sself.send_keys('%s', %s)" % (whitespace, selector, key)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_css_selector\\(\\"([\\S\\s]+)\\"\\)\\.click\\(\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '%s' % data.group(2)
            command = "%sself.click('%s')" % (whitespace, selector)
            if command.count('\\"') == command.count('"'):
                command = command.replace('\\"', '"')
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_css_selector\\(\\"([\\S\\s]+)\\"\\)\\.submit\\(\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '%s' % data.group(2)
            command = "%sself.submit('%s')" % (whitespace, selector)
            if command.count('\\"') == command.count('"'):
                command = command.replace('\\"', '"')
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_css_selector\\(\\"([\\S\\s]+)\\"\\)\\.send_keys\\(\\"([\\S\\s]+)\\"\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '%s' % data.group(2)
            text = data.group(3)
            command = "%sself.type('%s', '%s')" % (whitespace, selector, text)
            if command.count('\\"') == command.count('"'):
                command = command.replace('\\"', '"')
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_css_selector\\(\\"([\\S\\s]+)\\"\\)\\.send_keys\\(Keys\\.([\\S]+)\\)\\s*$', line)
        if data:
            uses_keys = True
            whitespace = data.group(1)
            selector = '%s' % data.group(2)
            key = 'Keys.%s' % data.group(3)
            command = "%sself.send_keys('%s', %s)" % (whitespace, selector, key)
            if command.count('\\"') == command.count('"'):
                command = command.replace('\\"', '"')
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_xpath\\(\\"([\\S\\s]+)\\"\\)\\.send_keys\\(\\"([\\S\\s]+)\\"\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '%s' % data.group(2)
            text = data.group(3)
            command = '%sself.type("%s", \'%s\')' % (whitespace, selector, text)
            if command.count('\\"') == command.count('"'):
                command = command.replace('\\"', '"')
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_xpath\\(\\"([\\S\\s]+)\\"\\)\\.send_keys\\(Keys\\.([\\S]+)\\)\\s*$', line)
        if data:
            uses_keys = True
            whitespace = data.group(1)
            selector = '%s' % data.group(2)
            key = 'Keys.%s' % data.group(3)
            command = '%sself.send_keys("%s", %s)' % (whitespace, selector, key)
            if command.count('\\"') == command.count('"'):
                command = command.replace('\\"', '"')
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)Select\\(driver\\.find_element_by_css_selector\\(\\"([\\S\\s]+)\\"\\)\\)\\.select_by_visible_text\\(\\"([\\S\\s]+)\\"\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '%s' % data.group(2)
            visible_text = '%s' % data.group(3)
            command = "%sself.select_option_by_text('%s', '%s')" % (whitespace, selector, visible_text)
            if command.count('\\"') == command.count('"'):
                command = command.replace('\\"', '"')
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)Select\\(driver\\.find_element_by_id\\(\\"([\\S\\s]+)\\"\\)\\)\\.select_by_visible_text\\(\\"([\\S\\s]+)\\"\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '#%s' % data.group(2).replace('#', '\\#')
            selector = selector.replace('[', '\\[').replace(']', '\\]')
            selector = selector.replace('.', '\\.')
            raw = ''
            if '\\[' in selector or '\\]' in selector or '\\.' in selector:
                raw = 'r'
            visible_text = '%s' % data.group(3)
            command = "%sself.select_option_by_text(%s'%s', '%s')" % (whitespace, raw, selector, visible_text)
            if command.count('\\"') == command.count('"'):
                command = command.replace('\\"', '"')
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)Select\\(driver\\.find_element_by_xpath\\(\\"([\\S\\s]+)\\"\\)\\)\\.select_by_visible_text\\(\\"([\\S\\s]+)\\"\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '%s' % data.group(2)
            visible_text = '%s' % data.group(3)
            command = '%sself.select_option_by_text("%s", \'%s\')' % (whitespace, selector, visible_text)
            if command.count('\\"') == command.count('"'):
                command = command.replace('\\"', '"')
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)Select\\(driver\\.find_element_by_name\\(\\"([\\S\\s]+)\\"\\)\\)\\.select_by_visible_text\\(\\"([\\S\\s]+)\\"\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            selector = '[name="%s"]' % data.group(2)
            visible_text = '%s' % data.group(3)
            command = "%sself.select_option_by_text('%s', '%s')" % (whitespace, selector, visible_text)
            if command.count('\\"') == command.count('"'):
                command = command.replace('\\"', '"')
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_xpath\\(u?\\"([\\S\\s]+)\\"\\)\\.click\\(\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            xpath = '%s' % data.group(2)
            if './/*[normalize-space(text())' in xpath and "normalize-space(.)='" in xpath:
                x_match = re.match("^[\\S\\s]+normalize-space\\(\\.\\)=\\'([\\S\\s]+)\\'\\]\\)[\\S\\s]+", xpath)
                if x_match:
                    partial_link_text = x_match.group(1)
                    xpath = 'partial_link=%s' % partial_link_text
            uni = ''
            if '(u"' in line:
                uni = 'u'
            command = '%sself.click(%s"%s")' % (whitespace, uni, xpath)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_xpath\\(u?\\"([\\S\\s]+)\\"\\)\\.submit\\(\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            xpath = '%s' % data.group(2)
            uni = ''
            if '(u"' in line:
                uni = 'u'
            command = '%sself.submit(%s"%s")' % (whitespace, uni, xpath)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)driver\\.find_element_by_link_text\\(u?\\"([\\S\\s]+)\\"\\)\\.click\\(\\)\\s*$', line)
        if data:
            whitespace = data.group(1)
            link_text = '%s' % data.group(2)
            uni = ''
            if '(u"' in line:
                uni = 'u'
            command = '%sself.click(%s"link=%s")' % (whitespace, uni, link_text)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)([\\S\\s]*)self\\.is_element_present\\(By.LINK_TEXT, u?\\"([\\S\\s]+)\\"\\)([\\S\\s]*)$', line)
        if data:
            whitespace = data.group(1)
            pre = data.group(2)
            link_text = '%s' % data.group(3)
            post = data.group(4)
            uni = ''
            if '(u"' in line:
                uni = 'u'
            command = '%s%sself.is_link_text_present(%s"%s")%s' % (whitespace, pre, uni, link_text, post)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)([\\S\\s]*)self\\.is_element_present\\(By.NAME, u?\\"([\\S\\s]+)\\"\\)([\\S\\s]*)$', line)
        if data:
            whitespace = data.group(1)
            pre = data.group(2)
            name = '%s' % data.group(3)
            post = data.group(4)
            uni = ''
            if '(u"' in line:
                uni = 'u'
            command = '%s%sself.is_element_present(\'[name="%s"]\')%s' % (whitespace, pre, name, post)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)([\\S\\s]*)self\\.is_element_present\\(By.ID, u?\\"([\\S\\s]+)\\"\\)([\\S\\s]*)$', line)
        if data:
            whitespace = data.group(1)
            pre = data.group(2)
            the_id = '%s' % data.group(3)
            post = data.group(4)
            uni = ''
            if '(u"' in line:
                uni = 'u'
            command = '%s%sself.is_element_present("#%s")%s' % (whitespace, pre, the_id, post)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)([\\S\\s]*)self\\.is_element_present\\(By.CLASS, u?\\"([\\S\\s]+)\\"\\)([\\S\\s]*)$', line)
        if data:
            whitespace = data.group(1)
            pre = data.group(2)
            the_class = '%s' % data.group(3)
            post = data.group(4)
            uni = ''
            if '(u"' in line:
                uni = 'u'
            command = '%s%sself.is_element_present(".%s")%s' % (whitespace, pre, the_class, post)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)([\\S\\s]*)self\\.is_element_present\\(By.CSS_SELECTOR, u?\\"([\\S\\s]+)\\"\\)([\\S\\s]*)$', line)
        if data:
            whitespace = data.group(1)
            pre = data.group(2)
            selector = '%s' % data.group(3)
            post = data.group(4)
            uni = ''
            if '(u"' in line:
                uni = 'u'
            command = '%s%sself.is_element_present("%s")%s' % (whitespace, pre, selector, post)
            seleniumbase_lines.append(command)
            continue
        data = re.match('^(\\s*)([\\S\\s]*)self\\.is_element_present\\(By.XPATH, u?\\"([\\S\\s]+)\\"\\)([\\S\\s]*)$', line)
        if data:
            whitespace = data.group(1)
            pre = data.group(2)
            xpath = '%s' % data.group(3)
            post = data.group(4)
            uni = ''
            if '(u"' in line:
                uni = 'u'
            command = '%s%sself.is_element_present("%s")%s' % (whitespace, pre, xpath, post)
            seleniumbase_lines.append(command)
            continue
        if 'self.base_url' in line:
            line = line.replace('self.base_url', '"%s"' % ide_base_url)
        if 'driver.' in line and 'self.driver' not in line:
            line = line.replace('driver.', 'self.driver.')
        seleniumbase_lines.append(line)
    in_inefficient_wait = False
    whitespace = ''
    lines = seleniumbase_lines
    seleniumbase_lines = []
    for line in lines:
        data = re.match('^(\\s*)for i in range\\(60\\):\\s*$', line)
        if data:
            in_inefficient_wait = True
            whitespace = data.group(1)
            continue
        data = re.match('^(\\s*)else: self.fail\\("time out"\\)\\s*$', line)
        if data:
            in_inefficient_wait = False
            continue
        if in_inefficient_wait:
            data = re.match('^\\s*if self.is_element_present\\("([\\S\\s]+)"\\): break\\s*$', line)
            if data:
                selector = data.group(1)
                command = '%sself.wait_for_element("%s")' % (whitespace, selector)
                seleniumbase_lines.append(command)
                continue
            data = re.match("^\\s*if self.is_element_present\\('([\\S\\s]+)'\\): break\\s*$", line)
            if data:
                selector = data.group(1)
                command = "%sself.wait_for_element('%s')" % (whitespace, selector)
                seleniumbase_lines.append(command)
                continue
            data = re.match('^\\s*if self.is_link_text_present\\("([\\S\\s]+)"\\): break\\s*$', line)
            if data:
                uni = ''
                if '(u"' in line:
                    uni = 'u'
                link_text = data.group(1)
                command = '%sself.wait_for_link_text(%s"%s")' % (whitespace, uni, link_text)
                seleniumbase_lines.append(command)
                continue
        else:
            seleniumbase_lines.append(line)
            continue
    lines = seleniumbase_lines
    for line_num in range(len(lines)):
        if 'Select(self.driver' in lines[line_num]:
            uses_select = True
    lines = seleniumbase_lines
    seleniumbase_lines = []
    num_lines = len(lines)
    for line_num in range(len(lines)):
        data = re.match('^\\s*self.wait_for_element\\((["|\'])([\\S\\s]+)(["|\'])\\)\\s*$', lines[line_num])
        if data:
            selector = data.group(2)
            selector = re.escape(selector)
            selector = js_utils.escape_quotes_if_needed(selector)
            if int(line_num) < num_lines - 1:
                regex_string = '^\\s*self.click\\(["|\']' + selector + '["|\']\\)\\s*$'
                data2 = re.match(regex_string, lines[line_num + 1])
                if data2:
                    continue
                regex_string = '^\\s*self.type\\(["|\']' + selector + '' + '["|\'], [\\S\\s]+\\)\\s*$'
                data2 = re.match(regex_string, lines[line_num + 1])
                if data2:
                    continue
        seleniumbase_lines.append(lines[line_num])
    lines = seleniumbase_lines
    seleniumbase_lines = []
    num_lines = len(lines)
    for line_num in range(len(lines)):
        data = re.match('^\\s*self.click\\((["|\'])([\\S\\s]+)(["|\'])\\)\\s*$', lines[line_num])
        if data:
            selector = data.group(2)
            selector = re.escape(selector)
            selector = js_utils.escape_quotes_if_needed(selector)
            if int(line_num) < num_lines - 1:
                regex_string = '^\\s*self.type\\(["|\']' + selector + '' + '["|\'], [\\S\\s]+\\)\\s*$'
                data2 = re.match(regex_string, lines[line_num + 1])
                if data2:
                    continue
        seleniumbase_lines.append(lines[line_num])
    lines = seleniumbase_lines
    seleniumbase_lines = []
    num_lines = len(lines)
    for line_num in range(len(lines)):
        data = re.match('^\\s*self.wait_for_link_text\\((["|\'])([\\S\\s]+)(["|\'])\\)\\s*$', lines[line_num])
        if data:
            link_text = data.group(2)
            link_text = re.escape(link_text)
            link_text = js_utils.escape_quotes_if_needed(link_text)
            if int(line_num) < num_lines - 2:
                regex_string = '^\\s*self.click\\(["|\']link=' + link_text + '["|\']\\)\\s*$'
                data2 = re.match(regex_string, lines[line_num + 1])
                if data2:
                    continue
        seleniumbase_lines.append(lines[line_num])
    seleniumbase_code = ''
    if uses_keys:
        seleniumbase_code += 'from selenium.webdriver.common.keys import Keys\n'
    if uses_select:
        seleniumbase_code += 'from selenium.webdriver.support.ui import Select\n'
    for line in seleniumbase_lines:
        seleniumbase_code += line
        seleniumbase_code += '\n'
    base_file_name = webdriver_python_file.split('.py')[0]
    converted_file_name = base_file_name + '_SB.py'
    out_file = codecs.open(converted_file_name, 'w+', encoding='utf-8')
    out_file.writelines(seleniumbase_code)
    out_file.close()
    print('\n>>> [%s] was created from [%s]\n' % (converted_file_name, webdriver_python_file))
if __name__ == '__main__':
    main()