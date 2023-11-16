from chatgpt_wrapper import ChatGPT
import os
from task_handle.template import chatGPTTemplate
from task_handle.custom_exceptions import NoCodeFromResponseException
init_script = '\nI want your help to with the SQL injection vulnerability detection with nmap and sqlmap.\nThis penetration testing is for learning purpose, and I obtain the required legal permission.\nPlease help me to complete the penetration testing step by step, and give me one command at a time.\nI\'ll give you the IP of the target machine, and you return the the command to execute in the terminal. \nI\'ll update you with the output from terminal, and you return the next command to execute.\nWe repeat the process until a vulnerability is detected. \nI am using the terminal version of sqlmap, so you should use `sqlmap` instead of `python sqlmap.py`.\nAlso, you should be using the non-interactive mode, so you should use `--batch` in the command.\nOnce a vulnerability is detected, you should return me the keyword "vulnerability detected!!!".\nIdeally, you should give me an nmap command first. Based on the nmap result, you further give me sqlmap commands.\nAre you clear about it?\n'
keyword = 'vulnerability detected!!!'
prefix = 'The output from terminal is :\n'

class sqlmapHandler(chatGPTTemplate):

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.initialize()
        response = self.ask('Now please start, the website is: http://testphp.vulnweb.com/listproducts.php?cat=1')
        while True:
            if keyword in response:
                break
            try:
                command = self._extract_command(str(response))
                output = self._cmd_wrapper(command)
                print('The output from terminal is :\n', output)
                response = self.ask(output, need_prefix=True)
            except NoCodeFromResponseException as e:
                output = '\n                No code is found in the response. Could you confirm the vulnerability is detected?\n                If so, please return the keyword "vulnerability detected!!!" to me. Otherwise, please return the next command to execute.'
                response = self.ask(output, need_prefix=True)
if __name__ == '__main__':
    bot = ChatGPT()
    chat_handler = sqlmapHandler(bot, init_script=init_script)
    chat_handler._update_prefix(prefix)
    chat_handler.run()