import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yt_dlp
BASH_COMPLETION_FILE = 'completions/bash/yt-dlp'
BASH_COMPLETION_TEMPLATE = 'devscripts/bash-completion.in'

def build_completion(opt_parser):
    if False:
        return 10
    opts_flag = []
    for group in opt_parser.option_groups:
        for option in group.option_list:
            opts_flag.append(option.get_opt_string())
    with open(BASH_COMPLETION_TEMPLATE) as f:
        template = f.read()
    with open(BASH_COMPLETION_FILE, 'w') as f:
        filled_template = template.replace('{{flags}}', ' '.join(opts_flag))
        f.write(filled_template)
parser = yt_dlp.parseOpts(ignore_config_files=True)[0]
build_completion(parser)