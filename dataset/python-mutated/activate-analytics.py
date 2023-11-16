"""
Copied from: https://github.com/jrieke/traingenerator

Update index.html from streamlit by
- adding tracking code for Google Analytics
- adding meta tags for search engines
- adding meta tags for social preview
WARNING: This changes your existing streamlit installation (specifically the file
static/index.html in streamlit's main folder). It should only be called once after
installation, so this file doesn't get cluttered!
The tag from Google Analytics (G-XXXXXXXXXX) has to be stored in an environment variable
GOOGLE_ANALYTICS_TAG (or in a .env file).
"""
import os
import sys
import streamlit as st

def replace_in_file(filename, oldvalue, newvalue):
    if False:
        return 10
    'Replace string in a file and optionally create backup_filename.'
    with open(filename, 'r') as f:
        filedata = f.read()
    filedata = filedata.replace(oldvalue, newvalue)
    with open(filename, 'w') as f:
        f.write(filedata)
st_dir = os.path.dirname(st.__file__)
index_filename = os.path.join(st_dir, 'static', 'index.html')
tag = os.getenv('GOOGLE_ANALYTICS_TAG')
if not tag:
    print('No tag provided, analytics is deactivated')
    sys.exit(1)
tracking_code = f"""<!-- Global site tag (gtag.js) - Google Analytics --><script async src="https://www.googletagmanager.com/gtag/js?id={tag}"></script><script>window.dataLayer = window.dataLayer || []; function gtag(){{dataLayer.push(arguments);}} gtag('js', new Date()); gtag('config', '{tag}');</script>"""
clarity_tag = os.getenv('CLARITY_TAG')
if clarity_tag:
    clarity_tracking_code = f'\n    <script type="text/javascript">\n        (function(c,l,a,r,i,t,y){{\n            c[a]=c[a]||function(){{(c[a].q=c[a].q||[]).push(arguments)}};\n            t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;\n            y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);\n        }})(window, document, "clarity", "script", "{clarity_tag}");\n    </script>\n    '
    tracking_code += clarity_tracking_code
size_before = os.stat(index_filename).st_size
replace_in_file(index_filename, '<head>', '<head>' + tracking_code)
size_after = os.stat(index_filename).st_size
META_TAGS = '\n<!-- Meta tags for search engines -->\n<meta name="description" content="Python functions with superpowers. Instantly deploy simple functions with REST API, UI, and more.">\n<!-- Meta tags for social preview -->\n<meta property="og:title" content="Opyrator Playground">\n<meta property="og:description" content="Python functions with superpowers">\n<meta property="og:url" content="https://github.com/ml-tooling/opyrator">\n<meta property="og:site_name" content="Opyrator Playground">\n<meta name="twitter:image:alt" content="Opyrator Playground">\n'
size_before = os.stat(index_filename).st_size
replace_in_file(index_filename, '<head>', '<head>' + META_TAGS)
size_after = os.stat(index_filename).st_size