"""Create a tag for the PYPI nightly version

Increment the version number, add a dev suffix and add todays date
"""
from datetime import datetime
import packaging.version
import pytz
import streamlit.version

def create_tag():
    if False:
        while True:
            i = 10
    'Create tag with updated version, a suffix and date.'
    current_version = streamlit.version._get_latest_streamlit_version()
    version_with_inc_micro = (current_version.major, current_version.minor, current_version.micro + 1)
    version_with_date = '.'.join([str(x) for x in version_with_inc_micro]) + '.dev' + datetime.now(pytz.timezone('US/Pacific')).strftime('%Y%m%d')
    packaging.version.Version(version_with_date)
    return version_with_date
if __name__ == '__main__':
    tag = create_tag()
    print(tag)