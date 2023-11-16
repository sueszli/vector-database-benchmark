import ssl
import sys
import httpx
import paddle

def download_file():
    if False:
        print('Hello World!')
    'Get disabled unit tests'
    ssl._create_default_https_context = ssl._create_unverified_context
    sysstr = sys.platform
    if sysstr == 'win32':
        url = 'https://sys-p0.bj.bcebos.com/prec/{}'.format('disable_ut_win')
    else:
        url = 'https://sys-p0.bj.bcebos.com/prec/{}'.format('disable_ut')
    if paddle.is_compiled_with_rocm():
        url = 'https://sys-p0.bj.bcebos.com/prec/{}'.format('disable_ut_rocm')
    f = httpx.get(url, timeout=None, follow_redirects=True)
    data = f.text
    status_code = f.status_code
    if len(data.strip()) == 0 or status_code != 200:
        sys.exit(1)
    else:
        lt = data.strip().split('\n')
        lt = '^' + '$|^'.join(lt) + '$'
        print(lt)
        sys.exit(0)
if __name__ == '__main__':
    try:
        download_file()
    except Exception as e:
        print(e)
        sys.exit(1)