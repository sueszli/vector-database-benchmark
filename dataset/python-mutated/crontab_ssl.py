import os, json, sys, time
os.chdir('/www/server/panel')
if not 'class/' in sys.path:
    sys.path.insert(0, 'class/')
import public
sys.path.append('.')
import panelSSL
import panelSite

class dict_obj:

    def __contains__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        setattr(self, key, value)

    def __getitem__(self, key):
        if False:
            return 10
        return getattr(self, key, None)

    def __delitem__(self, key):
        if False:
            print('Hello World!')
        delattr(self, key)

    def __delattr__(self, key):
        if False:
            i = 10
            return i + 15
        delattr(self, key)

    def get_items(self):
        if False:
            while True:
                i = 10
        return self
if __name__ == '__main__':
    get = dict_obj()
    obj = panelSSL.panelSSL()
    CertList = obj.GetCertList(get)
    cmd_list = json.loads(public.ReadFile('/www/server/panel/vhost/crontab.json'))
    panelSite_ = panelSite.panelSite()
    for i in CertList:
        timeArray = time.strptime(i['notAfter'], '%Y-%m-%d')
        timestamp = time.mktime(timeArray)
        if int(timestamp) - time.time() < 86400 * 30:
            subject = i['subject']
            for j in cmd_list:
                if subject == j['siteName']:
                    cmd = j['cmd']
                    public.ExecShell(cmd)
                    get.siteName = subject
                    result = panelSite_.save_cert(get)
                    public.serviceReload()