import io
import quopri
import re
import uuid
from .fragment import FragmentFD
from ..compat import imghdr
from ..utils import escapeHTML, formatSeconds, srt_subtitles_timecode, urljoin
from ..version import __version__ as YT_DLP_VERSION

class MhtmlFD(FragmentFD):
    _STYLESHEET = 'html, body {\n    margin: 0;\n    padding: 0;\n    height: 100vh;\n}\n\nhtml {\n    overflow-y: scroll;\n    scroll-snap-type: y mandatory;\n}\n\nbody {\n    scroll-snap-type: y mandatory;\n    display: flex;\n    flex-flow: column;\n}\n\nbody > figure {\n    max-width: 100vw;\n    max-height: 100vh;\n    scroll-snap-align: center;\n}\n\nbody > figure > figcaption {\n    text-align: center;\n    height: 2.5em;\n}\n\nbody > figure > img {\n    display: block;\n    margin: auto;\n    max-width: 100%;\n    max-height: calc(100vh - 5em);\n}\n'
    _STYLESHEET = re.sub('\\s+', ' ', _STYLESHEET)
    _STYLESHEET = re.sub('\\B \\B|(?<=[\\w\\-]) (?=[^\\w\\-])|(?<=[^\\w\\-]) (?=[\\w\\-])', '', _STYLESHEET)

    @staticmethod
    def _escape_mime(s):
        if False:
            print('Hello World!')
        return '=?utf-8?Q?' + b''.join((bytes((b,)) if b >= 32 else b'=%02X' % b for b in quopri.encodestring(s.encode(), header=True))).decode('us-ascii') + '?='

    def _gen_cid(self, i, fragment, frag_boundary):
        if False:
            for i in range(10):
                print('nop')
        return '%u.%s@yt-dlp.github.io.invalid' % (i, frag_boundary)

    def _gen_stub(self, *, fragments, frag_boundary, title):
        if False:
            return 10
        output = io.StringIO()
        output.write('<!DOCTYPE html><html><head><meta name="generator" content="yt-dlp {version}"><title>{title}</title><style>{styles}</style><body>'.format(version=escapeHTML(YT_DLP_VERSION), styles=self._STYLESHEET, title=escapeHTML(title)))
        t0 = 0
        for (i, frag) in enumerate(fragments):
            output.write('<figure>')
            try:
                t1 = t0 + frag['duration']
                output.write('<figcaption>Slide #{num}: {t0} – {t1} (duration: {duration})</figcaption>'.format(num=i + 1, t0=srt_subtitles_timecode(t0), t1=srt_subtitles_timecode(t1), duration=formatSeconds(frag['duration'], msec=True)))
            except (KeyError, ValueError, TypeError):
                t1 = None
                output.write('<figcaption>Slide #{num}</figcaption>'.format(num=i + 1))
            output.write('<img src="cid:{cid}">'.format(cid=self._gen_cid(i, frag, frag_boundary)))
            output.write('</figure>')
            t0 = t1
        return output.getvalue()

    def real_download(self, filename, info_dict):
        if False:
            while True:
                i = 10
        fragment_base_url = info_dict.get('fragment_base_url')
        fragments = info_dict['fragments'][:1] if self.params.get('test', False) else info_dict['fragments']
        title = info_dict.get('title', info_dict['format_id'])
        origin = info_dict.get('webpage_url', info_dict['url'])
        ctx = {'filename': filename, 'total_frags': len(fragments)}
        self._prepare_and_start_frag_download(ctx, info_dict)
        extra_state = ctx.setdefault('extra_state', {'header_written': False, 'mime_boundary': str(uuid.uuid4()).replace('-', '')})
        frag_boundary = extra_state['mime_boundary']
        if not extra_state['header_written']:
            stub = self._gen_stub(fragments=fragments, frag_boundary=frag_boundary, title=title)
            ctx['dest_stream'].write('MIME-Version: 1.0\r\nFrom: <nowhere@yt-dlp.github.io.invalid>\r\nTo: <nowhere@yt-dlp.github.io.invalid>\r\nSubject: {title}\r\nContent-type: multipart/related; boundary="{boundary}"; type="text/html"\r\nX.yt-dlp.Origin: {origin}\r\n\r\n--{boundary}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {length}\r\n\r\n{stub}\r\n'.format(origin=origin, boundary=frag_boundary, length=len(stub), title=self._escape_mime(title), stub=stub).encode())
            extra_state['header_written'] = True
        for (i, fragment) in enumerate(fragments):
            if i + 1 <= ctx['fragment_index']:
                continue
            fragment_url = fragment.get('url')
            if not fragment_url:
                assert fragment_base_url
                fragment_url = urljoin(fragment_base_url, fragment['path'])
            success = self._download_fragment(ctx, fragment_url, info_dict)
            if not success:
                continue
            frag_content = self._read_fragment(ctx)
            frag_header = io.BytesIO()
            frag_header.write(b'--%b\r\n' % frag_boundary.encode('us-ascii'))
            frag_header.write(b'Content-ID: <%b>\r\n' % self._gen_cid(i, fragment, frag_boundary).encode('us-ascii'))
            frag_header.write(b'Content-type: %b\r\n' % f"image/{imghdr.what(h=frag_content) or 'jpeg'}".encode())
            frag_header.write(b'Content-length: %u\r\n' % len(frag_content))
            frag_header.write(b'Content-location: %b\r\n' % fragment_url.encode('us-ascii'))
            frag_header.write(b'X.yt-dlp.Duration: %f\r\n' % fragment['duration'])
            frag_header.write(b'\r\n')
            self._append_fragment(ctx, frag_header.getvalue() + frag_content + b'\r\n')
        ctx['dest_stream'].write(b'--%b--\r\n\r\n' % frag_boundary.encode('us-ascii'))
        return self._finish_frag_download(ctx, info_dict)