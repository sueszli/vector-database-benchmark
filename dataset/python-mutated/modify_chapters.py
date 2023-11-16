import copy
import heapq
import os
from .common import PostProcessor
from .ffmpeg import FFmpegPostProcessor, FFmpegSubtitlesConvertorPP
from .sponsorblock import SponsorBlockPP
from ..utils import PostProcessingError, orderedSet, prepend_extension
_TINY_CHAPTER_DURATION = 1
DEFAULT_SPONSORBLOCK_CHAPTER_TITLE = '[SponsorBlock]: %(category_names)l'

class ModifyChaptersPP(FFmpegPostProcessor):

    def __init__(self, downloader, remove_chapters_patterns=None, remove_sponsor_segments=None, remove_ranges=None, *, sponsorblock_chapter_title=DEFAULT_SPONSORBLOCK_CHAPTER_TITLE, force_keyframes=False):
        if False:
            i = 10
            return i + 15
        FFmpegPostProcessor.__init__(self, downloader)
        self._remove_chapters_patterns = set(remove_chapters_patterns or [])
        self._remove_sponsor_segments = set(remove_sponsor_segments or []) - set(SponsorBlockPP.NON_SKIPPABLE_CATEGORIES.keys())
        self._ranges_to_remove = set(remove_ranges or [])
        self._sponsorblock_chapter_title = sponsorblock_chapter_title
        self._force_keyframes = force_keyframes

    @PostProcessor._restrict_to(images=False)
    def run(self, info):
        if False:
            while True:
                i = 10
        self._fixup_chapters(info)
        (chapters, sponsor_chapters) = self._mark_chapters_to_remove(copy.deepcopy(info.get('chapters')) or [], copy.deepcopy(info.get('sponsorblock_chapters')) or [])
        if not chapters and (not sponsor_chapters):
            return ([], info)
        real_duration = self._get_real_video_duration(info['filepath'])
        if not chapters:
            chapters = [{'start_time': 0, 'end_time': info.get('duration') or real_duration, 'title': info['title']}]
        (info['chapters'], cuts) = self._remove_marked_arrange_sponsors(chapters + sponsor_chapters)
        if not cuts:
            return ([], info)
        elif not info['chapters']:
            self.report_warning('You have requested to remove the entire video, which is not possible')
            return ([], info)
        (original_duration, info['duration']) = (info.get('duration'), info['chapters'][-1]['end_time'])
        if self._duration_mismatch(real_duration, original_duration, 1):
            if not self._duration_mismatch(real_duration, info['duration']):
                self.to_screen(f'Skipping {self.pp_key()} since the video appears to be already cut')
                return ([], info)
            if not info.get('__real_download'):
                raise PostProcessingError('Cannot cut video since the real and expected durations mismatch. Different chapters may have already been removed')
            else:
                self.write_debug('Expected and actual durations mismatch')
        concat_opts = self._make_concat_opts(cuts, real_duration)
        self.write_debug('Concat spec = %s' % ', '.join((f"{c.get('inpoint', 0.0)}-{c.get('outpoint', 'inf')}" for c in concat_opts)))

        def remove_chapters(file, is_sub):
            if False:
                while True:
                    i = 10
            return (file, self.remove_chapters(file, cuts, concat_opts, self._force_keyframes and (not is_sub)))
        in_out_files = [remove_chapters(info['filepath'], False)]
        in_out_files.extend((remove_chapters(in_file, True) for in_file in self._get_supported_subs(info)))
        files_to_remove = []
        for (in_file, out_file) in in_out_files:
            mtime = os.stat(in_file).st_mtime
            uncut_file = prepend_extension(in_file, 'uncut')
            os.replace(in_file, uncut_file)
            os.replace(out_file, in_file)
            self.try_utime(in_file, mtime, mtime)
            files_to_remove.append(uncut_file)
        return (files_to_remove, info)

    def _mark_chapters_to_remove(self, chapters, sponsor_chapters):
        if False:
            for i in range(10):
                print('nop')
        if self._remove_chapters_patterns:
            warn_no_chapter_to_remove = True
            if not chapters:
                self.to_screen('Chapter information is unavailable')
                warn_no_chapter_to_remove = False
            for c in chapters:
                if any((regex.search(c['title']) for regex in self._remove_chapters_patterns)):
                    c['remove'] = True
                    warn_no_chapter_to_remove = False
            if warn_no_chapter_to_remove:
                self.to_screen('There are no chapters matching the regex')
        if self._remove_sponsor_segments:
            warn_no_chapter_to_remove = True
            if not sponsor_chapters:
                self.to_screen('SponsorBlock information is unavailable')
                warn_no_chapter_to_remove = False
            for c in sponsor_chapters:
                if c['category'] in self._remove_sponsor_segments:
                    c['remove'] = True
                    warn_no_chapter_to_remove = False
            if warn_no_chapter_to_remove:
                self.to_screen('There are no matching SponsorBlock chapters')
        sponsor_chapters.extend(({'start_time': start, 'end_time': end, 'category': 'manually_removed', '_categories': [('manually_removed', start, end, 'Manually removed')], 'remove': True} for (start, end) in self._ranges_to_remove))
        return (chapters, sponsor_chapters)

    def _get_supported_subs(self, info):
        if False:
            for i in range(10):
                print('nop')
        for sub in (info.get('requested_subtitles') or {}).values():
            sub_file = sub.get('filepath')
            if not sub_file or not os.path.exists(sub_file):
                continue
            ext = sub['ext']
            if ext not in FFmpegSubtitlesConvertorPP.SUPPORTED_EXTS:
                self.report_warning(f'Cannot remove chapters from external {ext} subtitles; "{sub_file}" is now out of sync')
                continue
            yield sub_file

    def _remove_marked_arrange_sponsors(self, chapters):
        if False:
            print('Hello World!')
        cuts = []

        def append_cut(c):
            if False:
                for i in range(10):
                    print('nop')
            assert 'remove' in c, 'Not a cut is appended to cuts'
            last_to_cut = cuts[-1] if cuts else None
            if last_to_cut and last_to_cut['end_time'] >= c['start_time']:
                last_to_cut['end_time'] = max(last_to_cut['end_time'], c['end_time'])
            else:
                cuts.append(c)
            return len(cuts) - 1

        def excess_duration(c):
            if False:
                i = 10
                return i + 15
            (cut_idx, excess) = (c.pop('cut_idx', len(cuts)), 0)
            while cut_idx < len(cuts):
                cut = cuts[cut_idx]
                if cut['start_time'] >= c['end_time']:
                    break
                if cut['end_time'] > c['start_time']:
                    excess += min(cut['end_time'], c['end_time'])
                    excess -= max(cut['start_time'], c['start_time'])
                cut_idx += 1
            return excess
        new_chapters = []

        def append_chapter(c):
            if False:
                print('Hello World!')
            assert 'remove' not in c, 'Cut is appended to chapters'
            length = c['end_time'] - c['start_time'] - excess_duration(c)
            if length <= 0:
                return
            start = new_chapters[-1]['end_time'] if new_chapters else 0
            c.update(start_time=start, end_time=start + length)
            new_chapters.append(c)
        chapters = [(c['start_time'], i, c) for (i, c) in enumerate(chapters)]
        heapq.heapify(chapters)
        (_, cur_i, cur_chapter) = heapq.heappop(chapters)
        while chapters:
            (_, i, c) = heapq.heappop(chapters)
            if cur_chapter['end_time'] <= c['start_time']:
                (append_chapter if 'remove' not in cur_chapter else append_cut)(cur_chapter)
                (cur_i, cur_chapter) = (i, c)
                continue
            if 'remove' in cur_chapter:
                if 'remove' in c:
                    cur_chapter['end_time'] = max(cur_chapter['end_time'], c['end_time'])
                elif cur_chapter['end_time'] < c['end_time']:
                    c['start_time'] = cur_chapter['end_time']
                    c['_was_cut'] = True
                    heapq.heappush(chapters, (c['start_time'], i, c))
            elif 'remove' in c:
                cur_chapter['_was_cut'] = True
                if cur_chapter['end_time'] <= c['end_time']:
                    cur_chapter['end_time'] = c['start_time']
                    append_chapter(cur_chapter)
                    (cur_i, cur_chapter) = (i, c)
                    continue
                if '_categories' in cur_chapter:
                    after_c = dict(cur_chapter, start_time=c['end_time'], _categories=[])
                    cur_cats = []
                    for cat_start_end in cur_chapter['_categories']:
                        if cat_start_end[1] < c['start_time']:
                            cur_cats.append(cat_start_end)
                        if cat_start_end[2] > c['end_time']:
                            after_c['_categories'].append(cat_start_end)
                    cur_chapter['_categories'] = cur_cats
                    if cur_chapter['_categories'] != after_c['_categories']:
                        heapq.heappush(chapters, (after_c['start_time'], cur_i, after_c))
                        cur_chapter['end_time'] = c['start_time']
                        append_chapter(cur_chapter)
                        (cur_i, cur_chapter) = (i, c)
                        continue
                cur_chapter.setdefault('cut_idx', append_cut(c))
            elif '_categories' in cur_chapter and '_categories' not in c:
                if cur_chapter['end_time'] < c['end_time']:
                    c['start_time'] = cur_chapter['end_time']
                    c['_was_cut'] = True
                    heapq.heappush(chapters, (c['start_time'], i, c))
            else:
                assert '_categories' in c, 'Normal chapters overlap'
                cur_chapter['_was_cut'] = True
                c['_was_cut'] = True
                if cur_chapter['end_time'] > c['end_time']:
                    after_c = dict(copy.deepcopy(cur_chapter), start_time=c['end_time'])
                    heapq.heappush(chapters, (after_c['start_time'], cur_i, after_c))
                elif c['end_time'] > cur_chapter['end_time']:
                    after_cur = dict(copy.deepcopy(c), start_time=cur_chapter['end_time'])
                    heapq.heappush(chapters, (after_cur['start_time'], cur_i, after_cur))
                    c['end_time'] = cur_chapter['end_time']
                if '_categories' in cur_chapter:
                    c['_categories'] = cur_chapter['_categories'] + c['_categories']
                if 'cut_idx' in cur_chapter:
                    c['cut_idx'] = cur_chapter['cut_idx']
                cur_chapter['end_time'] = c['start_time']
                append_chapter(cur_chapter)
                (cur_i, cur_chapter) = (i, c)
        (append_chapter if 'remove' not in cur_chapter else append_cut)(cur_chapter)
        return (self._remove_tiny_rename_sponsors(new_chapters), cuts)

    def _remove_tiny_rename_sponsors(self, chapters):
        if False:
            return 10
        new_chapters = []
        for (i, c) in enumerate(chapters):
            if ('_was_cut' in c or '_categories' in c) and c['end_time'] - c['start_time'] < _TINY_CHAPTER_DURATION:
                if not new_chapters:
                    if i < len(chapters) - 1:
                        chapters[i + 1]['start_time'] = c['start_time']
                        continue
                else:
                    old_c = new_chapters[-1]
                    if i < len(chapters) - 1:
                        next_c = chapters[i + 1]
                        prev_is_sponsor = 'categories' in old_c
                        next_is_sponsor = '_categories' in next_c
                        if '_categories' not in c and prev_is_sponsor and (not next_is_sponsor) or ('_categories' in c and (not prev_is_sponsor) and next_is_sponsor):
                            next_c['start_time'] = c['start_time']
                            continue
                    old_c['end_time'] = c['end_time']
                    continue
            c.pop('_was_cut', None)
            cats = c.pop('_categories', None)
            if cats:
                (category, _, _, category_name) = min(cats, key=lambda c: c[2] - c[1])
                c.update({'category': category, 'categories': orderedSet((x[0] for x in cats)), 'name': category_name, 'category_names': orderedSet((x[3] for x in cats))})
                c['title'] = self._downloader.evaluate_outtmpl(self._sponsorblock_chapter_title, c.copy())
                if new_chapters and 'categories' in new_chapters[-1] and (new_chapters[-1]['title'] == c['title']):
                    new_chapters[-1]['end_time'] = c['end_time']
                    continue
            new_chapters.append(c)
        return new_chapters

    def remove_chapters(self, filename, ranges_to_cut, concat_opts, force_keyframes=False):
        if False:
            i = 10
            return i + 15
        in_file = filename
        out_file = prepend_extension(in_file, 'temp')
        if force_keyframes:
            in_file = self.force_keyframes(in_file, (t for c in ranges_to_cut for t in (c['start_time'], c['end_time'])))
        self.to_screen(f'Removing chapters from {filename}')
        self.concat_files([in_file] * len(concat_opts), out_file, concat_opts)
        if in_file != filename:
            self._delete_downloaded_files(in_file, msg=None)
        return out_file

    @staticmethod
    def _make_concat_opts(chapters_to_remove, duration):
        if False:
            for i in range(10):
                print('nop')
        opts = [{}]
        for s in chapters_to_remove:
            if s['start_time'] == 0:
                opts[-1]['inpoint'] = f"{s['end_time']:.6f}"
                continue
            opts[-1]['outpoint'] = f"{s['start_time']:.6f}"
            if s['end_time'] < duration:
                opts.append({'inpoint': f"{s['end_time']:.6f}"})
        return opts