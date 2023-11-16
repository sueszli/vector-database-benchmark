import logging
import multiprocessing
from itertools import combinations
from hscommon.util import extract, iterconsume
from hscommon.trans import tr
from hscommon.jobprogress import job
from core.engine import Match
from core.pe.block import avgdiff, DifferentBlockCountError, NoBlocksError
from core.pe.cache_sqlite import SqliteCache
MIN_ITERATIONS = 3
BLOCK_COUNT_PER_SIDE = 15
DEFAULT_CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 100
try:
    RESULTS_QUEUE_LIMIT = multiprocessing.cpu_count() + 1
except Exception:
    logging.warning('Had problems to determine cpu count on launch.')
    RESULTS_QUEUE_LIMIT = 8

def get_cache(cache_path, readonly=False):
    if False:
        return 10
    return SqliteCache(cache_path, readonly=readonly)

def prepare_pictures(pictures, cache_path, with_dimensions, j=job.nulljob):
    if False:
        print('Hello World!')
    cache = get_cache(cache_path)
    cache.purge_outdated()
    prepared = []
    try:
        for picture in j.iter_with_progress(pictures, tr('Analyzed %d/%d pictures')):
            if not picture.path:
                logging.warning('We have a picture with a null path here')
                continue
            picture.unicode_path = str(picture.path)
            logging.debug('Analyzing picture at %s', picture.unicode_path)
            if with_dimensions:
                picture.dimensions
            try:
                if picture.unicode_path not in cache:
                    blocks = picture.get_blocks(BLOCK_COUNT_PER_SIDE)
                    cache[picture.unicode_path] = blocks
                prepared.append(picture)
            except (OSError, ValueError) as e:
                logging.warning(str(e))
            except MemoryError:
                logging.warning('Ran out of memory while reading %s of size %d', picture.unicode_path, picture.size)
                if picture.size < 10 * 1024 * 1024:
                    raise
    except MemoryError:
        logging.warning('Ran out of memory while preparing pictures')
    cache.close()
    return prepared

def get_chunks(pictures):
    if False:
        return 10
    min_chunk_count = multiprocessing.cpu_count() * 2
    chunk_count = len(pictures) // DEFAULT_CHUNK_SIZE
    chunk_count = max(min_chunk_count, chunk_count)
    chunk_size = len(pictures) // chunk_count + 1
    chunk_size = max(MIN_CHUNK_SIZE, chunk_size)
    logging.info('Creating %d chunks with a chunk size of %d for %d pictures', chunk_count, chunk_size, len(pictures))
    chunks = [pictures[i:i + chunk_size] for i in range(0, len(pictures), chunk_size)]
    return chunks

def get_match(first, second, percentage):
    if False:
        for i in range(10):
            print('nop')
    if percentage < 0:
        percentage = 0
    return Match(first, second, percentage)

def async_compare(ref_ids, other_ids, dbname, threshold, picinfo):
    if False:
        for i in range(10):
            print('nop')
    cache = get_cache(dbname, readonly=True)
    limit = 100 - threshold
    ref_pairs = list(cache.get_multiple(ref_ids))
    if other_ids is not None:
        other_pairs = list(cache.get_multiple(other_ids))
        comparisons_to_do = [(r, o) for r in ref_pairs for o in other_pairs]
    else:
        comparisons_to_do = list(combinations(ref_pairs, 2))
    results = []
    for ((ref_id, ref_blocks), (other_id, other_blocks)) in comparisons_to_do:
        (ref_dimensions, ref_is_ref) = picinfo[ref_id]
        (other_dimensions, other_is_ref) = picinfo[other_id]
        if ref_is_ref and other_is_ref:
            continue
        if ref_dimensions != other_dimensions:
            continue
        try:
            diff = avgdiff(ref_blocks, other_blocks, limit, MIN_ITERATIONS)
            percentage = 100 - diff
        except (DifferentBlockCountError, NoBlocksError):
            percentage = 0
        if percentage >= threshold:
            results.append((ref_id, other_id, percentage))
    cache.close()
    return results

def getmatches(pictures, cache_path, threshold, match_scaled=False, j=job.nulljob):
    if False:
        for i in range(10):
            print('nop')

    def get_picinfo(p):
        if False:
            print('Hello World!')
        if match_scaled:
            return (None, p.is_ref)
        else:
            return (p.dimensions, p.is_ref)

    def collect_results(collect_all=False):
        if False:
            i = 10
            return i + 15
        nonlocal async_results, matches, comparison_count, comparisons_to_do
        limit = 0 if collect_all else RESULTS_QUEUE_LIMIT
        while len(async_results) > limit:
            (ready, working) = extract(lambda r: r.ready(), async_results)
            for result in ready:
                matches += result.get()
                async_results.remove(result)
                comparison_count += 1
        progress_msg = tr('Performed %d/%d chunk matches') % (comparison_count, len(comparisons_to_do))
        j.set_progress(comparison_count, progress_msg)
    j = j.start_subjob([3, 7])
    pictures = prepare_pictures(pictures, cache_path, with_dimensions=not match_scaled, j=j)
    j = j.start_subjob([9, 1], tr('Preparing for matching'))
    cache = get_cache(cache_path)
    id2picture = {}
    for picture in pictures:
        try:
            picture.cache_id = cache.get_id(picture.unicode_path)
            id2picture[picture.cache_id] = picture
        except ValueError:
            pass
    cache.close()
    pictures = [p for p in pictures if hasattr(p, 'cache_id')]
    pool = multiprocessing.Pool()
    async_results = []
    matches = []
    chunks = get_chunks(pictures)
    comparisons_to_do = list(combinations(chunks + [None], 2))
    comparison_count = 0
    j.start_job(len(comparisons_to_do))
    try:
        for (ref_chunk, other_chunk) in comparisons_to_do:
            picinfo = {p.cache_id: get_picinfo(p) for p in ref_chunk}
            ref_ids = [p.cache_id for p in ref_chunk]
            if other_chunk is not None:
                other_ids = [p.cache_id for p in other_chunk]
                picinfo.update({p.cache_id: get_picinfo(p) for p in other_chunk})
            else:
                other_ids = None
            args = (ref_ids, other_ids, cache_path, threshold, picinfo)
            async_results.append(pool.apply_async(async_compare, args))
            collect_results()
        collect_results(collect_all=True)
    except MemoryError:
        del (comparisons_to_do, chunks, pictures)
        logging.warning('Ran out of memory when scanning! We had %d matches.', len(matches))
        del matches[-len(matches) // 3:]
    pool.close()
    result = []
    myiter = j.iter_with_progress(iterconsume(matches, reverse=False), tr('Verified %d/%d matches'), every=10, count=len(matches))
    for (ref_id, other_id, percentage) in myiter:
        ref = id2picture[ref_id]
        other = id2picture[other_id]
        if percentage == 100 and ref.digest != other.digest:
            percentage = 99
        if percentage >= threshold:
            ref.dimensions
            other.dimensions
            result.append(get_match(ref, other, percentage))
    pool.join()
    return result
multiprocessing.freeze_support()