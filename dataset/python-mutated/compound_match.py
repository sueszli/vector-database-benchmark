"""
This is the default compound matcher function.
"""
import logging
import salt.loader
import salt.utils.minions
HAS_RANGE = False
try:
    import seco.range
    HAS_RANGE = True
except ImportError:
    pass
log = logging.getLogger(__name__)

def _load_matchers(opts):
    if False:
        i = 10
        return i + 15
    "\n    Store matchers in __context__ so they're only loaded once\n    "
    __context__['matchers'] = salt.loader.matchers(opts)

def match(tgt, opts=None, minion_id=None):
    if False:
        i = 10
        return i + 15
    '\n    Runs the compound target check\n    '
    if not opts:
        opts = __opts__
    nodegroups = opts.get('nodegroups', {})
    if 'matchers' not in __context__:
        _load_matchers(opts)
    if not minion_id:
        minion_id = opts.get('id')
    if not isinstance(tgt, str) and (not isinstance(tgt, (list, tuple))):
        log.error('Compound target received that is neither string, list nor tuple')
        return False
    log.debug('compound_match: %s ? %s', minion_id, tgt)
    ref = {'G': 'grain', 'P': 'grain_pcre', 'I': 'pillar', 'J': 'pillar_pcre', 'L': 'list', 'N': None, 'S': 'ipcidr', 'E': 'pcre'}
    if HAS_RANGE:
        ref['R'] = 'range'
    results = []
    opers = ['and', 'or', 'not', '(', ')']
    if isinstance(tgt, str):
        words = tgt.split()
    else:
        words = tgt[:]
    while words:
        word = words.pop(0)
        target_info = salt.utils.minions.parse_target(word)
        if word in opers:
            if results:
                if results[-1] == '(' and word in ('and', 'or'):
                    log.error('Invalid beginning operator after "(": %s', word)
                    return False
                if word == 'not':
                    if not results[-1] in ('and', 'or', '('):
                        results.append('and')
                results.append(word)
            else:
                if word not in ['(', 'not']:
                    log.error('Invalid beginning operator: %s', word)
                    return False
                results.append(word)
        elif target_info and target_info['engine']:
            if 'N' == target_info['engine']:
                decomposed = salt.utils.minions.nodegroup_comp(target_info['pattern'], nodegroups)
                if decomposed:
                    words = decomposed + words
                continue
            engine = ref.get(target_info['engine'])
            if not engine:
                log.error('Unrecognized target engine "%s" for target expression "%s"', target_info['engine'], word)
                return False
            engine_args = [target_info['pattern']]
            engine_kwargs = {'opts': opts, 'minion_id': minion_id}
            if target_info['delimiter']:
                engine_kwargs['delimiter'] = target_info['delimiter']
            results.append(str(__context__['matchers']['{}_match.match'.format(engine)](*engine_args, **engine_kwargs)))
        else:
            results.append(str(__context__['matchers']['glob_match.match'](word, opts, minion_id)))
    results = ' '.join(results)
    log.debug('compound_match %s ? "%s" => "%s"', minion_id, tgt, results)
    try:
        return eval(results)
    except Exception:
        log.error('Invalid compound target: %s for results: %s', tgt, results)
    return False