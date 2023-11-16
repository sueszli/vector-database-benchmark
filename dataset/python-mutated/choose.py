from copy import copy
from visidata import vd, options, VisiData, ListOfDictSheet, ENTER, CompleteKey, ReturnValue
vd.option('fancy_chooser', False, 'a nicer selection interface for aggregators and jointype')

@VisiData.api
def chooseOne(vd, choices, type=''):
    if False:
        return 10
    'Return one user-selected key from *choices*.'
    return vd.choose(choices, 1, type=type)

@VisiData.api
def choose(vd, choices, n=None, type=''):
    if False:
        return 10
    'Return a list of 1 to *n* "key" from elements of *choices* (see chooseMany).'
    ret = vd.chooseMany(choices, type=type) or vd.fail('no choice made')
    if n and len(ret) > n:
        vd.fail('can only choose %s' % n)
    return ret[0] if n == 1 else ret

class ChoiceSheet(ListOfDictSheet):
    rowtype = 'choices'
    precious = False

    def makeChoice(self, rows):
        if False:
            print('Hello World!')
        raise ReturnValue([r['key'] for r in rows])

@VisiData.api
def chooseFancy(vd, choices):
    if False:
        return 10
    vs = ChoiceSheet('choices', source=copy(choices))
    options.set('disp_splitwin_pct', -75, vs)
    vs.reload()
    vs.setKeys([vs.column('key')])
    vd.push(vs)
    chosen = vd.runresult()
    vd.remove(vs)
    return chosen

@VisiData.api
def chooseMany(vd, choices, type=''):
    if False:
        while True:
            i = 10
    'Return a list of 1 or more keys from *choices*, which is a list of\n    dicts. Each element dict must have a unique "key", which must be typed\n    directly by the user in non-fancy mode (therefore no spaces).  All other\n    items in the dicts are also shown in fancy chooser mode.  Use previous\n    choices from the replay input if available.  Add chosen keys\n    (space-separated) to the cmdlog as input for the current command.'
    if vd.cmdlog:
        v = vd.getLastArgs()
        if v is not None:
            vd.setLastArgs(v)
            return v.split()
    if options.fancy_chooser:
        chosen = vd.chooseFancy(choices)
    else:
        chosen = []
        choice_keys = [c['key'] for c in choices]
        prompt = 'choose any of %d options (Ctrl+X for menu)' % len(choice_keys)
        try:

            def throw_fancy(v, i):
                if False:
                    while True:
                        i = 10
                ret = vd.chooseFancy(choices)
                if ret:
                    raise ReturnValue(ret)
                return (v, i)
            chosenstr = vd.input(prompt + ': ', completer=CompleteKey(choice_keys), bindings={'^X': throw_fancy}, type=type)
            for c in chosenstr.split():
                if c in choice_keys:
                    chosen.append(c)
                else:
                    vd.warning('invalid choice "%s"' % c)
        except ReturnValue as e:
            chosen = e.args[0]
    if vd.cmdlog:
        vd.setLastArgs(' '.join(chosen))
    return chosen
ChoiceSheet.addCommand(ENTER, 'choose-rows', 'makeChoice([cursorRow])')
ChoiceSheet.addCommand('g' + ENTER, 'choose-rows-selected', 'makeChoice(onlySelectedRows)')