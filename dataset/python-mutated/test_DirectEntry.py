from direct.gui.DirectEntry import DirectEntry

def test_entry_destroy():
    if False:
        print('Hello World!')
    entry = DirectEntry()
    entry.destroy()

def test_entry_get():
    if False:
        print('Hello World!')
    entry = DirectEntry()
    assert isinstance(entry.get(), str)

def test_entry_auto_capitalize():
    if False:
        print('Hello World!')
    entry = DirectEntry()
    entry.set('auto capitalize test')
    entry._autoCapitalize()
    assert entry.get() == 'Auto Capitalize Test'
    entry.set(u'àütò çapítalízè ţèsţ')
    assert entry.get() == u'àütò çapítalízè ţèsţ'
    entry._autoCapitalize()
    assert entry.get() == u'Àütò Çapítalízè Ţèsţ'