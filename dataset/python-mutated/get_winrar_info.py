"""Automate WinRAR evaluation copy

We hit a few dialogs and save XML dump and
screenshot from each dialog.

Specify a language at the command line:
    0 Czech
    1 German
    2 French

More then likely you will need to modify the apppath
entry in the 't' dictionary to where you have
extracted the WinRAR executables.
"""
__revision__ = '$Revision$'
import sys
from pywinauto.application import Application
import time
folders = ['wrar351cz', 'wrar351d', 'wrar351fr']
t = {'apppath': ('c:\\.temp\\wrar351fr\\winrar.exe', 'c:\\.temp\\wrar351d\\winrar.exe', 'c:\\.temp\\wrar351cz\\winrar.exe'), 'Buy Licence': ('Acheter une licence pur winRAR', 'Bittekaufensieeine', 'Zakuptesiprosmlicenci WinRARu'), 'Close': ('Fermer', 'Schleissen', 'Zavrit'), 'Options->Configure': ('Options->Configuration', 'Optionen->Einstellungen', 'Moznosti->Nastaveni'), 'Configure': ('Configuration', 'Einstellungen', 'Nastaveni'), 'Buttons': ('Boutons', 'Schaltflachen', 'Vybrattlacitka'), 'PeronnaliseToolbars': ('Peronnalisation de la barre doutils', 'Werkzeugleisteanpassen', 'Vybertlaciteknastrojovelisty'), 'CreateDefaultProfile': (u'Creerleprofilpard�fault', 'Standardfestlegen', 'Vytvoritimplicitni'), 'ConfigureDefaultOptions': ('Configurer les options de compre...', 'Standardkomprimierungs', 'Zmenaimplicitnichnast...'), 'ContextMenus': ('Menus contextuels', 'Optionenimkontextmenu', 'Polozkykontextovehamenu'), 'contextMenuDlg': ('Rubriques des menus contextuels', 'OptionenindenKontextmenus', 'Polozkykontextovehamenu'), 'File->Exit': ('Fichier->Quitter', 'Datei->Beenden', 'Soubor->Konec')}

def get_winrar_dlgs(rar_dlg, app, lang):
    if False:
        return 10
    rar_dlg.menu_select(t['Options->Configure'][lang])
    optionsdlg = app[t['Configure'][lang]]
    optionsdlg.write_to_xml('Options_%d.xml' % lang)
    optionsdlg.capture_as_image().save('Options_%d.png' % lang)
    optionsdlg[t['Buttons'][lang]].click()
    contextMenuDlg = app[t['PeronnaliseToolbars'][lang]]
    contextMenuDlg.write_to_xml('PersonaliseToolbars_%d.xml' % lang)
    contextMenuDlg.capture_as_image().save('PersonaliseToolbars_%d.png' % lang)
    contextMenuDlg.OK.click()
    optionsdlg.TabCtrl.select(1)
    optionsdlg[t['CreateDefaultProfile'][lang]].click()
    defaultOptionsDlg = app[t['ConfigureDefaultOptions'][lang]]
    defaultOptionsDlg.write_to_xml('DefaultOptions_%d.xml' % lang)
    defaultOptionsDlg.capture_as_image().save('DefaultOptions_%d.png' % lang)
    defaultOptionsDlg.OK.click()
    optionsdlg.TabCtrl.select(6)
    optionsdlg[t['ContextMenus'][lang]].click()
    anotherMenuDlg = app[t['contextMenuDlg'][lang]]
    anotherMenuDlg.write_to_xml('2ndMenuDlg_%d.xml' % lang)
    anotherMenuDlg.capture_as_image().save('2ndMenuDlg_%d.png' % lang)
    anotherMenuDlg.OK.click()
    optionsdlg.OK.click()
langs = [int(arg) for arg in sys.argv[1:]]
for lang in langs:
    app = Application().start(t['apppath'][lang])
    time.sleep(2)
    licence_dlg = app[t['Buy Licence'][lang]]
    licence_dlg[t['Close'][lang]].click()
    rar_dlg = app.window(title_re='.* - WinRAR.*')
    get_winrar_dlgs(rar_dlg, app, lang)
    time.sleep(0.5)
    rar_dlg.menu_select(t['File->Exit'][lang])