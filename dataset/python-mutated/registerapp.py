import shutil
import os
import stat
import platform
import subprocess

def registerapp(app):
    if False:
        print('Hello World!')
    if not [int(n) for n in platform.mac_ver()[0].split('.')] >= [10, 8]:
        return (None, 'Registering requires OS X version >= 10.8')
    app_path = None
    app_path = subprocess.check_output(['/usr/bin/mdfind', 'kMDItemCFBundleIdentifier == "ade.plexpy.osxnotify"']).strip()
    if app_path:
        return (app_path, 'App previously registered')
    app = app.strip()
    if not app:
        return (None, 'Path/Application not entered')
    if os.path.splitext(app)[1] == '.app':
        app_path = app
    else:
        app_path = app + '.app'
    if os.path.exists(app_path):
        return (None, 'App %s already exists, choose a different name' % app_path)
    try:
        os.mkdir(app_path)
        os.mkdir(app_path + '/Contents')
        os.mkdir(app_path + '/Contents/MacOS')
        os.mkdir(app_path + '/Contents/Resources')
        shutil.copy(os.path.join(os.path.dirname(__file__), 'appIcon.icns'), app_path + '/Contents/Resources/')
        version = '1.0.0'
        bundleName = 'OSXNotify'
        bundleIdentifier = 'ade.plexpy.osxnotify'
        f = open(app_path + '/Contents/Info.plist', 'w')
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0">\n<dict>\n    <key>CFBundleDevelopmentRegion</key>\n    <string>English</string>\n    <key>CFBundleExecutable</key>\n    <string>main.py</string>\n    <key>CFBundleGetInfoString</key>\n    <string>%s</string>\n    <key>CFBundleIconFile</key>\n    <string>appIcon.icns</string>\n    <key>CFBundleIdentifier</key>\n    <string>%s</string>\n    <key>CFBundleInfoDictionaryVersion</key>\n    <string>6.0</string>\n    <key>CFBundleName</key>\n    <string>%s</string>\n    <key>CFBundlePackageType</key>\n    <string>APPL</string>\n    <key>CFBundleShortVersionString</key>\n    <string>%s</string>\n    <key>CFBundleSignature</key>\n    <string>????</string>\n    <key>CFBundleVersion</key>\n    <string>%s</string>\n    <key>NSAppleScriptEnabled</key>\n    <string>YES</string>\n    <key>NSMainNibFile</key>\n    <string>MainMenu</string>\n    <key>NSPrincipalClass</key>\n    <string>NSApplication</string>\n</dict>\n</plist>\n' % (bundleName + ' ' + version, bundleIdentifier, bundleName, bundleName + ' ' + version, version))
        f.close()
        f = open(app_path + '/Contents/PkgInfo', 'w')
        f.write('APPL????')
        f.close()
        f = open(app_path + '/Contents/MacOS/main.py', 'w')
        f.write('#!/usr/bin/python\n\nobjc = None\n\ndef swizzle(cls, SEL, func):\n    old_IMP = cls.instanceMethodForSelector_(SEL)\n    def wrapper(self, *args, **kwargs):\n        return func(self, old_IMP, *args, **kwargs)\n    new_IMP = objc.selector(wrapper, selector=old_IMP.selector,\n        signature=old_IMP.signature)\n    objc.classAddMethod(cls, SEL, new_IMP)\n\ndef notify(title, subtitle=None, text=None, sound=True):\n    global objc\n    objc = __import__("objc")\n    swizzle(objc.lookUpClass(\'NSBundle\'),\n        b\'bundleIdentifier\',\n        swizzled_bundleIdentifier)\n    NSUserNotification = objc.lookUpClass(\'NSUserNotification\')\n    NSUserNotificationCenter = objc.lookUpClass(\'NSUserNotificationCenter\')\n    NSAutoreleasePool = objc.lookUpClass(\'NSAutoreleasePool\')\n    pool = NSAutoreleasePool.alloc().init()\n    notification = NSUserNotification.alloc().init()\n    notification.setTitle_(title)\n    notification.setSubtitle_(subtitle)\n    notification.setInformativeText_(text)\n    notification.setSoundName_("NSUserNotificationDefaultSoundName")\n    notification_center = NSUserNotificationCenter.defaultUserNotificationCenter()\n    notification_center.deliverNotification_(notification)\n    del pool\n\ndef swizzled_bundleIdentifier(self, original):\n    return \'ade.plexpy.osxnotify\'\n\nif __name__ == \'__main__\':\n    notify(\'Tautulli\', \'Test Subtitle\', \'Test Body\')\n')
        f.close()
        oldmode = os.stat(app_path + '/Contents/MacOS/main.py').st_mode
        os.chmod(app_path + '/Contents/MacOS/main.py', oldmode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return (app_path, 'App registered')
    except Exception as e:
        return (None, 'Error creating App %s. %s' % (app_path, e))