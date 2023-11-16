from win32com.shell import shell, shellcon

def walk(folder, depth=2, indent=''):
    if False:
        return 10
    try:
        pidls = folder.EnumObjects(0, shellcon.SHCONTF_FOLDERS)
    except shell.error:
        return
    for pidl in pidls:
        dn = folder.GetDisplayNameOf(pidl, shellcon.SHGDN_NORMAL)
        print(indent, dn)
        if depth:
            try:
                child = folder.BindToObject(pidl, None, shell.IID_IShellFolder)
            except shell.error:
                pass
            else:
                walk(child, depth - 1, indent + ' ')
walk(shell.SHGetDesktopFolder())