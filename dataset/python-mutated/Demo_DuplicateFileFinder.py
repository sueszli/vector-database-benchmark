import PySimpleGUI as sg
import hashlib
import os
'\n    Find dups with PySimpleGUI\n'

def FindDuplicatesFilesInFolder(path):
    if False:
        for i in range(10):
            print('nop')
    shatab = []
    (total, small_count, dup_count, error_count) = [0] * 4
    pngdir = path
    if not os.path.exists(path):
        sg.popup('Duplicate Finder', "** Folder doesn't exist***", path)
        return
    pngfiles = os.listdir(pngdir)
    total_files = len(pngfiles)
    for (idx, f) in enumerate(pngfiles):
        if not sg.one_line_progress_meter('Counting Duplicates', idx + 1, total_files, 'Counting Duplicate Files'):
            break
        total += 1
        fname = os.path.join(pngdir, f)
        if os.path.isdir(fname):
            continue
        x = open(fname, 'rb').read()
        m = hashlib.sha256()
        m.update(x)
        f_sha = m.digest()
        if f_sha in shatab:
            dup_count += 1
            continue
        shatab.append(f_sha)
    msg = '{} Files processed\n {} Duplicates found'.format(total_files, dup_count)
    sg.popup('Duplicate Finder Ended', msg)
if __name__ == '__main__':
    source_folder = None
    source_folder = sg.popup_get_folder('Duplicate Finder - Count number of duplicate files', 'Enter path to folder you wish to find duplicates in')
    if source_folder is not None:
        FindDuplicatesFilesInFolder(source_folder)
    else:
        sg.popup_cancel('Cancelling', '*** Cancelling ***')
    exit(0)