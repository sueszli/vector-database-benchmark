import glob
import os
import shutil
import sys
'\nBuildHHP.py\n\nBuild HTML Help project file.\n'
sHHPFormat = '\n[OPTIONS]\nBinary TOC=Yes\nCompatibility=1.1 or later\nCompiled file=%(output)s.chm\nContents file=%(output)s.hhc\nDefault Window=Home\nDefault topic=%(target)s.HTML\nDisplay compile progress=Yes\nFull-text search=Yes\nIndex file=%(output)s.hhk\nLanguage=0x409 English (United States)\n\n[WINDOWS]\nHome="%(target)s","%(target)s.hhc","%(target)s.hhk","%(target)s.HTML","%(target)s.HTML",,,,,0x63520,,0x387e,,,,,,2,,0\n\n\n[FILES]\n%(output)s.HTML\n%(html_files)s\n\n[INFOTYPES]\n'

def handle_globs(lGlobs):
    if False:
        i = 10
        return i + 15
    assert lGlobs, 'you must pass some patterns!'
    lFiles = []
    for g in lGlobs:
        new = glob.glob(g)
        if len(new) == 0:
            print(f"The pattern '{g}' yielded no files!")
        lFiles = lFiles + new
    cFiles = len(lFiles)
    for i in range(cFiles):
        lFiles[i] = os.path.normpath(lFiles[i])
    i = 0
    while i < cFiles:
        if not os.path.isfile(lFiles[i]):
            del lFiles[i]
            cFiles = cFiles - 1
            continue
        i = i + 1
    sCommonPrefix = os.path.commonprefix(lFiles)
    if sCommonPrefix[-1] not in '\\/':
        sCommonPrefix = os.path.split(sCommonPrefix)[0]
        sCommonPrefix = os.path.normpath(sCommonPrefix) + '\\'
    assert os.path.isdir(sCommonPrefix) and sCommonPrefix[-1] == '\\', 'commonprefix splitting aint gunna work!'
    print('sCommonPrefix=', sCommonPrefix)
    lRelativeFiles = []
    for file in lFiles:
        lRelativeFiles.append(file[len(sCommonPrefix):])
    return (lRelativeFiles, lFiles)
import document_object

def main():
    if False:
        return 10
    doc = document_object.GetDocument()
    output = os.path.abspath(sys.argv[1])
    target = sys.argv[2]
    f = open(output + '.hhp', 'w')
    html_files = ''
    if len(sys.argv) > 2:
        output_dir = os.path.abspath(sys.argv[3])
        html_dir = os.path.abspath(os.path.join(output_dir, 'html'))
        if not os.path.isdir(html_dir):
            os.makedirs(html_dir)
        lGlobs = sys.argv[4:]
        (lDestFiles, lSrcFiles) = handle_globs(lGlobs)
        try:
            os.makedirs(html_dir)
        except:
            pass
        for i in range(len(lDestFiles)):
            file = lDestFiles[i]
            file = os.path.join(html_dir, file)
            try:
                os.makedirs(os.path.split(file)[0])
            except:
                pass
            shutil.copyfile(lSrcFiles[i], file)
        for file in lDestFiles:
            html_files = html_files + f'{html_dir}\\{file}\n'
    for cat in doc:
        html_files = html_files + f'{output_dir}\\{cat.id}.html\n'
        for suffix in '_overview _modules _objects _constants'.split():
            html_files = html_files + f'{output_dir}\\{cat.id}{suffix}.html\n'
    f.write(sHHPFormat % {'output': output, 'target': target, 'html_files': html_files})
    f.close()
if __name__ == '__main__':
    main()