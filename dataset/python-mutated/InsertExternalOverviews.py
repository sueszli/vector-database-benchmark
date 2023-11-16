import os
import sys
'\nReplace: <!--index:extopics-->\nWith:    <LI><A HREF="<context>">Topic Name</A>\n         <LI><A HREF="<context2>">Topic Name2</A>\nNote: The replacement string must be on one line.\nUsage:\n      AdExtTopics.py htmlfile ext_overviewfile\n'

def processFile(input, out, extLinksHTML, extTopicHTML, importantHTML):
    if False:
        for i in range(10):
            print('nop')
    while 1:
        line = input.readline()
        if not line:
            break
        line = line.replace('<!--index:exlinks-->', extLinksHTML)
        line = line.replace('<!--index:extopics-->', extTopicHTML)
        line = line.replace('<!--index:eximportant-->', importantHTML)
        out.write(line + '\n')

def genHTML(doc):
    if False:
        i = 10
        return i + 15
    s = ''
    for cat in doc:
        s = s + f'<H3>{cat.label}</H3>\n'
        dict = {}
        for item in cat.overviewItems.items:
            dict[item.name] = item.href
        keys = list(dict.keys())
        keys.sort()
        for k in keys:
            s = s + f'<LI><A HREF="html/{dict[k]}">{k}</A>\n'
    return s

def genLinksHTML(links):
    if False:
        print('Hello World!')
    s = ''
    for link in links:
        s = s + f'<LI><A HREF="{link.href}">{link.name}</A>\n'
    return s
import document_object

def main():
    if False:
        print('Hello World!')
    if len(sys.argv) != 2:
        print('Invalid args')
        sys.exit(1)
    file = sys.argv[1]
    input = open(file, 'r')
    out = open(file + '.2', 'w')
    doc = document_object.GetDocument()
    linksHTML = genLinksHTML(doc.links)
    extTopicHTML = genHTML(doc)
    importantHTML = genLinksHTML(doc.important)
    processFile(input, out, linksHTML, extTopicHTML, importantHTML)
    input.close()
    out.close()
    sCmd = 'del "%s"' % file
    os.unlink(file)
    os.rename(file + '.2', file)
if __name__ == '__main__':
    main()