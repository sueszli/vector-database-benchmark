"""
Topic: 读取修改某个XML文档
Desc : 
"""
from xml.etree.ElementTree import parse, Element

def rw_xml():
    if False:
        print('Hello World!')
    doc = parse('pred.xml')
    root = doc.getroot()
    root.remove(root.find('sri'))
    root.remove(root.find('cr'))
    root.getchildren().index(root.find('nm'))
    e = Element('spam')
    e.text = 'This is a test'
    root.insert(2, e)
    doc.write('newpred.xml', xml_declaration=True)
if __name__ == '__main__':
    rw_xml()