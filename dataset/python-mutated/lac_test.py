from LAC import LAC
'\n本文件测试LAC的分词,词性标注及重要性功能\n'
os.environ['PYTHONIOENCODING'] = 'UTF-8'

def fun_seg():
    if False:
        print('Hello World!')
    lac = LAC('models_general/seg_model', mode='seg')
    text = u'LAC是个优秀的分词工具'
    seg_result = lac.run(text)
    print(seg_result)
    texts = [u'LAC是个优秀的分词工具', u'百度是一家高科技公司']
    seg_result = lac.run(texts)
    print(seg_result)

def fun_add_word():
    if False:
        print('Hello World!')
    lac = LAC(model_path='models_general/seg_model', mode='seg')
    lac.add_word('红红 火火', sep=None)
    seg_result = lac.run('他这一生红红火火了一把')
    print(seg_result)

def run():
    if False:
        print('Hello World!')
    lac = LAC(model_path='models_general/lac_model', mode='lac')
    result = lac.run('百度是一家很好的公司')
    print(result)

def rank():
    if False:
        i = 10
        return i + 15
    lac = LAC(model_path='models_general/rank_model', mode='rank')
    result = lac.run('百度是一家很好的公司')
    print(result)