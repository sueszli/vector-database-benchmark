from django import template
register = template.Library()

@register.filter(name='int2str')
def int2str(value):
    if False:
        i = 10
        return i + 15
    '\n    int 转换为 str\n    '
    return str(value)

@register.filter(name='res_splict')
def res_split(value):
    if False:
        for i in range(10):
            print('nop')
    '\n    将结果格式化换行\n    '
    res = []
    if isinstance(value, tuple):
        for v in value:
            if v is not None:
                data = v.replace('\n', '<br>')
                res.append(data)
        return res
    else:
        return value