from django import template
register = template.Library()

@register.simple_tag
def parse_commenttree(commentlist, comment):
    if False:
        i = 10
        return i + 15
    '获得当前评论子评论的列表\n        用法: {% parse_commenttree article_comments comment as childcomments %}\n    '
    datas = []

    def parse(c):
        if False:
            for i in range(10):
                print('nop')
        childs = commentlist.filter(parent_comment=c, is_enable=True)
        for child in childs:
            datas.append(child)
            parse(child)
    parse(comment)
    return datas

@register.inclusion_tag('comments/tags/comment_item.html')
def show_comment_item(comment, ischild):
    if False:
        for i in range(10):
            print('nop')
    '评论'
    depth = 1 if ischild else 2
    return {'comment_item': comment, 'depth': depth}