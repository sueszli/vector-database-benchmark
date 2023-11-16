x = range(10)
list(x)[0]
tuple(x)[0]
list((i for i in x))[0]
[i for i in x][0]
list(x)
list(x)[1]
list(x)[-1]
list(x)[1:]
list(x)[:3:2]
list(x)[::2]
list(x)[:]
[i for i in x]
[i for i in x][1]
[i for i in x][-1]
[i for i in x][:1]
[i for i in x][:1:1]
[i for i in x][:1:2]
[i for i in x][1:]
[i for i in x][:3:2]
[i for i in x][::2]
[i for i in x][:]
[i + 1 for i in x][0]
[i for i in x if i > 5][0]
[(i, i + 1) for i in x][0]
y = range(10)
[i + j for i in x for j in y][0]
list(range(10))[0]
list(x.y)[0]
list(x['y'])[0]
[*range(10)][0]
[*x['y']][0]
[*x.y][0]
[*x.y][0]
[*x.y][0]
revision_heads_map_ast = [a for a in revision_heads_map_ast_obj.body if isinstance(a, ast.Assign) and a.targets[0].id == 'REVISION_HEADS_MAP'][0]
list(zip(x, y))[0]
[*zip(x, y)][0]

def test():
    if False:
        print('Hello World!')
    zip = list
    list(zip(x, y))[0]