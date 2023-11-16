import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def user():
    if False:
        print('Hello World!')
    a = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))[0:4]
    a.head()
    print(a[0].names)
    print(a[2, 0])
    print(a[2, 'sepal_len'])
    (a[0] + 2).show()
    (a[0] + a[1]).show()
    sum(a).show()
    print(a['sepal_len'].mean())
    print()
    print('Rows 50 through 77 in the `sepal_len` column')
    a[50:78, 'sepal_len'].show()
    print()
    a['sepal_len'].show()
    print(a[50:78, ['sepal_len', 'sepal_wid']].show())
    a.show()
    print('The column means: ')
    print(a.mean())
    print()
    try:
        print(a['Sepal_len'].dim)
    except Exception:
        pass
    b = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))[0:4]
    c = a + b
    d = c + c + sum(a)
    e = c + a + 1
    e.show()
    c.show()
    c = None
    print(1 + (a[0] + b[1]).mean())
    import collections
    c = h2o.H2OFrame(collections.OrderedDict({'A': [1, 2, 3], 'B': [4, 5, 6]}))
    c.show()
    c.describe()
    c.head()
    c[0].show()
    print(c[1, 0])
    c[0:2, 0].show()
    sliced = a[0:51, 0]
    sliced.show()
if __name__ == '__main__':
    pyunit_utils.standalone_test(user)
else:
    user()