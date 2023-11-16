def drop_view_if_exists(cr, viewname):
    if False:
        while True:
            i = 10
    cr.execute('DROP view IF EXISTS %s CASCADE' % (viewname,))
    cr.commit()

def escape_psql(to_escape):
    if False:
        print('Hello World!')
    return to_escape.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')

def pg_varchar(size=0):
    if False:
        for i in range(10):
            print('nop')
    " Returns the VARCHAR declaration for the provided size:\n\n    * If no size (or an empty or negative size is provided) return an\n      'infinite' VARCHAR\n    * Otherwise return a VARCHAR(n)\n\n    :type int size: varchar size, optional\n    :rtype: str\n    "
    if size:
        if not isinstance(size, int):
            raise ValueError('VARCHAR parameter should be an int, got %s' % type(size))
        if size > 0:
            return 'VARCHAR(%d)' % size
    return 'VARCHAR'

def reverse_order(order):
    if False:
        i = 10
        return i + 15
    ' Reverse an ORDER BY clause '
    items = []
    for item in order.split(','):
        item = item.lower().split()
        direction = 'asc' if item[1:] == ['desc'] else 'desc'
        items.append('%s %s' % (item[0], direction))
    return ', '.join(items)