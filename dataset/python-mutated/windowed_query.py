def windowed_query(s, q, column, windowsize):
    if False:
        return 10
    'Break a Query into chunks on a given column.'
    q = q.add_columns(column).order_by(column)
    last_id = None
    while True:
        subq = q
        if last_id is not None:
            subq = subq.filter(column > last_id)
        chunk = s.execute(subq.limit(windowsize)).all()
        if not chunk:
            break
        last_id = chunk[-1][-1]
        yield from chunk