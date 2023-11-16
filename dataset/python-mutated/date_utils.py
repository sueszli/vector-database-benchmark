from toolz import partition_all

def compute_date_range_chunks(sessions, start_date, end_date, chunksize):
    if False:
        while True:
            i = 10
    'Compute the start and end dates to run a pipeline for.\n\n    Parameters\n    ----------\n    sessions : DatetimeIndex\n        The available dates.\n    start_date : pd.Timestamp\n        The first date in the pipeline.\n    end_date : pd.Timestamp\n        The last date in the pipeline.\n    chunksize : int or None\n        The size of the chunks to run. Setting this to None returns one chunk.\n\n    Returns\n    -------\n    ranges : iterable[(np.datetime64, np.datetime64)]\n        A sequence of start and end dates to run the pipeline for.\n    '
    if start_date not in sessions:
        raise KeyError('Start date %s is not found in calendar.' % (start_date.strftime('%Y-%m-%d'),))
    if end_date not in sessions:
        raise KeyError('End date %s is not found in calendar.' % (end_date.strftime('%Y-%m-%d'),))
    if end_date < start_date:
        raise ValueError('End date %s cannot precede start date %s.' % (end_date.strftime('%Y-%m-%d'), start_date.strftime('%Y-%m-%d')))
    if chunksize is None:
        return [(start_date, end_date)]
    (start_ix, end_ix) = sessions.slice_locs(start_date, end_date)
    return ((r[0], r[-1]) for r in partition_all(chunksize, sessions[start_ix:end_ix]))