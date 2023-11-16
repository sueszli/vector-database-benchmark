from datetime import date
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, Datetime
from featuretools.primitives.base import TransformPrimitive

class Season(TransformPrimitive):
    """Determines the season of a given datetime.
        Returns winter, spring, summer, or fall.
        This only works for northern hemisphere.

    Description:
        Given a list of datetimes, return the season of each one
        (`winter`, `spring`, `summer`, or `fall`).

    Examples:
        >>> from datetime import datetime
        >>> times = [datetime(2019, 1, 1),
        ...          datetime(2019, 4, 15),
        ...          datetime(2019, 7, 20),
        ...          datetime(2019, 12, 30)]
        >>> season = Season()
        >>> season(times).tolist()
        ['winter', 'spring', 'summer', 'winter']
    """
    name = 'season'
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def get_function(self):
        if False:
            return 10

        def season(x):
            if False:
                return 10
            Y = 2000
            seasons = [('winter', (date(Y, 1, 1), date(Y, 3, 20))), ('spring', (date(Y, 3, 21), date(Y, 6, 20))), ('summer', (date(Y, 6, 21), date(Y, 9, 22))), ('fall', (date(Y, 9, 23), date(Y, 12, 20))), ('winter', (date(Y, 12, 21), date(Y, 12, 31)))]
            x = x.apply(lambda x: x.replace(year=2000))

            def get_season(dt):
                if False:
                    i = 10
                    return i + 15
                for (season, (start, end)) in seasons:
                    if not pd.isna(dt) and start <= dt.date() <= end:
                        return season
                return pd.NA
            new = x.apply(get_season).astype(dtype='string')
            return new
        return season