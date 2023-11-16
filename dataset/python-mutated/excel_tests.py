from datetime import datetime, timezone
import pandas as pd
from superset.utils.excel import df_to_excel

def test_timezone_conversion() -> None:
    if False:
        while True:
            i = 10
    '\n    Test that columns with timezones are converted to a string.\n    '
    df = pd.DataFrame({'dt': [datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)]})
    contents = df_to_excel(df)
    assert pd.read_excel(contents)['dt'][0] == '2023-01-01 00:00:00+00:00'