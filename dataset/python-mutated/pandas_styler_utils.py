from typing import TYPE_CHECKING, Any, List, Mapping, TypeVar
import pandas as pd
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
if TYPE_CHECKING:
    from pandas.io.formats.style import Styler

def marshall_styler(proto: ArrowProto, styler: 'Styler', default_uuid: str) -> None:
    if False:
        print('Hello World!')
    'Marshall pandas.Styler into an Arrow proto.\n\n    Parameters\n    ----------\n    proto : proto.Arrow\n        Output. The protobuf for Streamlit Arrow proto.\n\n    styler : pandas.Styler\n        Helps style a DataFrame or Series according to the data with HTML and CSS.\n\n    default_uuid : str\n        If pandas.Styler uuid is not provided, this value will be used.\n\n    '
    styler_data_df: pd.DataFrame = styler.data
    if styler_data_df.size > int(pd.options.styler.render.max_elements):
        raise StreamlitAPIException(f'The dataframe has `{styler_data_df.size}` cells, but the maximum number of cells allowed to be rendered by Pandas Styler is configured to `{pd.options.styler.render.max_elements}`. To allow more cells to be styled, you can change the `"styler.render.max_elements"` config. For example: `pd.set_option("styler.render.max_elements", {styler_data_df.size})`')
    _marshall_uuid(proto, styler, default_uuid)
    styler._compute()
    pandas_styles = styler._translate(False, False)
    _marshall_caption(proto, styler)
    _marshall_styles(proto, styler, pandas_styles)
    _marshall_display_values(proto, styler_data_df, pandas_styles)

def _marshall_uuid(proto: ArrowProto, styler: 'Styler', default_uuid: str) -> None:
    if False:
        i = 10
        return i + 15
    'Marshall pandas.Styler uuid into an Arrow proto.\n\n    Parameters\n    ----------\n    proto : proto.Arrow\n        Output. The protobuf for Streamlit Arrow proto.\n\n    styler : pandas.Styler\n        Helps style a DataFrame or Series according to the data with HTML and CSS.\n\n    default_uuid : str\n        If pandas.Styler uuid is not provided, this value will be used.\n\n    '
    if styler.uuid is None:
        styler.set_uuid(default_uuid)
    proto.styler.uuid = str(styler.uuid)

def _marshall_caption(proto: ArrowProto, styler: 'Styler') -> None:
    if False:
        return 10
    'Marshall pandas.Styler caption into an Arrow proto.\n\n    Parameters\n    ----------\n    proto : proto.Arrow\n        Output. The protobuf for Streamlit Arrow proto.\n\n    styler : pandas.Styler\n        Helps style a DataFrame or Series according to the data with HTML and CSS.\n\n    '
    if styler.caption is not None:
        proto.styler.caption = styler.caption

def _marshall_styles(proto: ArrowProto, styler: 'Styler', styles: Mapping[str, Any]) -> None:
    if False:
        print('Hello World!')
    'Marshall pandas.Styler styles into an Arrow proto.\n\n    Parameters\n    ----------\n    proto : proto.Arrow\n        Output. The protobuf for Streamlit Arrow proto.\n\n    styler : pandas.Styler\n        Helps style a DataFrame or Series according to the data with HTML and CSS.\n\n    styles : dict\n        pandas.Styler translated styles.\n\n    '
    css_rules = []
    if 'table_styles' in styles:
        table_styles = styles['table_styles']
        table_styles = _trim_pandas_styles(table_styles)
        for style in table_styles:
            rule = _pandas_style_to_css('table_styles', style, styler.uuid, separator=' ')
            css_rules.append(rule)
    if 'cellstyle' in styles:
        cellstyle = styles['cellstyle']
        cellstyle = _trim_pandas_styles(cellstyle)
        for style in cellstyle:
            rule = _pandas_style_to_css('cell_style', style, styler.uuid)
            css_rules.append(rule)
    if len(css_rules) > 0:
        proto.styler.styles = '\n'.join(css_rules)
M = TypeVar('M', bound=Mapping[str, Any])

def _trim_pandas_styles(styles: List[M]) -> List[M]:
    if False:
        i = 10
        return i + 15
    "Filter out empty styles.\n\n    Every cell will have a class, but the list of props\n    may just be [['', '']].\n\n    Parameters\n    ----------\n    styles : list\n        pandas.Styler translated styles.\n\n    "
    return [x for x in styles if any((any(y) for y in x['props']))]

def _pandas_style_to_css(style_type: str, style: Mapping[str, Any], uuid: str, separator: str='') -> str:
    if False:
        while True:
            i = 10
    'Convert pandas.Styler translated style to CSS.\n\n    Parameters\n    ----------\n    style_type : str\n        Either "table_styles" or "cell_style".\n\n    style : dict\n        pandas.Styler translated style.\n\n    uuid : str\n        pandas.Styler uuid.\n\n    separator : str\n        A string separator used between table and cell selectors.\n\n    '
    declarations = []
    for (css_property, css_value) in style['props']:
        declaration = css_property.strip() + ': ' + css_value.strip()
        declarations.append(declaration)
    table_selector = f'#T_{uuid}'
    if style_type == 'table_styles':
        cell_selectors = [style['selector']]
    else:
        cell_selectors = style['selectors']
    selectors = []
    for cell_selector in cell_selectors:
        selectors.append(table_selector + separator + cell_selector)
    selector = ', '.join(selectors)
    declaration_block = '; '.join(declarations)
    rule_set = selector + ' { ' + declaration_block + ' }'
    return rule_set

def _marshall_display_values(proto: ArrowProto, df: pd.DataFrame, styles: Mapping[str, Any]) -> None:
    if False:
        return 10
    'Marshall pandas.Styler display values into an Arrow proto.\n\n    Parameters\n    ----------\n    proto : proto.Arrow\n        Output. The protobuf for Streamlit Arrow proto.\n\n    df : pandas.DataFrame\n        A dataframe with original values.\n\n    styles : dict\n        pandas.Styler translated styles.\n\n    '
    new_df = _use_display_values(df, styles)
    proto.styler.display_values = type_util.data_frame_to_bytes(new_df)

def _use_display_values(df: pd.DataFrame, styles: Mapping[str, Any]) -> pd.DataFrame:
    if False:
        print('Hello World!')
    'Create a new pandas.DataFrame where display values are used instead of original ones.\n\n    Parameters\n    ----------\n    df : pandas.DataFrame\n        A dataframe with original values.\n\n    styles : dict\n        pandas.Styler translated styles.\n\n    '
    import re
    new_df = df.astype(str)
    cell_selector_regex = re.compile('row(\\d+)_col(\\d+)')
    if 'body' in styles:
        rows = styles['body']
        for row in rows:
            for cell in row:
                if 'id' in cell:
                    if (match := cell_selector_regex.match(cell['id'])):
                        (r, c) = map(int, match.groups())
                        new_df.iat[r, c] = str(cell['display_value'])
    return new_df