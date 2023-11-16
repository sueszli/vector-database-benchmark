import lux
from lux.vis.Vis import Vis
from lux.processor.Compiler import Compiler
from lux.utils import utils
from lux.interestingness.interestingness import interestingness

def generalize(ldf):
    if False:
        i = 10
        return i + 15
    '\n    Generates all possible visualizations when one attribute or filter from the current vis is removed.\n\n    Parameters\n    ----------\n    ldf : lux.core.frame\n            LuxDataFrame with underspecified intent.\n\n    Returns\n    -------\n    recommendations : Dict[str,obj]\n            object with a collection of visualizations that result from the Generalize action.\n    '
    output = []
    excluded_columns = []
    attributes = list(filter(lambda x: x.value == '' and x.attribute != 'Record', ldf._intent))
    filters = utils.get_filter_specs(ldf._intent)
    fltr_str = [fltr.attribute + fltr.filter_op + str(fltr.value) for fltr in filters]
    attr_str = [str(clause.attribute) for clause in attributes]
    intended_attrs = f"""<p class="highlight-intent">{', '.join(attr_str + fltr_str)}</p>"""
    recommendation = {'action': 'Generalize', 'description': f'Remove an attribute or filter from {intended_attrs}.', 'long_description': f'Remove one aspect of the Current Vis. We can either remove an attribute or filter from {intended_attrs}.'}
    if len(attributes) < 1 or len(attributes) > 4:
        recommendation['collection'] = []
        return recommendation
    if len(attributes) > 1:
        for clause in attributes:
            columns = clause.attribute
            if type(columns) == list:
                for column in columns:
                    if column not in excluded_columns:
                        temp_vis = Vis(ldf.copy_intent(), score=1)
                        temp_vis.remove_column_from_spec(column, remove_first=True)
                        excluded_columns.append(column)
                        output.append(temp_vis)
            elif columns not in excluded_columns:
                temp_vis = Vis(ldf.copy_intent(), score=1)
                temp_vis.remove_column_from_spec(columns, remove_first=True)
                excluded_columns.append(columns)
            output.append(temp_vis)
    for clause in filters:
        temp_vis = Vis(ldf.current_vis[0]._inferred_intent.copy(), source=ldf, title='Overall', score=0)
        temp_vis.remove_filter_from_spec(clause.value)
        output.append(temp_vis)
    vlist = lux.vis.VisList.VisList(output, source=ldf)
    vlist.remove_duplicates()
    vlist.sort(remove_invalid=True)
    vlist._collection = list(filter(lambda x: x.score != -1, vlist._collection))
    recommendation['collection'] = vlist
    return recommendation