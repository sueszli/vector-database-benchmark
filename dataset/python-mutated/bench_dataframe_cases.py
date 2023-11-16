from utils import benchmark_with_object

@benchmark_with_object(cls='dataframe', dtype='int', nulls=False)
def where_case_1(dataframe):
    if False:
        while True:
            i = 10
    return (dataframe, dataframe % 2 == 0, 0)

@benchmark_with_object(cls='dataframe', dtype='int', nulls=False)
def where_case_2(dataframe):
    if False:
        print('Hello World!')
    cond = dataframe[dataframe.columns[0]] % 2 == 0
    return (dataframe, cond, 0)