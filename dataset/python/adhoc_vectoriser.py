import json


def vectorise(list):
    """
    convert list of strings into sparse (dictionary) vectors using existing terms
    :param list: list of strings to be vectorised
    :return: sparse vector in dictionary form
    """
    with open('./dictionaries/term_to_id_dictionary.json') as fp:
        reference = json.load(fp)

    vector = {}
    for term in list:
        term = term.lower()
        try:
            vector[reference[term]] += 1
        except KeyError:
            try:
                vector[reference[term]] = 1
            except:
                pass

    return vector
