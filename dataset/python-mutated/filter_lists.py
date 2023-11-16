def uninformative(title):
    if False:
        for i in range(10):
            print('nop')
    return title.lower().startswith('list_of_') or title.lower().startswith('lists_of_') or title.lower().startswith('index_of_.') or title.lower().startswith('outline_of_')

def preprocess(doc):
    if False:
        print('Hello World!')
    return None if uninformative(doc['id']) else doc