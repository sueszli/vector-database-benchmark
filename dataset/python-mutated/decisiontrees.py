import graphviz
import itertools
import random
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import OneHotEncoder
classes = {'supplies': ['low', 'med', 'high'], 'weather': ['raining', 'cloudy', 'sunny'], 'worked?': ['yes', 'no']}
data = [['low', 'sunny', 'yes'], ['high', 'sunny', 'yes'], ['med', 'cloudy', 'yes'], ['low', 'raining', 'yes'], ['low', 'cloudy', 'no'], ['high', 'sunny', 'no'], ['high', 'raining', 'no'], ['med', 'cloudy', 'yes'], ['low', 'raining', 'yes'], ['low', 'raining', 'no'], ['med', 'sunny', 'no'], ['high', 'sunny', 'yes']]
target = ['yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no']
categories = [classes['supplies'], classes['weather'], classes['worked?']]
encoder = OneHotEncoder(categories=categories)
x_data = encoder.fit_transform(data)
classifier = DecisionTreeClassifier()
tree = classifier.fit(x_data, target)
prediction_data = []
for _ in itertools.repeat(None, 5):
    prediction_data.append([random.choice(classes['supplies']), random.choice(classes['weather']), random.choice(classes['worked?'])])
prediction_results = tree.predict(encoder.transform(prediction_data))

def format_array(arr):
    if False:
        print('Hello World!')
    return ''.join(['| {:<10}'.format(item) for item in arr])

def print_table(data, results):
    if False:
        i = 10
        return i + 15
    line = 'day  ' + format_array(list(classes.keys()) + ['went shopping?'])
    print('-' * len(line))
    print(line)
    print('-' * len(line))
    for (day, row) in enumerate(data):
        print('{:<5}'.format(day + 1) + format_array(row + [results[day]]))
    print('')
feature_names = ['supplies-' + x for x in classes['supplies']] + ['weather-' + x for x in classes['weather']] + ['worked-' + x for x in classes['worked?']]
dot_data = export_graphviz(tree, filled=True, proportion=True, feature_names=feature_names)
graph = graphviz.Source(dot_data)
graph.render(filename='decision_tree', cleanup=True, view=True)
print('Training Data:')
print_table(data, target)
print('Predicted Random Results:')
print_table(prediction_data, prediction_results)