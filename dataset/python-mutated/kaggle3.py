import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import modin.pandas as pd
data = pd.read_csv('pokemon.csv')
data.info()
data.corr()
(f, ax) = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt='.1f', ax=ax)
data.head(10)
data.columns
data.Speed.plot(kind='line', color='g', label='Speed', linewidth=1, alpha=0.5, grid=True, linestyle=':')
data.Defense.plot(color='r', label='Defense', linewidth=1, alpha=0.5, grid=True, linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color='red')
plt.xlabel('Attack')
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')
data.Speed.plot(kind='hist', bins=50, figsize=(12, 12))
data.Speed.plot(kind='hist', bins=50)
dictionary = {'spain': 'madrid', 'usa': 'vegas'}
print(dictionary.keys())
print(dictionary.values())
dictionary['spain'] = 'barcelona'
print(dictionary)
dictionary['france'] = 'paris'
print(dictionary)
del dictionary['spain']
print(dictionary)
print('france' in dictionary)
dictionary.clear()
print(dictionary)
print(dictionary)
data = pd.read_csv('pokemon.csv')
series = data['Defense']
print(type(series))
data_frame = data[['Defense']]
print(type(data_frame))
print(3 > 2)
print(3 != 2)
print(True and False)
print(True or False)
x = data['Defense'] > 200
data[x]
data[np.logical_and(data['Defense'] > 200, data['Attack'] > 100)]
data[(data['Defense'] > 200) & (data['Attack'] > 100)]
i = 0
while i != 5:
    print('i is: ', i)
    i += 1
print(i, ' is equal to 5')
lis = [1, 2, 3, 4, 5]
for i in lis:
    print('i is: ', i)
print('')
for (index, value) in enumerate(lis):
    print(index, ' : ', value)
print('')
dictionary = {'spain': 'madrid', 'france': 'paris'}
for (key, value) in dictionary.items():
    print(key, ' : ', value)
print('')
for (index, value) in data[['Attack']][0:1].iterrows():
    print(index, ' : ', value)

def tuble_ex():
    if False:
        for i in range(10):
            print('nop')
    'return defined t tuble'
    t = (1, 2, 3)
    return t
(a, b, c) = tuble_ex()
print(a, b, c)
x = 2

def f():
    if False:
        i = 10
        return i + 15
    x = 3
    return x
print(x)
print(f())
x = 5

def f():
    if False:
        for i in range(10):
            print('nop')
    y = 2 * x
    return y
print(f())
import builtins
dir(builtins)

def square():
    if False:
        while True:
            i = 10
    'return square of value'

    def add():
        if False:
            return 10
        'add two local variable'
        x = 2
        y = 3
        z = x + y
        return z
    return add() ** 2
print(square())

def f(a, b=1, c=2):
    if False:
        print('Hello World!')
    y = a + b + c
    return y
print(f(5))
print(f(5, 4, 3))

def f(*args):
    if False:
        while True:
            i = 10
    for i in args:
        print(i)
f(1)
print('')
f(1, 2, 3, 4)

def f(**kwargs):
    if False:
        print('Hello World!')
    'print key and value of dictionary'
    for (key, value) in kwargs.items():
        print(key, ' ', value)
f(country='spain', capital='madrid', population=123456)
number_list = [1, 2, 3]
y = map(lambda x: x ** 2, number_list)
print(list(y))
name = 'ronaldo'
it = iter(name)
print(next(it))
print(*it)
list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
z = zip(list1, list2)
print(z)
z_list = list(z)
print(z_list)
un_zip = zip(*z_list)
(un_list1, un_list2) = list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list2))
num1 = [1, 2, 3]
num2 = [i + 1 for i in num1]
print(num2)
num1 = [5, 10, 15]
num2 = [i ** 2 if i == 10 else i - 5 if i < 7 else i + 5 for i in num1]
print(num2)
threshold = sum(data.Speed) / len(data.Speed)
data['speed_level'] = ['high' if i > threshold else 'low' for i in data.Speed]
data.loc[:10, ['speed_level', 'Speed']]
data = pd.read_csv('pokemon.csv')
data.head()
data.tail()
data.columns
data.shape
data.info()
print(data['Type 1'].value_counts(dropna=False))
data.describe()
data.boxplot(column='Attack', by='Legendary')
data_new = data.head()
data_new
melted = pd.melt(frame=data_new, id_vars='Name', value_vars=['Attack', 'Defense'])
melted
melted.pivot(index='Name', columns='variable', values='value')
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1, data2], axis=0, ignore_index=True)
conc_data_row
data1 = data['Attack'].head()
data2 = data['Defense'].head()
conc_data_col = pd.concat([data1, data2], axis=1)
conc_data_col
data.dtypes
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')
data.dtypes
data.info()
data['Type 2'].value_counts(dropna=False)
data1 = data
data1['Type 2'].dropna(inplace=True)
assert 1 == 1
assert data['Type 2'].notnull().all()
data['Type 2'].fillna('empty', inplace=True)
assert data['Type 2'].notnull().all()
country = ['Spain', 'France']
population = ['11', '12']
list_label = ['country', 'population']
list_col = [country, population]
zipped = list(zip(list_label, list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
df['capital'] = ['madrid', 'paris']
df
df['income'] = 0
df
data1 = data.loc[:, ['Attack', 'Defense', 'Speed']]
data1.plot()
data1.plot(subplots=True)
plt.show()
data1.plot(kind='scatter', x='Attack', y='Defense')
plt.show()
data1.plot(kind='hist', y='Defense', bins=50, range=(0, 250), normed=True)
(fig, axes) = plt.subplots(nrows=2, ncols=1)
data1.plot(kind='hist', y='Defense', bins=50, range=(0, 250), normed=True, ax=axes[0])
data1.plot(kind='hist', y='Defense', bins=50, range=(0, 250), normed=True, ax=axes[1], cumulative=True)
plt.savefig('graph.png')
plt
data.describe()
time_list = ['1992-03-08', '1992-04-12']
print(type(time_list[1]))
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
import warnings
warnings.filterwarnings('ignore')
data2 = data.head()
date_list = ['1992-01-10', '1992-02-10', '1992-03-10', '1993-03-15', '1993-03-16']
datetime_object = pd.to_datetime(date_list)
data2['date'] = datetime_object
data2 = data2.set_index('date')
data2
print(data2.loc['1993-03-16'])
print(data2.loc['1992-03-10':'1993-03-16'])
data2.resample('A').mean()
data2.resample('M').mean()
data2.resample('M').first().interpolate('linear')
data2.resample('M').mean().interpolate('linear')
data = pd.read_csv('pokemon.csv')
data = data.set_index('#')
data.head()
data['HP'][1]
data.HP[1]
data.loc[1, ['HP']]
data[['HP', 'Attack']]
print(type(data['HP']))
print(type(data[['HP']]))
data.loc[1:10, 'HP':'Defense']
data.loc[10:1:-1, 'HP':'Defense']
data.loc[1:10, 'Speed':]
boolean = data.HP > 200
data[boolean]
first_filter = data.HP > 150
second_filter = data.Speed > 35
data[first_filter & second_filter]
data.HP[data.Speed < 15]

def div(n):
    if False:
        print('Hello World!')
    return n / 2
data.HP.apply(div)
data.HP.apply(lambda n: n / 2)
data['total_power'] = data.Attack + data.Defense
data.head()
print(data.index.name)
data.index.name = 'index_name'
data.head()
data.head()
data3 = data.copy()
data3.index = range(100, 100 + len(data3.index), 1)
data3.head()
data = pd.read_csv('pokemon.csv')
data.head()
data1 = data.set_index(['Type 1', 'Type 2'])
data1.head(100)
dic = {'treatment': ['A', 'A', 'B', 'B'], 'gender': ['F', 'M', 'F', 'M'], 'response': [10, 45, 5, 9], 'age': [15, 4, 72, 65]}
df = pd.DataFrame(dic)
df
df.pivot(index='treatment', columns='gender', values='response')
df1 = df.set_index(['treatment', 'gender'])
df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0, 1)
df2
df
pd.melt(df, id_vars='treatment', value_vars=['age', 'response'])
df
df.groupby('treatment').mean()
df.groupby('treatment').age.max()
df.groupby('treatment')[['age', 'response']].min()
df.info()