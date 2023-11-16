import json
import json
import json
import copy

def write_json(file_path, data):
    if False:
        for i in range(10):
            print('nop')
    with open(file_path, 'w') as f:
        json.dump(data, f)

def read_json(file_path):
    if False:
        for i in range(10):
            print('nop')
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(data)
        return data
file_path = '/Users/zhangyujuan/graduation/final.json'
data = read_json(file_path)
new = copy.deepcopy(data)
for i in data:
    if data[i]['category'] == 'Face Serums':
        new[i]['class'] = 'Serum'
    elif data[i]['category'] == 'Moisturizers' or data[i]['category'] == 'Moisturizer & Treatments':
        new[i]['class'] = 'Cream'
    elif data[i]['category'] == 'Eye Creams & Treatments' or data[i]['category'] == 'Eye Cream':
        new[i]['class'] = 'EyeCream'
    elif data[i]['category'] == 'Face Sunscreen' or data[i]['category'] == 'Sunscreen':
        new[i]['class'] = 'Sunscreen'
    elif data[i]['category'] == 'Blemish & Acne Treatments':
        new[i]['class'] = 'Acne'
    elif data[i]['category'] == 'Face Oils':
        new[i]['class'] = 'Oils'
    elif data[i]['category'] == 'Face Wash & Cleansers':
        new[i]['class'] = 'Cleanser'
    elif data[i]['category'] == 'Lotions & Oils':
        del new[i]
    elif data[i]['category'] == 'Night Creams':
        new[i]['class'] = 'NightCream'
    elif data[i]['category'] == 'Toners' or data[i]['category'] == 'Mists & Essences':
        new[i]['class'] = 'Toner'
    else:
        pass
print(new)
tt = 'd.json'
write_json(tt, new)