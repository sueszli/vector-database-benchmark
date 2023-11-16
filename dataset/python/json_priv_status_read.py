import os
import requests
import json
import time


GITHUB_TOKEN = os.getenv("HAWTWHEELZ_GITHUB_TOKEN")

link_to_ping = 'https://layer.bicyclesharing.net/map/v1/nyc/stations'
a = requests.get(link_to_ping).json()
dest = {}
for g in a['features']:
    dest[g['properties']['terminal']] = g['properties']

    
#dest_json = json.dump(dest)    

os.chdir('/tmp')
os.system(f'git clone https://{GITHUB_TOKEN}@github.com/pjlanger1/hotwheels.git')
os.system('mkdir -p /tmp/hotwheels/data/system-status-priv')

update_time = time.time()
with open(f'data{update_time}.json', 'w', encoding='utf-8') as f:
    json.dump(dest,f,ensure_ascii=False, indent=4)
    #df.to_json(f"/tmp/hotwheels/data/system-status-priv/{update_time}.json")
    # with open("/tmp/output.json") as output_file:
    #     print(output_file.read()

os.chdir('/tmp/hotwheels')
os.system('git add data/')
os.system(f'git commit -m "system status update - {update_time}"')
os.system('git push')
