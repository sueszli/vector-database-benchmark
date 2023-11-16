import json
from scripts.render_tools import blender_render as blender
with open('params.json', 'r') as params_file:
    params = json.load(params_file)

def run_blender_task():
    if False:
        while True:
            i = 10
    paths = params
    results_info = blender.render(params, paths)
    print(results_info)
run_blender_task()