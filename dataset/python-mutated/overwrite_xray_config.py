import gradio as gr
import random
import json
import os

def fake_func():
    if False:
        while True:
            i = 10
    return 'Hello There'

def xray_model(diseases, img):
    if False:
        for i in range(10):
            print('nop')
    return {disease: random.random() for disease in diseases}

def ct_model(diseases, img):
    if False:
        for i in range(10):
            print('nop')
    return {disease: 0.1 for disease in diseases}
with gr.Blocks() as demo:
    gr.Markdown('\n    # Detect Disease From Scan\n    With this model you can lorem ipsum\n    - ipsum 1\n    - ipsum 2\n    ')
    disease = gr.CheckboxGroup(choices=[['Covid', 'Covid'], ['Malaria', 'Malaria'], ['Lung Cancer', 'Lung Cancer']], label='Disease to Scan For')
    with gr.Tabs():
        with gr.TabItem('X-ray'):
            with gr.Row():
                xray_scan = gr.Image()
                xray_results = gr.JSON()
            xray_run = gr.Button('Run')
            xray_run.click(xray_model, inputs=[disease, xray_scan], outputs=xray_results)
        with gr.TabItem('CT Scan'):
            with gr.Row():
                ct_scan = gr.Image()
                ct_results = gr.JSON()
            ct_run = gr.Button('Run')
            ct_run.click(ct_model, inputs=[disease, ct_scan], outputs=ct_results)
    textbox = gr.Textbox()
gr.context.Context.id = 100
with gr.Blocks() as demo2:
    gr.Markdown('\n    # Detect Disease From Scan\n    With this model you can lorem ipsum\n    - ipsum 1\n    - ipsum 2\n    ')
    disease = gr.CheckboxGroup(choices=[['Covid', 'Covid'], ['Malaria', 'Malaria'], ['Lung Cancer', 'Lung Cancer']], label='Disease to Scan For')
    with gr.Tabs():
        with gr.TabItem('X-ray'):
            with gr.Row():
                xray_scan = gr.Image()
                xray_results = gr.JSON()
            xray_run = gr.Button('Run')
            xray_run.click(xray_model, inputs=[disease, xray_scan], outputs=xray_results)
        with gr.TabItem('CT Scan'):
            with gr.Row():
                ct_scan = gr.Image()
                ct_results = gr.JSON()
            ct_run = gr.Button('Run')
            ct_run.click(ct_model, inputs=[disease, ct_scan], outputs=ct_results)
    textbox = gr.Textbox()
with gr.Blocks() as demo3:
    demo.render()
    t = gr.Textbox()
    demo3.load(fake_func, [], [t])
config = demo.get_config_file()
config2 = demo2.get_config_file()
config3 = demo3.get_config_file()
json_file_path = '../test/test_files/xray_config.json'
json_file_path2 = '../test/test_files/xray_config_diff_ids.json'
json_file_path3 = '../test/test_files/xray_config_wrong.json'
for (c, j) in zip([config, config2, config3], [json_file_path, json_file_path2, json_file_path3]):
    assert os.path.exists(j), f'{j} does not exist'
    with open(j, 'w') as fp:
        json.dump(c, fp, indent=2)