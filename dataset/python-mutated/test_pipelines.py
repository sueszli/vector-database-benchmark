import pytest
import transformers
import gradio as gr

def test_text_to_text_model_from_pipeline():
    if False:
        i = 10
        return i + 15
    pipe = transformers.pipeline(model='sshleifer/bart-tiny-random')
    io = gr.Interface.from_pipeline(pipe)
    output = io('My name is Sylvain and I work at Hugging Face in Brooklyn')
    assert isinstance(output, str)

@pytest.mark.flaky
def test_interface_in_blocks():
    if False:
        for i in range(10):
            print('nop')
    pipe1 = transformers.pipeline(model='sshleifer/bart-tiny-random')
    pipe2 = transformers.pipeline(model='sshleifer/bart-tiny-random')
    with gr.Blocks() as demo:
        with gr.Tab('Image Inference'):
            gr.Interface.from_pipeline(pipe1)
        with gr.Tab('Image Inference'):
            gr.Interface.from_pipeline(pipe2)
    demo.launch(prevent_thread_lock=True)
    demo.close()