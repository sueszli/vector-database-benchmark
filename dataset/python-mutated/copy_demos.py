import argparse
import os
import pathlib
import shutil
import textwrap

def copy_all_demos(source_dir: str, dest_dir: str):
    if False:
        i = 10
        return i + 15
    demos_to_copy = ['audio_debugger', 'altair_plot', 'blocks_essay', 'blocks_group', 'blocks_js_methods', 'blocks_layout', 'blocks_multiple_event_triggers', 'blocks_update', 'calculator', 'cancel_events', 'chatbot_multimodal', 'chatinterface_streaming_echo', 'clear_components', 'code', 'fake_gan', 'fake_diffusion_with_gif', 'image_mod_default_image', 'image_segmentation', 'interface_random_slider', 'kitchen_sink', 'kitchen_sink_random', 'matrix_transpose', 'model3D', 'native_plots', 'reverse_audio', 'stt_or_tts', 'stream_audio', 'stream_frames', 'video_component', 'zip_files']
    for demo in demos_to_copy:
        shutil.copytree(os.path.join(source_dir, demo), os.path.join(dest_dir, demo), dirs_exist_ok=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy all demos to all_demos and update requirements')
    parser.add_argument('gradio_version', type=str, help='Gradio')
    parser.add_argument('gradio_client_version', type=str, help='Gradio Client Version')
    args = parser.parse_args()
    source_dir = pathlib.Path(pathlib.Path(__file__).parent, '..', 'demo')
    dest_dir = pathlib.Path(pathlib.Path(__file__).parent, '..', 'demo', 'all_demos', 'demos')
    copy_all_demos(source_dir, dest_dir)
    reqs_file_path = pathlib.Path(pathlib.Path(__file__).parent, '..', 'demo', 'all_demos', 'requirements.txt')
    requirements = f'\n    {args.gradio_client_version}\n    {args.gradio_version}\n    pypistats==1.1.0\n    plotly==5.10.0\n    opencv-python==4.6.0.66\n    transformers==4.21.1\n    torch==1.12.1\n    altair\n    vega_datasets\n    pydantic==2.1.1\n    pydantic_core==2.4.0\n    '
    open(reqs_file_path, 'w').write(textwrap.dedent(requirements))