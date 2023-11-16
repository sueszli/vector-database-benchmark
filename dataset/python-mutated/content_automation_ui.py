import gradio as gr
from gui.ui_tab_short_automation import ShortAutomationUI
from gui.ui_tab_video_automation import VideoAutomationUI
from gui.ui_tab_video_translation import VideoTranslationUI

class GradioContentAutomationUI:

    def __init__(self, shortGPTUI):
        if False:
            i = 10
            return i + 15
        self.shortGPTUI = shortGPTUI
        self.content_automation_ui = None

    def create_ui(self):
        if False:
            for i in range(10):
                print('nop')
        'Create Gradio interface'
        with gr.Tab('Content Automation') as self.content_automation_ui:
            gr.Markdown('# 🏆 Content Automation 🚀')
            gr.Markdown('## Choose your desired automation task.')
            choice = gr.Radio(['🎬 Automate the creation of shorts', '🎞️ Automate a video with stock assets', '🌐 Automate multilingual video dubbing'], label='Choose an option')
            video_automation_ui = VideoAutomationUI(self.shortGPTUI).create_ui()
            short_automation_ui = ShortAutomationUI(self.shortGPTUI).create_ui()
            video_translation_ui = VideoTranslationUI(self.shortGPTUI).create_ui()
            choice.change(lambda x: (gr.update(visible=x == choice.choices[1]), gr.update(visible=x == choice.choices[0]), gr.update(visible=x == choice.choices[2])), [choice], [video_automation_ui, short_automation_ui, video_translation_ui])
        return self.content_automation_ui