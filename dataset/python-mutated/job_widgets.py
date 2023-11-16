"""A module of widgets for job tracking"""
import ipywidgets as widgets
from IPython.display import display, Javascript

def make_clear_button(watcher):
    if False:
        for i in range(10):
            print('nop')
    'Makes the clear button\n\n    Args:\n        watcher (widget): The watcher widget instance.\n\n    Returns:\n        widget: The clear button widget.\n    '
    clear = widgets.Button(description='Clear', button_style='primary', layout=widgets.Layout(width='70px', grid_area='right', padding='0px 0px 0px 0px'))

    def on_clear_button_clicked(_):
        if False:
            for i in range(10):
                print('nop')
        watcher.clear_done()
    clear.on_click(on_clear_button_clicked)
    clear_button = widgets.GridBox(children=[clear], layout=widgets.Layout(width='100%', grid_template_columns='20% 20% 20% 20% 20%', grid_template_areas='\n                                       ". . . . right "\n                                        '))
    return clear_button

def make_labels():
    if False:
        for i in range(10):
            print('nop')
    'Makes the labels widget.\n\n    Returns:\n        widget: The labels widget.\n    '
    labels0 = widgets.HTML(value='<h5>Job ID</h5>', layout=widgets.Layout(width='190px'))
    labels1 = widgets.HTML(value='<h5>Backend</h5>', layout=widgets.Layout(width='145px'))
    labels2 = widgets.HTML(value='<h5>Status</h5>', layout=widgets.Layout(width='95px'))
    labels3 = widgets.HTML(value='<h5>Queue</h5>', layout=widgets.Layout(width='70px'))
    labels4 = widgets.HTML(value='<h5>Message</h5>')
    labels = widgets.HBox(children=[labels0, labels1, labels2, labels3, labels4], layout=widgets.Layout(width='600px', margin='0px 0px 0px 37px'))
    return labels

def create_job_widget(watcher, job, backend, status='', queue_pos=None, msg=''):
    if False:
        while True:
            i = 10
    'Creates a widget corresponding to a particular job instance.\n\n    Args:\n        watcher (widget): The job watcher instance.\n        job (IBMQJob): The job.\n        backend (str): The backend the job is running on.\n        status (str): The job status.\n        queue_pos (int): Queue position, if any.\n        msg (str): Job message, if any.\n\n    Returns:\n        widget: The job widget\n    '
    job_id = job.job_id()
    id_label = widgets.HTML(value=f'{job_id}', layout=widgets.Layout(width='190px'))
    backend_label = widgets.HTML(value=f'{backend}', layout=widgets.Layout(width='145px'))
    status_label = widgets.HTML(value=f'{status}', layout=widgets.Layout(width='95px'))
    if queue_pos is None:
        queue_pos = '-'
    else:
        queue_pos = str(queue_pos)
    queue_label = widgets.HTML(value=f'{queue_pos}', layout=widgets.Layout(width='70px'))
    msg_label = widgets.HTML(value=f'<p style=white-space:nowrap;>{msg}</p>', layout=widgets.Layout(overflow_x='scroll'))
    close_button = widgets.Button(button_style='', icon='close', layout=widgets.Layout(width='32px', margin='0px 5px 0px 0px'))
    close_button.style.button_color = 'white'

    def cancel_on_click(_):
        if False:
            return 10
        watcher.cancel_job(job_id)
    close_button.on_click(cancel_on_click)
    job_grid = widgets.HBox(children=[close_button, id_label, backend_label, status_label, queue_label, msg_label], layout=widgets.Layout(min_width='700px', max_width='700px'))
    job_grid.job_id = job_id
    job_grid.job = job
    return job_grid

def build_job_viewer():
    if False:
        return 10
    'Builds the job viewer widget\n\n    Returns:\n        widget: Job viewer.\n    '
    acc = widgets.Accordion(children=[widgets.VBox(layout=widgets.Layout(max_width='710px', min_width='710px'))], layout=widgets.Layout(width='auto', max_width='750px', max_height='500px', overflow_y='scroll', overflow_x='hidden'))
    acc.set_title(0, 'IBMQ Jobs')
    acc.selected_index = None
    acc.layout.visibility = 'hidden'
    display(acc)
    acc._dom_classes = ['job_widget']
    display(Javascript("\n        const isLab = window['Jupyter'] === undefined;\n        const notebook = document.querySelector( isLab ? 'div.jp-Notebook' : '#site');\n        const jobWidget = document.querySelector('div.job_widget');\n        notebook.prepend(jobWidget);\n        jobWidget.style.zIndex = '999';\n        jobWidget.style.position = isLab ? 'sticky' : 'fixed';\n        jobWidget.style.boxShadow = '5px 5px 5px -3px black';\n        jobWidget.style.opacity = '0.95';\n        if (isLab) {\n            jobWidget.style.top = '0';\n            jobWidget.style.left = '0';\n        }\n        "))
    acc.layout.visibility = 'visible'
    return acc