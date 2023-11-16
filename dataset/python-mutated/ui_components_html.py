class GradioComponentsHTML:

    @staticmethod
    def get_html_header() -> str:
        if False:
            return 10
        'Create HTML for the header'
        return '\n            <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px;">\n            <h1 style="margin-left: 0px; font-size: 35px;">ShortGPT</h1>\n            <div style="flex-grow: 1; text-align: right;">\n                <a href="https://discord.gg/uERx39ru3R" target="_blank" style="text-decoration: none;">\n                <button style="margin-right: 10px; padding: 10px 20px; font-size: 16px; color: #fff; background-color: #7289DA; border: none; border-radius: 5px; cursor: pointer;">Join Discord</button>\n                </a>\n                <a href="https://github.com/RayVentura/ShortGPT" target="_blank" style="text-decoration: none;">\n                <button style="padding: 10px 20px; font-size: 16px; color: #fff; background-color: #333; border: none; border-radius: 5px; cursor: pointer;">Add a Star on Github ⭐</button>\n                </a>\n            </div>\n            </div>\n        '

    @staticmethod
    def get_html_error_template() -> str:
        if False:
            print('Hello World!')
        return "\n        <div style='text-align: center; background: #f2dede; color: #a94442; padding: 20px; border-radius: 5px; margin: 10px;'>\n          <h2 style='margin: 0;'>ERROR : {error_message}</h2>\n          <p style='margin: 10px 0;'>Traceback Info : {stack_trace}</p>\n          <p style='margin: 10px 0;'>If the problem persists, don't hesitate to contact our support. We're here to assist you.</p>\n          <a href='https://discord.gg/qn2WJaRH' target='_blank' style='background: #a94442; color: #fff; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; text-decoration: none;'>Get Help on Discord</a>\n        </div>\n        "

    @staticmethod
    def get_html_video_template(file_url_path, file_name, width='auto', height='auto'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate an HTML code snippet for embedding and downloading a video.\n\n        Parameters:\n        file_url_path (str): The URL or path to the video file.\n        file_name (str): The name of the video file.\n        width (str, optional): The width of the video. Defaults to "auto".\n        height (str, optional): The height of the video. Defaults to "auto".\n\n        Returns:\n        str: The generated HTML code snippet.\n        '
        html = f'\n            <div style="display: flex; flex-direction: column; align-items: center;">\n                <video width="{width}" height="{height}" style="max-height: 100%;" controls>\n                    <source src="{file_url_path}" type="video/mp4">\n                    Your browser does not support the video tag.\n                </video>\n                <a href="{file_url_path}" download="{file_name}" style="margin-top: 10px;">\n                    <button style="font-size: 1em; padding: 10px; border: none; cursor: pointer; color: white; background: #007bff;">Download Video</button>\n                </a>\n            </div>\n        '
        return html