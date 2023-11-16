from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        print('Hello World!')

    def alert():
        if False:
            return 10
        ui.run_javascript('alert("Hello!")')

    async def get_date():
        time = await ui.run_javascript('Date()')
        ui.notify(f'Browser time: {time}')

    def access_elements():
        if False:
            while True:
                i = 10
        ui.run_javascript(f'getElement({label.id}).innerText += " Hello!"')
    ui.button('fire and forget', on_click=alert)
    ui.button('receive result', on_click=get_date)
    ui.button('access elements', on_click=access_elements)
    label = ui.label()

def more() -> None:
    if False:
        for i in range(10):
            print('nop')

    @text_demo('Run async JavaScript', '\n        Using `run_javascript` you can also run asynchronous code in the browser.\n        The following demo shows how to get the current location of the user.\n    ')
    def run_async_javascript():
        if False:
            for i in range(10):
                print('nop')

        async def show_location():
            response = await ui.run_javascript("\n                return await new Promise((resolve, reject) => {\n                    if (!navigator.geolocation) {\n                        reject(new Error('Geolocation is not supported by your browser'));\n                    } else {\n                        navigator.geolocation.getCurrentPosition(\n                            (position) => {\n                                resolve({\n                                    latitude: position.coords.latitude,\n                                    longitude: position.coords.longitude,\n                                });\n                            },\n                            () => {\n                                reject(new Error('Unable to retrieve your location'));\n                            }\n                        );\n                    }\n                });\n            ", timeout=5.0)
            ui.notify(f"Your location is {response['latitude']}, {response['longitude']}")
        ui.button('Show location', on_click=show_location)