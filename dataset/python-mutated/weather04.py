import httpx
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Input, Static
from textual.worker import Worker

class WeatherApp(App):
    """App to display the current weather."""
    CSS_PATH = 'weather.tcss'

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Input(placeholder='Enter a City')
        with VerticalScroll(id='weather-container'):
            yield Static(id='weather')

    async def on_input_changed(self, message: Input.Changed) -> None:
        """Called when the input changes"""
        self.update_weather(message.value)

    @work(exclusive=True)
    async def update_weather(self, city: str) -> None:
        """Update the weather for the given city."""
        weather_widget = self.query_one('#weather', Static)
        if city:
            url = f'https://wttr.in/{city}'
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                weather = Text.from_ansi(response.text)
                weather_widget.update(weather)
        else:
            weather_widget.update('')

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if False:
            while True:
                i = 10
        'Called when the worker state changes.'
        self.log(event)
if __name__ == '__main__':
    app = WeatherApp()
    app.run()