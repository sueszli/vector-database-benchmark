from pathlib import Path
from nicegui import app, ui
from .screen import Screen
example_file = Path(__file__).parent / '../examples/slideshow/slides/slide1.jpg'

def test_base64_image(screen: Screen):
    if False:
        while True:
            i = 10
    data = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABmJLR0QA/wD/AP+gvaeTAAAEHElEQVRoge2ZXUxbZRjHf6enH3QtBQ7paIFlMO2AMXTGqZE40bCpiRdzF06Nsu3O6G68MH5MnYkk3vhx4cWCJppFlvgZ74wXbsZdLCYaQMeWUWM30EJZgVM+WtpS2uNFoQzTU3pKu2O0v8v38//Pe57ned8cKFOmTBk9EVR7vrxsRlJ6gR7AfdMUrWcC6EcWTnK4fSnbAKPqVEl5C3ipRMLypR54GUkBeCXbAEOOyUdKoahAjqp15DKg12eTDZdaRy4DN43p+1s55HTwVF0Vk/taNM3V3UCDxUStSWQ4HKPDXsFwOK5pvm4GTILADquZbslGPKUAsNdRwXg8wQ6rOe911NPo2UvKplXmYOcWM957Par9wrnL6xv2786qVbcT8EUTSOdH+Co4T//kLE0XfgfgwcFRpPMjea+jm4GkohBaTuKxmhlaiNFoMZFS4Jf5KKHlZN7rqBeyEvPF7kYO11UBsKdyLUuGH2jjNV+Qt0en8lpHtxN41RfkyUt+APYPjfJNcJ7v5TB7f77KJxOhvNfRzcDVaPpqM51Ick6O4DQbuTC7yMBClMml5bzX0bUOdNgtXAzHAGi3WRiOaKsBoGMa1cy/LY0Wi7IBvfl/GhCAJ+qq+HbPdgL7Whi8+5YN59zjsOLr9ODr9PB6s7OQbbOiuRI7jAa+7tjGAcmeaQtukLdNgsBHbfWZW2atSdS6rSqaDAjAp7saOSDZSSoKpwOznJmcw7uYO3+/uL2W2+wVm9GpiiYD3ZKNg85KAI57A3w4vnHJv9Vq5o1mJ9FUCqMgYBLUS08haIqBY+4aAK5E4lyJxDnV4ub0rgaOuasRswgTgL7WeqwGA73XpjIPl2Ki6QQ6q6wAbDUb+fHO5kwZP+qu5qDTwaGLf64bf8RdTbdkYzgc492xGU40FS94V9F0Ai5L2q9kEunzyxz3BhhYiALwmLOSh24IbKfZyHseFykFnh0JkFBKczPRZMBqSA//eCLE894Ap/wyDw+NsZhMAWTiA+B9Tx21JpG+cZmf5haLKHk9mgysCp1bTmXaZhJJvIvpq3HTSpq83V7BM65qAHrc1chdrchdrdjE9HbPNUjIXa2bV49GA6tC22yWTJsoCLhXPq3ZRHKlbW1OpWigxihSYxQzMWMxCNQYi1MLNAXxZ9fnuKOygkckO0+7qjgrR3hhWy0uc3qZ72bCAPwWjmd9mPvv28kW0UDfuMyJP4JFkK/RwAd/zfD4Vgd3OaycaW9c1/dDKMLn1+eAtQf7P1kN41gqe38haPqE4imF7sFR3hmbZiyWIKEo+KJL9F6b4tFfx1jeINMMLcQYWIjijyU2JfpG/tMvsokSSSkAYVytJ5eB/hIoKQxBUdWiHsSycHLlz0gP6T8lepD+xTQjvKnT/mXKlCmzAX8Dl7JCqRHaepQAAAAASUVORK5CYII='
    ui.image(data).style('width: 50px;')
    screen.open('/')
    screen.wait(0.2)
    image = screen.find_by_class('q-img__image')
    assert 'data:image/png;base64,iVB' in image.get_attribute('src')

def test_setting_local_file(screen: Screen):
    if False:
        i = 10
        return i + 15
    ui.image(example_file)
    screen.open('/')
    image = screen.find_by_class('q-img__image')
    screen.should_load_image(image)

def test_binding_local_file(screen: Screen):
    if False:
        for i in range(10):
            print('nop')
    images = {'one': example_file}
    ui.image().bind_source_from(images, 'one')
    screen.open('/')
    image = screen.find_by_class('q-img__image')
    screen.should_load_image(image)

def test_set_source_with_local_file(screen: Screen):
    if False:
        return 10
    ui.image().set_source(example_file)
    screen.open('/')
    image = screen.find_by_class('q-img__image')
    screen.should_load_image(image)

def test_removal_of_generated_routes(screen: Screen):
    if False:
        while True:
            i = 10
    img = ui.image(example_file)
    ui.button('Slide 2', on_click=lambda : img.set_source(str(example_file).replace('slide1', 'slide2')))
    ui.button('Slide 3', on_click=lambda : img.set_source(str(example_file).replace('slide1', 'slide3')))
    screen.open('/')
    number_of_routes = len(app.routes)
    screen.click('Slide 2')
    screen.wait(0.5)
    assert len(app.routes) == number_of_routes
    screen.click('Slide 3')
    screen.wait(0.5)
    assert len(app.routes) == number_of_routes