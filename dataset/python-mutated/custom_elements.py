from .vector_elements import Vector, Rectangle
TEXT_INPUT_ELEMENT_TYPES = {'TextArea': 'Text', 'TextBox': 'Entry'}

class Button(Rectangle):

    def __init__(self, node, frame, image_path, *, id_):
        if False:
            while True:
                i = 10
        super().__init__(node, frame)
        self.image_path = image_path
        self.id_ = id_

    def to_code(self):
        if False:
            for i in range(10):
                print('nop')
        return f'\nbutton_image_{self.id_} = PhotoImage(\n    file=relative_to_assets("{self.image_path}"))\nbutton_{self.id_} = Button(\n    image=button_image_{self.id_},\n    borderwidth=0,\n    highlightthickness=0,\n    command=lambda: print("button_{self.id_} clicked"),\n    relief="flat"\n)\nbutton_{self.id_}.place(\n    x={self.x},\n    y={self.y},\n    width={self.width},\n    height={self.height}\n)\n'

class Text(Vector):

    def __init__(self, node, frame):
        if False:
            return 10
        super().__init__(node)
        (self.x, self.y) = self.position(frame)
        (self.width, self.height) = self.size()
        self.text_color = self.color()
        (self.font, self.font_size) = self.font_property()
        self.text = self.characters.replace('\n', '\\n')

    @property
    def characters(self) -> str:
        if False:
            i = 10
            return i + 15
        string: str = self.node.get('characters')
        text_case: str = self.style.get('textCase', 'ORIGINAL')
        if text_case == 'UPPER':
            string = string.upper()
        elif text_case == 'LOWER':
            string = string.lower()
        elif text_case == 'TITLE':
            string = string.title()
        return string

    @property
    def style(self):
        if False:
            while True:
                i = 10
        return self.node.get('style')

    @property
    def character_style_overrides(self):
        if False:
            while True:
                i = 10
        return self.node.get('characterStyleOverrides')

    @property
    def style_override_table(self):
        if False:
            i = 10
            return i + 15
        return self.node.get('styleOverrideTable')

    def font_property(self):
        if False:
            for i in range(10):
                print('nop')
        style = self.node.get('style')
        font_name = style.get('fontPostScriptName')
        if font_name is None:
            font_name = style['fontFamily']
        font_name = font_name.replace('-', ' ')
        font_size = style['fontSize']
        return (font_name, font_size)

    def to_code(self):
        if False:
            for i in range(10):
                print('nop')
        return f'\ncanvas.create_text(\n    {self.x},\n    {self.y},\n    anchor="nw",\n    text="{self.text}",\n    fill="{self.text_color}",\n    font=("{self.font}", {int(self.font_size)} * -1)\n)\n'

class Image(Vector):

    def __init__(self, node, frame, image_path, *, id_):
        if False:
            i = 10
            return i + 15
        super().__init__(node)
        (self.x, self.y) = self.position(frame)
        (width, height) = self.size()
        self.x += width // 2
        self.y += height // 2
        self.image_path = image_path
        self.id_ = id_

    def to_code(self):
        if False:
            while True:
                i = 10
        return f'\nimage_image_{self.id_} = PhotoImage(\n    file=relative_to_assets("{self.image_path}"))\nimage_{self.id_} = canvas.create_image(\n    {self.x},\n    {self.y},\n    image=image_image_{self.id_}\n)\n'

class TextEntry(Vector):

    def __init__(self, node, frame, image_path, *, id_):
        if False:
            print('Hello World!')
        super().__init__(node)
        self.id_ = id_
        self.image_path = image_path
        (self.x, self.y) = self.position(frame)
        (width, height) = self.size()
        self.x += width / 2
        self.y += height / 2
        self.bg_color = self.color()
        corner_radius = self.get('cornerRadius', 0)
        corner_radius = min(corner_radius, height / 2)
        self.entry_width = width - corner_radius * 2
        self.entry_height = height - 2
        (self.entry_x, self.entry_y) = self.position(frame)
        self.entry_x += corner_radius
        self.entry_type = TEXT_INPUT_ELEMENT_TYPES.get(self.get('name'))

    def to_code(self):
        if False:
            for i in range(10):
                print('nop')
        return f'\nentry_image_{self.id_} = PhotoImage(\n    file=relative_to_assets("{self.image_path}"))\nentry_bg_{self.id_} = canvas.create_image(\n    {self.x},\n    {self.y},\n    image=entry_image_{self.id_}\n)\nentry_{self.id_} = {self.entry_type}(\n    bd=0,\n    bg="{self.bg_color}",\n    fg="#000716",\n    highlightthickness=0\n)\nentry_{self.id_}.place(\n    x={self.entry_x},\n    y={self.entry_y},\n    width={self.entry_width},\n    height={self.entry_height}\n)\n'