from . import xmlwriter

class Vml(xmlwriter.XMLwriter):
    """
    A class for writing the Excel XLSX Vml file.


    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor.\n\n        '
        super(Vml, self).__init__()

    def _assemble_xml_file(self, data_id, vml_shape_id, comments_data=None, buttons_data=None, header_images_data=None):
        if False:
            i = 10
            return i + 15
        z_index = 1
        self._write_xml_namespace()
        self._write_shapelayout(data_id)
        if buttons_data:
            self._write_button_shapetype()
            for button in buttons_data:
                vml_shape_id += 1
                self._write_button_shape(vml_shape_id, z_index, button)
                z_index += 1
        if comments_data:
            self._write_comment_shapetype()
            for comment in comments_data:
                vml_shape_id += 1
                self._write_comment_shape(vml_shape_id, z_index, comment)
                z_index += 1
        if header_images_data:
            self._write_image_shapetype()
            index = 1
            for image in header_images_data:
                vml_shape_id += 1
                self._write_image_shape(vml_shape_id, index, image)
                index += 1
        self._xml_end_tag('xml')
        self._xml_close()

    def _pixels_to_points(self, vertices):
        if False:
            while True:
                i = 10
        (left, top, width, height) = vertices[8:12]
        left *= 0.75
        top *= 0.75
        width *= 0.75
        height *= 0.75
        return (left, top, width, height)

    def _write_xml_namespace(self):
        if False:
            i = 10
            return i + 15
        schema = 'urn:schemas-microsoft-com:'
        xmlns = schema + 'vml'
        xmlns_o = schema + 'office:office'
        xmlns_x = schema + 'office:excel'
        attributes = [('xmlns:v', xmlns), ('xmlns:o', xmlns_o), ('xmlns:x', xmlns_x)]
        self._xml_start_tag('xml', attributes)

    def _write_shapelayout(self, data_id):
        if False:
            print('Hello World!')
        attributes = [('v:ext', 'edit')]
        self._xml_start_tag('o:shapelayout', attributes)
        self._write_idmap(data_id)
        self._xml_end_tag('o:shapelayout')

    def _write_idmap(self, data_id):
        if False:
            print('Hello World!')
        attributes = [('v:ext', 'edit'), ('data', data_id)]
        self._xml_empty_tag('o:idmap', attributes)

    def _write_comment_shapetype(self):
        if False:
            print('Hello World!')
        shape_id = '_x0000_t202'
        coordsize = '21600,21600'
        spt = 202
        path = 'm,l,21600r21600,l21600,xe'
        attributes = [('id', shape_id), ('coordsize', coordsize), ('o:spt', spt), ('path', path)]
        self._xml_start_tag('v:shapetype', attributes)
        self._write_stroke()
        self._write_comment_path('t', 'rect')
        self._xml_end_tag('v:shapetype')

    def _write_button_shapetype(self):
        if False:
            print('Hello World!')
        shape_id = '_x0000_t201'
        coordsize = '21600,21600'
        spt = 201
        path = 'm,l,21600r21600,l21600,xe'
        attributes = [('id', shape_id), ('coordsize', coordsize), ('o:spt', spt), ('path', path)]
        self._xml_start_tag('v:shapetype', attributes)
        self._write_stroke()
        self._write_button_path()
        self._write_shapetype_lock()
        self._xml_end_tag('v:shapetype')

    def _write_image_shapetype(self):
        if False:
            print('Hello World!')
        shape_id = '_x0000_t75'
        coordsize = '21600,21600'
        spt = 75
        o_preferrelative = 't'
        path = 'm@4@5l@4@11@9@11@9@5xe'
        filled = 'f'
        stroked = 'f'
        attributes = [('id', shape_id), ('coordsize', coordsize), ('o:spt', spt), ('o:preferrelative', o_preferrelative), ('path', path), ('filled', filled), ('stroked', stroked)]
        self._xml_start_tag('v:shapetype', attributes)
        self._write_stroke()
        self._write_formulas()
        self._write_image_path()
        self._write_aspect_ratio_lock()
        self._xml_end_tag('v:shapetype')

    def _write_stroke(self):
        if False:
            i = 10
            return i + 15
        joinstyle = 'miter'
        attributes = [('joinstyle', joinstyle)]
        self._xml_empty_tag('v:stroke', attributes)

    def _write_comment_path(self, gradientshapeok, connecttype):
        if False:
            i = 10
            return i + 15
        attributes = []
        if gradientshapeok:
            attributes.append(('gradientshapeok', 't'))
        attributes.append(('o:connecttype', connecttype))
        self._xml_empty_tag('v:path', attributes)

    def _write_button_path(self):
        if False:
            print('Hello World!')
        shadowok = 'f'
        extrusionok = 'f'
        strokeok = 'f'
        fillok = 'f'
        connecttype = 'rect'
        attributes = [('shadowok', shadowok), ('o:extrusionok', extrusionok), ('strokeok', strokeok), ('fillok', fillok), ('o:connecttype', connecttype)]
        self._xml_empty_tag('v:path', attributes)

    def _write_image_path(self):
        if False:
            return 10
        extrusionok = 'f'
        gradientshapeok = 't'
        connecttype = 'rect'
        attributes = [('o:extrusionok', extrusionok), ('gradientshapeok', gradientshapeok), ('o:connecttype', connecttype)]
        self._xml_empty_tag('v:path', attributes)

    def _write_shapetype_lock(self):
        if False:
            while True:
                i = 10
        ext = 'edit'
        shapetype = 't'
        attributes = [('v:ext', ext), ('shapetype', shapetype)]
        self._xml_empty_tag('o:lock', attributes)

    def _write_rotation_lock(self):
        if False:
            for i in range(10):
                print('nop')
        ext = 'edit'
        rotation = 't'
        attributes = [('v:ext', ext), ('rotation', rotation)]
        self._xml_empty_tag('o:lock', attributes)

    def _write_aspect_ratio_lock(self):
        if False:
            return 10
        ext = 'edit'
        aspectratio = 't'
        attributes = [('v:ext', ext), ('aspectratio', aspectratio)]
        self._xml_empty_tag('o:lock', attributes)

    def _write_comment_shape(self, shape_id, z_index, comment):
        if False:
            while True:
                i = 10
        shape_type = '#_x0000_t202'
        insetmode = 'auto'
        visibility = 'hidden'
        shape_id = '_x0000_s' + str(shape_id)
        row = comment[0]
        col = comment[1]
        visible = comment[4]
        fillcolor = comment[5]
        vertices = comment[9]
        (left, top, width, height) = self._pixels_to_points(vertices)
        if visible:
            visibility = 'visible'
        style = 'position:absolute;margin-left:%.15gpt;margin-top:%.15gpt;width:%.15gpt;height:%.15gpt;z-index:%d;visibility:%s' % (left, top, width, height, z_index, visibility)
        attributes = [('id', shape_id), ('type', shape_type), ('style', style), ('fillcolor', fillcolor), ('o:insetmode', insetmode)]
        self._xml_start_tag('v:shape', attributes)
        self._write_comment_fill()
        self._write_shadow()
        self._write_comment_path(None, 'none')
        self._write_comment_textbox()
        self._write_comment_client_data(row, col, visible, vertices)
        self._xml_end_tag('v:shape')

    def _write_button_shape(self, shape_id, z_index, button):
        if False:
            for i in range(10):
                print('nop')
        shape_type = '#_x0000_t201'
        shape_id = '_x0000_s' + str(shape_id)
        vertices = button['vertices']
        (left, top, width, height) = self._pixels_to_points(vertices)
        style = 'position:absolute;margin-left:%.15gpt;margin-top:%.15gpt;width:%.15gpt;height:%.15gpt;z-index:%d;mso-wrap-style:tight' % (left, top, width, height, z_index)
        attributes = [('id', shape_id), ('type', shape_type)]
        if button.get('description'):
            attributes.append(('alt', button['description']))
        attributes.append(('style', style))
        attributes.append(('o:button', 't'))
        attributes.append(('fillcolor', 'buttonFace [67]'))
        attributes.append(('strokecolor', 'windowText [64]'))
        attributes.append(('o:insetmode', 'auto'))
        self._xml_start_tag('v:shape', attributes)
        self._write_button_fill()
        self._write_rotation_lock()
        self._write_button_textbox(button['font'])
        self._write_button_client_data(button)
        self._xml_end_tag('v:shape')

    def _write_image_shape(self, shape_id, z_index, image_data):
        if False:
            for i in range(10):
                print('nop')
        shape_type = '#_x0000_t75'
        shape_id = '_x0000_s' + str(shape_id)
        width = image_data[0]
        height = image_data[1]
        name = image_data[2]
        position = image_data[3]
        x_dpi = image_data[4]
        y_dpi = image_data[5]
        ref_id = image_data[6]
        width = width * 72.0 / x_dpi
        height = height * 72.0 / y_dpi
        width = 72.0 / 96 * int(width * 96.0 / 72 + 0.25)
        height = 72.0 / 96 * int(height * 96.0 / 72 + 0.25)
        style = 'position:absolute;margin-left:0;margin-top:0;width:%.15gpt;height:%.15gpt;z-index:%d' % (width, height, z_index)
        attributes = [('id', position), ('o:spid', shape_id), ('type', shape_type), ('style', style)]
        self._xml_start_tag('v:shape', attributes)
        self._write_imagedata(ref_id, name)
        self._write_rotation_lock()
        self._xml_end_tag('v:shape')

    def _write_comment_fill(self):
        if False:
            return 10
        color_2 = '#ffffe1'
        attributes = [('color2', color_2)]
        self._xml_empty_tag('v:fill', attributes)

    def _write_button_fill(self):
        if False:
            i = 10
            return i + 15
        color_2 = 'buttonFace [67]'
        detectmouseclick = 't'
        attributes = [('color2', color_2), ('o:detectmouseclick', detectmouseclick)]
        self._xml_empty_tag('v:fill', attributes)

    def _write_shadow(self):
        if False:
            i = 10
            return i + 15
        on = 't'
        color = 'black'
        obscured = 't'
        attributes = [('on', on), ('color', color), ('obscured', obscured)]
        self._xml_empty_tag('v:shadow', attributes)

    def _write_comment_textbox(self):
        if False:
            for i in range(10):
                print('nop')
        style = 'mso-direction-alt:auto'
        attributes = [('style', style)]
        self._xml_start_tag('v:textbox', attributes)
        self._write_div('left')
        self._xml_end_tag('v:textbox')

    def _write_button_textbox(self, font):
        if False:
            for i in range(10):
                print('nop')
        style = 'mso-direction-alt:auto'
        attributes = [('style', style), ('o:singleclick', 'f')]
        self._xml_start_tag('v:textbox', attributes)
        self._write_div('center', font)
        self._xml_end_tag('v:textbox')

    def _write_div(self, align, font=None):
        if False:
            for i in range(10):
                print('nop')
        style = 'text-align:' + align
        attributes = [('style', style)]
        self._xml_start_tag('div', attributes)
        if font:
            self._write_font(font)
        self._xml_end_tag('div')

    def _write_font(self, font):
        if False:
            for i in range(10):
                print('nop')
        caption = font['caption']
        face = 'Calibri'
        size = 220
        color = '#000000'
        attributes = [('face', face), ('size', size), ('color', color)]
        self._xml_data_element('font', caption, attributes)

    def _write_comment_client_data(self, row, col, visible, vertices):
        if False:
            while True:
                i = 10
        object_type = 'Note'
        attributes = [('ObjectType', object_type)]
        self._xml_start_tag('x:ClientData', attributes)
        self._write_move_with_cells()
        self._write_size_with_cells()
        self._write_anchor(vertices)
        self._write_auto_fill()
        self._write_row(row)
        self._write_column(col)
        if visible:
            self._write_visible()
        self._xml_end_tag('x:ClientData')

    def _write_button_client_data(self, button):
        if False:
            return 10
        macro = button['macro']
        vertices = button['vertices']
        object_type = 'Button'
        attributes = [('ObjectType', object_type)]
        self._xml_start_tag('x:ClientData', attributes)
        self._write_anchor(vertices)
        self._write_print_object()
        self._write_auto_fill()
        self._write_fmla_macro(macro)
        self._write_text_halign()
        self._write_text_valign()
        self._xml_end_tag('x:ClientData')

    def _write_move_with_cells(self):
        if False:
            print('Hello World!')
        self._xml_empty_tag('x:MoveWithCells')

    def _write_size_with_cells(self):
        if False:
            return 10
        self._xml_empty_tag('x:SizeWithCells')

    def _write_visible(self):
        if False:
            return 10
        self._xml_empty_tag('x:Visible')

    def _write_anchor(self, vertices):
        if False:
            return 10
        (col_start, row_start, x1, y1, col_end, row_end, x2, y2) = vertices[:8]
        strings = [col_start, x1, row_start, y1, col_end, x2, row_end, y2]
        strings = [str(i) for i in strings]
        data = ', '.join(strings)
        self._xml_data_element('x:Anchor', data)

    def _write_auto_fill(self):
        if False:
            for i in range(10):
                print('nop')
        data = 'False'
        self._xml_data_element('x:AutoFill', data)

    def _write_row(self, data):
        if False:
            while True:
                i = 10
        self._xml_data_element('x:Row', data)

    def _write_column(self, data):
        if False:
            return 10
        self._xml_data_element('x:Column', data)

    def _write_print_object(self):
        if False:
            print('Hello World!')
        self._xml_data_element('x:PrintObject', 'False')

    def _write_text_halign(self):
        if False:
            while True:
                i = 10
        self._xml_data_element('x:TextHAlign', 'Center')

    def _write_text_valign(self):
        if False:
            print('Hello World!')
        self._xml_data_element('x:TextVAlign', 'Center')

    def _write_fmla_macro(self, data):
        if False:
            i = 10
            return i + 15
        self._xml_data_element('x:FmlaMacro', data)

    def _write_imagedata(self, ref_id, o_title):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('o:relid', 'rId' + str(ref_id)), ('o:title', o_title)]
        self._xml_empty_tag('v:imagedata', attributes)

    def _write_formulas(self):
        if False:
            i = 10
            return i + 15
        self._xml_start_tag('v:formulas')
        self._write_formula('if lineDrawn pixelLineWidth 0')
        self._write_formula('sum @0 1 0')
        self._write_formula('sum 0 0 @1')
        self._write_formula('prod @2 1 2')
        self._write_formula('prod @3 21600 pixelWidth')
        self._write_formula('prod @3 21600 pixelHeight')
        self._write_formula('sum @0 0 1')
        self._write_formula('prod @6 1 2')
        self._write_formula('prod @7 21600 pixelWidth')
        self._write_formula('sum @8 21600 0')
        self._write_formula('prod @7 21600 pixelHeight')
        self._write_formula('sum @10 21600 0')
        self._xml_end_tag('v:formulas')

    def _write_formula(self, eqn):
        if False:
            while True:
                i = 10
        attributes = [('eqn', eqn)]
        self._xml_empty_tag('v:f', attributes)