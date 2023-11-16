from . import xmlwriter
from .shape import Shape
from .utility import get_rgb_color

class Drawing(xmlwriter.XMLwriter):
    """
    A class for writing the Excel XLSX Drawing file.


    """

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Constructor.\n\n        '
        super(Drawing, self).__init__()
        self.drawings = []
        self.embedded = 0
        self.orientation = 0

    def _assemble_xml_file(self):
        if False:
            i = 10
            return i + 15
        self._xml_declaration()
        self._write_drawing_workspace()
        if self.embedded:
            index = 0
            for drawing_properties in self.drawings:
                index += 1
                self._write_two_cell_anchor(index, drawing_properties)
        else:
            self._write_absolute_anchor(1)
        self._xml_end_tag('xdr:wsDr')
        self._xml_close()

    def _add_drawing_object(self):
        if False:
            while True:
                i = 10
        drawing_object = {'anchor_type': None, 'dimensions': [], 'width': 0, 'height': 0, 'shape': None, 'anchor': None, 'rel_index': 0, 'url_rel_index': 0, 'tip': None, 'name': None, 'description': None, 'decorative': False}
        self.drawings.append(drawing_object)
        return drawing_object

    def _write_drawing_workspace(self):
        if False:
            print('Hello World!')
        schema = 'http://schemas.openxmlformats.org/drawingml/'
        xmlns_xdr = schema + '2006/spreadsheetDrawing'
        xmlns_a = schema + '2006/main'
        attributes = [('xmlns:xdr', xmlns_xdr), ('xmlns:a', xmlns_a)]
        self._xml_start_tag('xdr:wsDr', attributes)

    def _write_two_cell_anchor(self, index, drawing_properties):
        if False:
            for i in range(10):
                print('nop')
        anchor_type = drawing_properties['type']
        dimensions = drawing_properties['dimensions']
        col_from = dimensions[0]
        row_from = dimensions[1]
        col_from_offset = dimensions[2]
        row_from_offset = dimensions[3]
        col_to = dimensions[4]
        row_to = dimensions[5]
        col_to_offset = dimensions[6]
        row_to_offset = dimensions[7]
        col_absolute = dimensions[8]
        row_absolute = dimensions[9]
        width = drawing_properties['width']
        height = drawing_properties['height']
        shape = drawing_properties['shape']
        anchor = drawing_properties['anchor']
        rel_index = drawing_properties['rel_index']
        url_rel_index = drawing_properties['url_rel_index']
        tip = drawing_properties['tip']
        name = drawing_properties['name']
        description = drawing_properties['description']
        decorative = drawing_properties['decorative']
        attributes = []
        if anchor == 2:
            attributes.append(('editAs', 'oneCell'))
        elif anchor == 3:
            attributes.append(('editAs', 'absolute'))
        if shape and shape.edit_as:
            attributes.append(('editAs', shape.edit_as))
        self._xml_start_tag('xdr:twoCellAnchor', attributes)
        self._write_from(col_from, row_from, col_from_offset, row_from_offset)
        self._write_to(col_to, row_to, col_to_offset, row_to_offset)
        if anchor_type == 1:
            self._write_graphic_frame(index, rel_index, name, description, decorative)
        elif anchor_type == 2:
            self._write_pic(index, rel_index, col_absolute, row_absolute, width, height, shape, description, url_rel_index, tip, decorative)
        else:
            self._write_sp(index, col_absolute, row_absolute, width, height, shape, description, url_rel_index, tip, decorative)
        self._write_client_data()
        self._xml_end_tag('xdr:twoCellAnchor')

    def _write_absolute_anchor(self, frame_index):
        if False:
            print('Hello World!')
        self._xml_start_tag('xdr:absoluteAnchor')
        if self.orientation == 0:
            self._write_pos(0, 0)
            self._write_xdr_ext(9308969, 6078325)
        else:
            self._write_pos(0, -47625)
            self._write_xdr_ext(6162675, 6124575)
        self._write_graphic_frame(frame_index, frame_index)
        self._write_client_data()
        self._xml_end_tag('xdr:absoluteAnchor')

    def _write_from(self, col, row, col_offset, row_offset):
        if False:
            while True:
                i = 10
        self._xml_start_tag('xdr:from')
        self._write_col(col)
        self._write_col_off(col_offset)
        self._write_row(row)
        self._write_row_off(row_offset)
        self._xml_end_tag('xdr:from')

    def _write_to(self, col, row, col_offset, row_offset):
        if False:
            while True:
                i = 10
        self._xml_start_tag('xdr:to')
        self._write_col(col)
        self._write_col_off(col_offset)
        self._write_row(row)
        self._write_row_off(row_offset)
        self._xml_end_tag('xdr:to')

    def _write_col(self, data):
        if False:
            for i in range(10):
                print('nop')
        self._xml_data_element('xdr:col', data)

    def _write_col_off(self, data):
        if False:
            while True:
                i = 10
        self._xml_data_element('xdr:colOff', data)

    def _write_row(self, data):
        if False:
            i = 10
            return i + 15
        self._xml_data_element('xdr:row', data)

    def _write_row_off(self, data):
        if False:
            while True:
                i = 10
        self._xml_data_element('xdr:rowOff', data)

    def _write_pos(self, x, y):
        if False:
            while True:
                i = 10
        attributes = [('x', x), ('y', y)]
        self._xml_empty_tag('xdr:pos', attributes)

    def _write_xdr_ext(self, cx, cy):
        if False:
            while True:
                i = 10
        attributes = [('cx', cx), ('cy', cy)]
        self._xml_empty_tag('xdr:ext', attributes)

    def _write_graphic_frame(self, index, rel_index, name=None, description=None, decorative=None):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('macro', '')]
        self._xml_start_tag('xdr:graphicFrame', attributes)
        self._write_nv_graphic_frame_pr(index, name, description, decorative)
        self._write_xfrm()
        self._write_atag_graphic(rel_index)
        self._xml_end_tag('xdr:graphicFrame')

    def _write_nv_graphic_frame_pr(self, index, name, description, decorative):
        if False:
            i = 10
            return i + 15
        if not name:
            name = 'Chart ' + str(index)
        self._xml_start_tag('xdr:nvGraphicFramePr')
        self._write_c_nv_pr(index + 1, name, description, None, None, decorative)
        self._write_c_nv_graphic_frame_pr()
        self._xml_end_tag('xdr:nvGraphicFramePr')

    def _write_c_nv_pr(self, index, name, description, url_rel_index, tip, decorative):
        if False:
            print('Hello World!')
        attributes = [('id', index), ('name', name)]
        if description and (not decorative):
            attributes.append(('descr', description))
        if url_rel_index or decorative:
            self._xml_start_tag('xdr:cNvPr', attributes)
            if url_rel_index:
                self._write_a_hlink_click(url_rel_index, tip)
            if decorative:
                self._write_decorative()
            self._xml_end_tag('xdr:cNvPr')
        else:
            self._xml_empty_tag('xdr:cNvPr', attributes)

    def _write_decorative(self):
        if False:
            i = 10
            return i + 15
        self._xml_start_tag('a:extLst')
        self._write_uri_ext('{FF2B5EF4-FFF2-40B4-BE49-F238E27FC236}')
        self._write_a16_creation_id()
        self._xml_end_tag('a:ext')
        self._write_uri_ext('{C183D7F6-B498-43B3-948B-1728B52AA6E4}')
        self._write_adec_decorative()
        self._xml_end_tag('a:ext')
        self._xml_end_tag('a:extLst')

    def _write_uri_ext(self, uri):
        if False:
            print('Hello World!')
        attributes = [('uri', uri)]
        self._xml_start_tag('a:ext', attributes)

    def _write_adec_decorative(self):
        if False:
            print('Hello World!')
        xmlns = 'http://schemas.microsoft.com/office/drawing/2017/decorative'
        val = '1'
        attributes = [('xmlns:adec', xmlns), ('val', val)]
        self._xml_empty_tag('adec:decorative', attributes)

    def _write_a16_creation_id(self):
        if False:
            return 10
        xmlns_a_16 = 'http://schemas.microsoft.com/office/drawing/2014/main'
        creation_id = '{00000000-0008-0000-0000-000002000000}'
        attributes = [('xmlns:a16', xmlns_a_16), ('id', creation_id)]
        self._xml_empty_tag('a16:creationId', attributes)

    def _write_a_hlink_click(self, rel_index, tip):
        if False:
            print('Hello World!')
        schema = 'http://schemas.openxmlformats.org/officeDocument/'
        xmlns_r = schema + '2006/relationships'
        attributes = [('xmlns:r', xmlns_r), ('r:id', 'rId' + str(rel_index))]
        if tip:
            attributes.append(('tooltip', tip))
        self._xml_empty_tag('a:hlinkClick', attributes)

    def _write_c_nv_graphic_frame_pr(self):
        if False:
            i = 10
            return i + 15
        if self.embedded:
            self._xml_empty_tag('xdr:cNvGraphicFramePr')
        else:
            self._xml_start_tag('xdr:cNvGraphicFramePr')
            self._write_a_graphic_frame_locks()
            self._xml_end_tag('xdr:cNvGraphicFramePr')

    def _write_a_graphic_frame_locks(self):
        if False:
            return 10
        attributes = [('noGrp', 1)]
        self._xml_empty_tag('a:graphicFrameLocks', attributes)

    def _write_xfrm(self):
        if False:
            i = 10
            return i + 15
        self._xml_start_tag('xdr:xfrm')
        self._write_xfrm_offset()
        self._write_xfrm_extension()
        self._xml_end_tag('xdr:xfrm')

    def _write_xfrm_offset(self):
        if False:
            return 10
        attributes = [('x', 0), ('y', 0)]
        self._xml_empty_tag('a:off', attributes)

    def _write_xfrm_extension(self):
        if False:
            return 10
        attributes = [('cx', 0), ('cy', 0)]
        self._xml_empty_tag('a:ext', attributes)

    def _write_atag_graphic(self, index):
        if False:
            while True:
                i = 10
        self._xml_start_tag('a:graphic')
        self._write_atag_graphic_data(index)
        self._xml_end_tag('a:graphic')

    def _write_atag_graphic_data(self, index):
        if False:
            i = 10
            return i + 15
        uri = 'http://schemas.openxmlformats.org/drawingml/2006/chart'
        attributes = [('uri', uri)]
        self._xml_start_tag('a:graphicData', attributes)
        self._write_c_chart('rId' + str(index))
        self._xml_end_tag('a:graphicData')

    def _write_c_chart(self, r_id):
        if False:
            print('Hello World!')
        schema = 'http://schemas.openxmlformats.org/'
        xmlns_c = schema + 'drawingml/2006/chart'
        xmlns_r = schema + 'officeDocument/2006/relationships'
        attributes = [('xmlns:c', xmlns_c), ('xmlns:r', xmlns_r), ('r:id', r_id)]
        self._xml_empty_tag('c:chart', attributes)

    def _write_client_data(self):
        if False:
            while True:
                i = 10
        self._xml_empty_tag('xdr:clientData')

    def _write_sp(self, index, col_absolute, row_absolute, width, height, shape, description, url_rel_index, tip, decorative):
        if False:
            i = 10
            return i + 15
        if shape and shape.connect:
            attributes = [('macro', '')]
            self._xml_start_tag('xdr:cxnSp', attributes)
            self._write_nv_cxn_sp_pr(index, shape)
            self._write_xdr_sp_pr(index, col_absolute, row_absolute, width, height, shape)
            self._xml_end_tag('xdr:cxnSp')
        else:
            attributes = [('macro', ''), ('textlink', shape.textlink)]
            self._xml_start_tag('xdr:sp', attributes)
            self._write_nv_sp_pr(index, shape, url_rel_index, tip, description, decorative)
            self._write_xdr_sp_pr(index, col_absolute, row_absolute, width, height, shape)
            self._write_style()
            if shape.text is not None:
                self._write_tx_body(col_absolute, row_absolute, width, height, shape)
            self._xml_end_tag('xdr:sp')

    def _write_nv_cxn_sp_pr(self, index, shape):
        if False:
            return 10
        self._xml_start_tag('xdr:nvCxnSpPr')
        name = shape.name + ' ' + str(index)
        if name is not None:
            self._write_c_nv_pr(index, name, None, None, None, None)
        self._xml_start_tag('xdr:cNvCxnSpPr')
        attributes = [('noChangeShapeType', '1')]
        self._xml_empty_tag('a:cxnSpLocks', attributes)
        if shape.start:
            attributes = [('id', shape.start), ('idx', shape.start_index)]
            self._xml_empty_tag('a:stCxn', attributes)
        if shape.end:
            attributes = [('id', shape.end), ('idx', shape.end_index)]
            self._xml_empty_tag('a:endCxn', attributes)
        self._xml_end_tag('xdr:cNvCxnSpPr')
        self._xml_end_tag('xdr:nvCxnSpPr')

    def _write_nv_sp_pr(self, index, shape, url_rel_index, tip, description, decorative):
        if False:
            i = 10
            return i + 15
        attributes = []
        self._xml_start_tag('xdr:nvSpPr')
        name = shape.name + ' ' + str(index)
        self._write_c_nv_pr(index + 1, name, description, url_rel_index, tip, decorative)
        if shape.name == 'TextBox':
            attributes = [('txBox', 1)]
        self._xml_empty_tag('xdr:cNvSpPr', attributes)
        self._xml_end_tag('xdr:nvSpPr')

    def _write_pic(self, index, rel_index, col_absolute, row_absolute, width, height, shape, description, url_rel_index, tip, decorative):
        if False:
            for i in range(10):
                print('nop')
        self._xml_start_tag('xdr:pic')
        self._write_nv_pic_pr(index, rel_index, description, url_rel_index, tip, decorative)
        self._write_blip_fill(rel_index)
        self._write_sp_pr(col_absolute, row_absolute, width, height, shape)
        self._xml_end_tag('xdr:pic')

    def _write_nv_pic_pr(self, index, rel_index, description, url_rel_index, tip, decorative):
        if False:
            print('Hello World!')
        self._xml_start_tag('xdr:nvPicPr')
        self._write_c_nv_pr(index + 1, 'Picture ' + str(index), description, url_rel_index, tip, decorative)
        self._write_c_nv_pic_pr()
        self._xml_end_tag('xdr:nvPicPr')

    def _write_c_nv_pic_pr(self):
        if False:
            while True:
                i = 10
        self._xml_start_tag('xdr:cNvPicPr')
        self._write_a_pic_locks()
        self._xml_end_tag('xdr:cNvPicPr')

    def _write_a_pic_locks(self):
        if False:
            while True:
                i = 10
        attributes = [('noChangeAspect', 1)]
        self._xml_empty_tag('a:picLocks', attributes)

    def _write_blip_fill(self, index):
        if False:
            print('Hello World!')
        self._xml_start_tag('xdr:blipFill')
        self._write_a_blip(index)
        self._write_a_stretch()
        self._xml_end_tag('xdr:blipFill')

    def _write_a_blip(self, index):
        if False:
            print('Hello World!')
        schema = 'http://schemas.openxmlformats.org/officeDocument/'
        xmlns_r = schema + '2006/relationships'
        r_embed = 'rId' + str(index)
        attributes = [('xmlns:r', xmlns_r), ('r:embed', r_embed)]
        self._xml_empty_tag('a:blip', attributes)

    def _write_a_stretch(self):
        if False:
            for i in range(10):
                print('nop')
        self._xml_start_tag('a:stretch')
        self._write_a_fill_rect()
        self._xml_end_tag('a:stretch')

    def _write_a_fill_rect(self):
        if False:
            for i in range(10):
                print('nop')
        self._xml_empty_tag('a:fillRect')

    def _write_sp_pr(self, col_absolute, row_absolute, width, height, shape=None):
        if False:
            print('Hello World!')
        self._xml_start_tag('xdr:spPr')
        self._write_a_xfrm(col_absolute, row_absolute, width, height)
        self._write_a_prst_geom(shape)
        self._xml_end_tag('xdr:spPr')

    def _write_xdr_sp_pr(self, index, col_absolute, row_absolute, width, height, shape):
        if False:
            print('Hello World!')
        self._xml_start_tag('xdr:spPr')
        self._write_a_xfrm(col_absolute, row_absolute, width, height, shape)
        self._write_a_prst_geom(shape)
        if shape.fill:
            if not shape.fill['defined']:
                self._write_a_solid_fill_scheme('lt1')
            elif 'none' in shape.fill:
                self._xml_empty_tag('a:noFill')
            elif 'color' in shape.fill:
                self._write_a_solid_fill(get_rgb_color(shape.fill['color']))
        if shape.gradient:
            self._write_a_grad_fill(shape.gradient)
        self._write_a_ln(shape.line)
        self._xml_end_tag('xdr:spPr')

    def _write_a_xfrm(self, col_absolute, row_absolute, width, height, shape=None):
        if False:
            while True:
                i = 10
        attributes = []
        if shape:
            if shape.rotation:
                rotation = shape.rotation
                rotation *= 60000
                attributes.append(('rot', rotation))
            if shape.flip_h:
                attributes.append(('flipH', 1))
            if shape.flip_v:
                attributes.append(('flipV', 1))
        self._xml_start_tag('a:xfrm', attributes)
        self._write_a_off(col_absolute, row_absolute)
        self._write_a_ext(width, height)
        self._xml_end_tag('a:xfrm')

    def _write_a_off(self, x, y):
        if False:
            print('Hello World!')
        attributes = [('x', x), ('y', y)]
        self._xml_empty_tag('a:off', attributes)

    def _write_a_ext(self, cx, cy):
        if False:
            return 10
        attributes = [('cx', cx), ('cy', cy)]
        self._xml_empty_tag('a:ext', attributes)

    def _write_a_prst_geom(self, shape=None):
        if False:
            while True:
                i = 10
        attributes = [('prst', 'rect')]
        self._xml_start_tag('a:prstGeom', attributes)
        self._write_a_av_lst(shape)
        self._xml_end_tag('a:prstGeom')

    def _write_a_av_lst(self, shape=None):
        if False:
            i = 10
            return i + 15
        adjustments = []
        if shape and shape.adjustments:
            adjustments = shape.adjustments
        if adjustments:
            self._xml_start_tag('a:avLst')
            i = 0
            for adj in adjustments:
                i += 1
                if shape.connect:
                    suffix = i
                else:
                    suffix = ''
                adj_int = str(int(adj * 1000))
                attributes = [('name', 'adj' + suffix), ('fmla', 'val' + adj_int)]
                self._xml_empty_tag('a:gd', attributes)
            self._xml_end_tag('a:avLst')
        else:
            self._xml_empty_tag('a:avLst')

    def _write_a_solid_fill(self, rgb):
        if False:
            while True:
                i = 10
        if rgb is None:
            rgb = 'FFFFFF'
        self._xml_start_tag('a:solidFill')
        self._write_a_srgb_clr(rgb)
        self._xml_end_tag('a:solidFill')

    def _write_a_solid_fill_scheme(self, color, shade=None):
        if False:
            return 10
        attributes = [('val', color)]
        self._xml_start_tag('a:solidFill')
        if shade:
            self._xml_start_tag('a:schemeClr', attributes)
            self._write_a_shade(shade)
            self._xml_end_tag('a:schemeClr')
        else:
            self._xml_empty_tag('a:schemeClr', attributes)
        self._xml_end_tag('a:solidFill')

    def _write_a_ln(self, line):
        if False:
            while True:
                i = 10
        width = line.get('width', 0.75)
        width = int((width + 0.125) * 4) / 4.0
        width = int(0.5 + 12700 * width)
        attributes = [('w', width), ('cmpd', 'sng')]
        self._xml_start_tag('a:ln', attributes)
        if 'none' in line:
            self._xml_empty_tag('a:noFill')
        elif 'color' in line:
            self._write_a_solid_fill(get_rgb_color(line['color']))
        else:
            self._write_a_solid_fill_scheme('lt1', '50000')
        line_type = line.get('dash_type')
        if line_type:
            self._write_a_prst_dash(line_type)
        self._xml_end_tag('a:ln')

    def _write_tx_body(self, col_absolute, row_absolute, width, height, shape):
        if False:
            return 10
        attributes = []
        if shape.text_rotation != 0:
            if shape.text_rotation == 90:
                attributes.append(('vert', 'vert270'))
            if shape.text_rotation == -90:
                attributes.append(('vert', 'vert'))
            if shape.text_rotation == 270:
                attributes.append(('vert', 'wordArtVert'))
            if shape.text_rotation == 271:
                attributes.append(('vert', 'eaVert'))
        attributes.append(('wrap', 'square'))
        attributes.append(('rtlCol', '0'))
        if not shape.align['defined']:
            attributes.append(('anchor', 't'))
        else:
            if 'vertical' in shape.align:
                align = shape.align['vertical']
                if align == 'top':
                    attributes.append(('anchor', 't'))
                elif align == 'middle':
                    attributes.append(('anchor', 'ctr'))
                elif align == 'bottom':
                    attributes.append(('anchor', 'b'))
            else:
                attributes.append(('anchor', 't'))
            if 'horizontal' in shape.align:
                align = shape.align['horizontal']
                if align == 'center':
                    attributes.append(('anchorCtr', '1'))
            else:
                attributes.append(('anchorCtr', '0'))
        self._xml_start_tag('xdr:txBody')
        self._xml_empty_tag('a:bodyPr', attributes)
        self._xml_empty_tag('a:lstStyle')
        lines = shape.text.split('\n')
        font = shape.font
        style_attrs = Shape._get_font_style_attributes(font)
        latin_attrs = Shape._get_font_latin_attributes(font)
        style_attrs.insert(0, ('lang', font['lang']))
        if shape.textlink != '':
            attributes = [('id', '{B8ADDEFE-BF52-4FD4-8C5D-6B85EF6FF707}'), ('type', 'TxLink')]
            self._xml_start_tag('a:p')
            self._xml_start_tag('a:fld', attributes)
            self._write_font_run(font, style_attrs, latin_attrs, 'a:rPr')
            self._xml_data_element('a:t', shape.text)
            self._xml_end_tag('a:fld')
            self._write_font_run(font, style_attrs, latin_attrs, 'a:endParaRPr')
            self._xml_end_tag('a:p')
        else:
            for line in lines:
                self._xml_start_tag('a:p')
                if line == '':
                    self._write_font_run(font, style_attrs, latin_attrs, 'a:endParaRPr')
                    self._xml_end_tag('a:p')
                    continue
                elif 'text' in shape.align:
                    if shape.align['text'] == 'left':
                        self._xml_empty_tag('a:pPr', [('algn', 'l')])
                    if shape.align['text'] == 'center':
                        self._xml_empty_tag('a:pPr', [('algn', 'ctr')])
                    if shape.align['text'] == 'right':
                        self._xml_empty_tag('a:pPr', [('algn', 'r')])
                self._xml_start_tag('a:r')
                self._write_font_run(font, style_attrs, latin_attrs, 'a:rPr')
                self._xml_data_element('a:t', line)
                self._xml_end_tag('a:r')
                self._xml_end_tag('a:p')
        self._xml_end_tag('xdr:txBody')

    def _write_font_run(self, font, style_attrs, latin_attrs, run_type):
        if False:
            print('Hello World!')
        if font.get('color') is not None:
            has_color = True
        else:
            has_color = False
        if latin_attrs or has_color:
            self._xml_start_tag(run_type, style_attrs)
            if has_color:
                self._write_a_solid_fill(get_rgb_color(font['color']))
            if latin_attrs:
                self._write_a_latin(latin_attrs)
                self._write_a_cs(latin_attrs)
            self._xml_end_tag(run_type)
        else:
            self._xml_empty_tag(run_type, style_attrs)

    def _write_style(self):
        if False:
            while True:
                i = 10
        self._xml_start_tag('xdr:style')
        self._write_a_ln_ref()
        self._write_a_fill_ref()
        self._write_a_effect_ref()
        self._write_a_font_ref()
        self._xml_end_tag('xdr:style')

    def _write_a_ln_ref(self):
        if False:
            return 10
        attributes = [('idx', '0')]
        self._xml_start_tag('a:lnRef', attributes)
        self._write_a_scrgb_clr()
        self._xml_end_tag('a:lnRef')

    def _write_a_fill_ref(self):
        if False:
            i = 10
            return i + 15
        attributes = [('idx', '0')]
        self._xml_start_tag('a:fillRef', attributes)
        self._write_a_scrgb_clr()
        self._xml_end_tag('a:fillRef')

    def _write_a_effect_ref(self):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('idx', '0')]
        self._xml_start_tag('a:effectRef', attributes)
        self._write_a_scrgb_clr()
        self._xml_end_tag('a:effectRef')

    def _write_a_scrgb_clr(self):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('r', '0'), ('g', '0'), ('b', '0')]
        self._xml_empty_tag('a:scrgbClr', attributes)

    def _write_a_font_ref(self):
        if False:
            print('Hello World!')
        attributes = [('idx', 'minor')]
        self._xml_start_tag('a:fontRef', attributes)
        self._write_a_scheme_clr('dk1')
        self._xml_end_tag('a:fontRef')

    def _write_a_scheme_clr(self, val):
        if False:
            return 10
        attributes = [('val', val)]
        self._xml_empty_tag('a:schemeClr', attributes)

    def _write_a_shade(self, shade):
        if False:
            i = 10
            return i + 15
        attributes = [('val', shade)]
        self._xml_empty_tag('a:shade', attributes)

    def _write_a_prst_dash(self, val):
        if False:
            while True:
                i = 10
        attributes = [('val', val)]
        self._xml_empty_tag('a:prstDash', attributes)

    def _write_a_grad_fill(self, gradient):
        if False:
            i = 10
            return i + 15
        attributes = [('flip', 'none'), ('rotWithShape', '1')]
        if gradient['type'] == 'linear':
            attributes = []
        self._xml_start_tag('a:gradFill', attributes)
        self._write_a_gs_lst(gradient)
        if gradient['type'] == 'linear':
            self._write_a_lin(gradient['angle'])
        else:
            self._write_a_path(gradient['type'])
            self._write_a_tile_rect(gradient['type'])
        self._xml_end_tag('a:gradFill')

    def _write_a_gs_lst(self, gradient):
        if False:
            for i in range(10):
                print('nop')
        positions = gradient['positions']
        colors = gradient['colors']
        self._xml_start_tag('a:gsLst')
        for i in range(len(colors)):
            pos = int(positions[i] * 1000)
            attributes = [('pos', pos)]
            self._xml_start_tag('a:gs', attributes)
            color = get_rgb_color(colors[i])
            self._write_a_srgb_clr(color)
            self._xml_end_tag('a:gs')
        self._xml_end_tag('a:gsLst')

    def _write_a_lin(self, angle):
        if False:
            for i in range(10):
                print('nop')
        angle = int(60000 * angle)
        attributes = [('ang', angle), ('scaled', '0')]
        self._xml_empty_tag('a:lin', attributes)

    def _write_a_path(self, gradient_type):
        if False:
            return 10
        attributes = [('path', gradient_type)]
        self._xml_start_tag('a:path', attributes)
        self._write_a_fill_to_rect(gradient_type)
        self._xml_end_tag('a:path')

    def _write_a_fill_to_rect(self, gradient_type):
        if False:
            while True:
                i = 10
        if gradient_type == 'shape':
            attributes = [('l', '50000'), ('t', '50000'), ('r', '50000'), ('b', '50000')]
        else:
            attributes = [('l', '100000'), ('t', '100000')]
        self._xml_empty_tag('a:fillToRect', attributes)

    def _write_a_tile_rect(self, gradient_type):
        if False:
            for i in range(10):
                print('nop')
        if gradient_type == 'shape':
            attributes = []
        else:
            attributes = [('r', '-100000'), ('b', '-100000')]
        self._xml_empty_tag('a:tileRect', attributes)

    def _write_a_srgb_clr(self, val):
        if False:
            print('Hello World!')
        attributes = [('val', val)]
        self._xml_empty_tag('a:srgbClr', attributes)

    def _write_a_latin(self, attributes):
        if False:
            while True:
                i = 10
        self._xml_empty_tag('a:latin', attributes)

    def _write_a_cs(self, attributes):
        if False:
            while True:
                i = 10
        self._xml_empty_tag('a:cs', attributes)