import sys, os
from calibre.ebooks.rtf2xml import headings_to_sections, line_endings, footnote, fields_small, default_encoding, make_lists, preamble_div, header, colors, group_borders, check_encoding, add_brackets, table, combine_borders, fields_large, process_tokens, hex_2_utf8, tokenize, delete_info, sections, check_brackets, styles, paragraph_def, convert_to_tags, output, copy, list_numbers, info, pict, table_info, fonts, paragraphs, body_styles, preamble_rest, group_styles, inline
from calibre.ebooks.rtf2xml.old_rtf import OldRtf
from . import open_for_read, open_for_write
"\nHere is an example script using the ParseRTF module directly\n#!/usr/bin/env python\n\ndef Handle_Main():\n    # Handles options and creates a parse object\n    parse_obj =ParseRtf.ParseRtf(\n            in_file = 'in.rtf',\n            # All values from here on are optional\n            # determine the output file\n            out_file = 'out.xml',\n            # determine the run level. The default is 1.\n            run_level = 3,\n            # The name of a debug directory, if you are running at\n            # run level 3 or higher.\n            debug = 'debug_dir',\n            # Convert RTF caps to real caps.\n            # Default is 1.\n            convert_caps = 1,\n            # Indent resulting XML.\n            # Default is 0 (no indent).\n            indent = 1,\n            # Form lists from RTF. Default is 1.\n            form_lists = 1,\n            # Convert headings to sections. Default is 0.\n            headings_to_sections = 1,\n            # Group paragraphs with the same style name. Default is 1.\n            group_styles = 1,\n            # Group borders. Default is 1.\n            group_borders = 1,\n            # Write or do not write paragraphs. Default is 0.\n            empty_paragraphs = 0,\n            # Allow to use a custom default encoding as fallback\n            default_encoding = 'cp1252',\n    )\n    try:\n        parse_obj.parse_rtf()\n    except ParseRtf.InvalidRtfException, msg:\n        sys.stderr.write(msg)\n    except ParseRtf.RtfInvalidCodeException, msg:\n        sys.stderr.write(msg)\n"

class InvalidRtfException(Exception):
    """
    handle invalid RTF
    """
    pass

class RtfInvalidCodeException(Exception):
    """
    handle bugs in program
    """
    pass

class ParseRtf:
    """
    Main class for controlling the rest of the parsing.
    """

    def __init__(self, in_file, out_file='', out_dir=None, dtd='', deb_dir=None, convert_symbol=None, convert_wingdings=None, convert_zapf=None, convert_caps=None, run_level=1, indent=None, replace_illegals=1, form_lists=1, headings_to_sections=1, group_styles=1, group_borders=1, empty_paragraphs=1, no_dtd=0, char_data='', default_encoding='cp1252'):
        if False:
            while True:
                i = 10
        "\n        Requires:\n        'file' --file to parse\n        'char_data' --file containing character maps\n        'dtd' --path to dtd\n        Possible parameters, but not necessary:\n            'output' --a file to output the parsed file. (Default is standard\n            output.)\n            'temp_dir' --directory for temporary output (If not provided, the\n            script tries to output to directory where is script is executed.)\n            'deb_dir' --debug directory. If a debug_dir is provided, the script\n            will copy each run through as a file to examine in the debug_dir\n            'check_brackets' -- make sure the brackets match up after each run\n            through a file. Only for debugging.\n        Returns: Nothing\n        "
        self.__file = in_file
        self.__out_file = out_file
        self.__out_dir = out_dir
        self.__temp_dir = out_dir
        self.__dtd_path = dtd
        self.__check_file(in_file, 'file_to_parse')
        self.__char_data = char_data
        self.__debug_dir = deb_dir
        self.__check_dir(self.__temp_dir)
        self.__copy = self.__check_dir(self.__debug_dir)
        self.__convert_caps = convert_caps
        self.__convert_symbol = convert_symbol
        self.__convert_wingdings = convert_wingdings
        self.__convert_zapf = convert_zapf
        self.__run_level = run_level
        self.__exit_level = 0
        self.__indent = indent
        self.__replace_illegals = replace_illegals
        self.__form_lists = form_lists
        self.__headings_to_sections = headings_to_sections
        self.__group_styles = group_styles
        self.__group_borders = group_borders
        self.__empty_paragraphs = empty_paragraphs
        self.__no_dtd = no_dtd
        self.__default_encoding = default_encoding

    def __check_file(self, the_file, type):
        if False:
            for i in range(10):
                print('nop')
        'Check to see if files exist'
        if hasattr(the_file, 'read'):
            return
        if the_file is None:
            if type == 'file_to_parse':
                msg = '\nYou must provide a file for the script to work'
            raise RtfInvalidCodeException(msg)
        elif os.path.exists(the_file):
            pass
        else:
            msg = "\nThe file '%s' cannot be found" % the_file
            raise RtfInvalidCodeException(msg)

    def __check_dir(self, the_dir):
        if False:
            return 10
        'Check to see if directory exists'
        if not the_dir:
            return
        dir_exists = os.path.isdir(the_dir)
        if not dir_exists:
            msg = '\n%s is not a directory' % the_dir
            raise RtfInvalidCodeException(msg)
        return 1

    def parse_rtf(self):
        if False:
            i = 10
            return i + 15
        "\n        Parse the file by calling on other classes.\n        Requires:\n            Nothing\n        Returns:\n            A parsed file in XML, either to standard output or to a file,\n            depending on the value of 'output' when the instance was created.\n        "
        self.__temp_file = self.__make_temp_file(self.__file)
        if self.__debug_dir:
            copy_obj = copy.Copy(bug_handler=RtfInvalidCodeException)
            copy_obj.set_dir(self.__debug_dir)
            copy_obj.remove_files()
            copy_obj.copy_file(self.__temp_file, 'original_file')
        if self.__debug_dir or self.__run_level > 2:
            self.__check_brack_obj = check_brackets.CheckBrackets(file=self.__temp_file, bug_handler=RtfInvalidCodeException)
        line_obj = line_endings.FixLineEndings(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level, replace_illegals=self.__replace_illegals)
        return_value = line_obj.fix_endings()
        self.__return_code(return_value)
        tokenize_obj = tokenize.Tokenize(bug_handler=RtfInvalidCodeException, in_file=self.__temp_file, copy=self.__copy, run_level=self.__run_level)
        tokenize_obj.tokenize()
        process_tokens_obj = process_tokens.ProcessTokens(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level, exception_handler=InvalidRtfException)
        try:
            return_value = process_tokens_obj.process_tokens()
        except InvalidRtfException as msg:
            encode_obj = default_encoding.DefaultEncoding(in_file=self.__temp_file, run_level=self.__run_level, bug_handler=RtfInvalidCodeException, check_raw=True, default_encoding=self.__default_encoding)
            (platform, code_page, default_font_num) = encode_obj.find_default_encoding()
            check_encoding_obj = check_encoding.CheckEncoding(bug_handler=RtfInvalidCodeException)
            enc = encode_obj.get_codepage()
            enc = 'cp' + enc
            msg = '%s\nException in token processing' % str(msg)
            if check_encoding_obj.check_encoding(self.__file, enc):
                file_name = self.__file if isinstance(self.__file, bytes) else self.__file.encode('utf-8')
                msg += '\nFile %s does not appear to be correctly encoded.\n' % file_name
            try:
                os.remove(self.__temp_file)
            except OSError:
                pass
            raise InvalidRtfException(msg)
        delete_info_obj = delete_info.DeleteInfo(in_file=self.__temp_file, copy=self.__copy, bug_handler=RtfInvalidCodeException, run_level=self.__run_level)
        found_destination = delete_info_obj.delete_info()
        self.__bracket_match('delete_data_info')
        pict_obj = pict.Pict(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, orig_file=self.__file, out_file=self.__out_file, run_level=self.__run_level)
        pict_obj.process_pict()
        self.__bracket_match('pict_data_info')
        combine_obj = combine_borders.CombineBorders(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
        combine_obj.combine_borders()
        self.__bracket_match('combine_borders_info')
        footnote_obj = footnote.Footnote(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
        footnote_obj.separate_footnotes()
        self.__bracket_match('separate_footnotes_info')
        header_obj = header.Header(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
        header_obj.separate_headers()
        self.__bracket_match('separate_headers_info')
        list_numbers_obj = list_numbers.ListNumbers(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
        list_numbers_obj.fix_list_numbers()
        self.__bracket_match('list_number_info')
        preamble_div_obj = preamble_div.PreambleDiv(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
        list_of_lists = preamble_div_obj.make_preamble_divisions()
        self.__bracket_match('make_preamble_divisions')
        encode_obj = default_encoding.DefaultEncoding(in_file=self.__temp_file, run_level=self.__run_level, bug_handler=RtfInvalidCodeException, default_encoding=self.__default_encoding)
        (platform, code_page, default_font_num) = encode_obj.find_default_encoding()
        hex2utf_obj = hex_2_utf8.Hex2Utf8(in_file=self.__temp_file, copy=self.__copy, area_to_convert='preamble', char_file=self.__char_data, default_char_map=code_page, run_level=self.__run_level, bug_handler=RtfInvalidCodeException, invalid_rtf_handler=InvalidRtfException)
        hex2utf_obj.convert_hex_2_utf8()
        self.__bracket_match('hex_2_utf_preamble')
        fonts_obj = fonts.Fonts(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, default_font_num=default_font_num, run_level=self.__run_level)
        special_font_dict = fonts_obj.convert_fonts()
        self.__bracket_match('fonts_info')
        color_obj = colors.Colors(in_file=self.__temp_file, copy=self.__copy, bug_handler=RtfInvalidCodeException, run_level=self.__run_level)
        color_obj.convert_colors()
        self.__bracket_match('colors_info')
        style_obj = styles.Styles(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
        style_obj.convert_styles()
        self.__bracket_match('styles_info')
        info_obj = info.Info(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
        info_obj.fix_info()
        default_font = special_font_dict.get('default-font')
        preamble_rest_obj = preamble_rest.Preamble(file=self.__temp_file, copy=self.__copy, bug_handler=RtfInvalidCodeException, platform=platform, default_font=default_font, code_page=code_page)
        preamble_rest_obj.fix_preamble()
        self.__bracket_match('preamble_rest_info')
        old_rtf_obj = OldRtf(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, run_level=self.__run_level)
        old_rtf = old_rtf_obj.check_if_old_rtf()
        if old_rtf:
            if self.__run_level > 5:
                msg = 'Older RTF\nself.__run_level is "%s"\n' % self.__run_level
                raise RtfInvalidCodeException(msg)
            if self.__run_level > 1:
                sys.stderr.write('File could be older RTF...\n')
            if found_destination:
                if self.__run_level > 1:
                    sys.stderr.write('File also has newer RTF.\nWill do the best to convert...\n')
            add_brackets_obj = add_brackets.AddBrackets(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
            add_brackets_obj.add_brackets()
        fields_small_obj = fields_small.FieldsSmall(in_file=self.__temp_file, copy=self.__copy, bug_handler=RtfInvalidCodeException, run_level=self.__run_level)
        fields_small_obj.fix_fields()
        self.__bracket_match('fix_small_fields_info')
        fields_large_obj = fields_large.FieldsLarge(in_file=self.__temp_file, copy=self.__copy, bug_handler=RtfInvalidCodeException, run_level=self.__run_level)
        fields_large_obj.fix_fields()
        self.__bracket_match('fix_large_fields_info')
        sections_obj = sections.Sections(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
        sections_obj.make_sections()
        self.__bracket_match('sections_info')
        paragraphs_obj = paragraphs.Paragraphs(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, write_empty_para=self.__empty_paragraphs, run_level=self.__run_level)
        paragraphs_obj.make_paragraphs()
        self.__bracket_match('paragraphs_info')
        default_font = special_font_dict['default-font']
        paragraph_def_obj = paragraph_def.ParagraphDef(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, default_font=default_font, run_level=self.__run_level)
        list_of_styles = paragraph_def_obj.make_paragraph_def()
        body_styles_obj = body_styles.BodyStyles(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, list_of_styles=list_of_styles, run_level=self.__run_level)
        body_styles_obj.insert_info()
        self.__bracket_match('body_styles_info')
        self.__bracket_match('paragraph_def_info')
        table_obj = table.Table(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
        table_data = table_obj.make_table()
        self.__bracket_match('table_info')
        table_info_obj = table_info.TableInfo(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, table_data=table_data, run_level=self.__run_level)
        table_info_obj.insert_info()
        self.__bracket_match('table__data_info')
        if self.__form_lists:
            make_list_obj = make_lists.MakeLists(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, headings_to_sections=self.__headings_to_sections, run_level=self.__run_level, list_of_lists=list_of_lists)
            make_list_obj.make_lists()
            self.__bracket_match('form_lists_info')
        if self.__headings_to_sections:
            headings_to_sections_obj = headings_to_sections.HeadingsToSections(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
            headings_to_sections_obj.make_sections()
            self.__bracket_match('headings_to_sections_info')
        if self.__group_styles:
            group_styles_obj = group_styles.GroupStyles(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, wrap=1, run_level=self.__run_level)
            group_styles_obj.group_styles()
            self.__bracket_match('group_styles_info')
        if self.__group_borders:
            group_borders_obj = group_borders.GroupBorders(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, wrap=1, run_level=self.__run_level)
            group_borders_obj.group_borders()
            self.__bracket_match('group_borders_info')
        inline_obj = inline.Inline(in_file=self.__temp_file, bug_handler=RtfInvalidCodeException, copy=self.__copy, run_level=self.__run_level)
        inline_obj.form_tags()
        self.__bracket_match('inline_info')
        hex2utf_obj.update_values(file=self.__temp_file, area_to_convert='body', copy=self.__copy, char_file=self.__char_data, convert_caps=self.__convert_caps, convert_symbol=self.__convert_symbol, convert_wingdings=self.__convert_wingdings, convert_zapf=self.__convert_zapf, symbol=1, wingdings=1, dingbats=1)
        hex2utf_obj.convert_hex_2_utf8()
        header_obj.join_headers()
        footnote_obj.join_footnotes()
        tags_obj = convert_to_tags.ConvertToTags(in_file=self.__temp_file, copy=self.__copy, dtd_path=self.__dtd_path, indent=self.__indent, run_level=self.__run_level, no_dtd=self.__no_dtd, encoding=encode_obj.get_codepage(), bug_handler=RtfInvalidCodeException)
        tags_obj.convert_to_tags()
        output_obj = output.Output(file=self.__temp_file, orig_file=self.__file, output_dir=self.__out_dir, out_file=self.__out_file)
        output_obj.output()
        os.remove(self.__temp_file)
        return self.__exit_level

    def __bracket_match(self, file_name):
        if False:
            print('Hello World!')
        if self.__run_level > 2:
            (good_br, msg) = self.__check_brack_obj.check_brackets()
            if good_br:
                pass
            else:
                msg = f'{msg} in file {file_name}'
                print(msg, file=sys.stderr)

    def __return_code(self, num):
        if False:
            while True:
                i = 10
        if num is None:
            return
        if int(num) > self.__exit_level:
            self.__exit_level = num

    def __make_temp_file(self, file):
        if False:
            i = 10
            return i + 15
        'Make a temporary file to parse'
        write_file = 'rtf_write_file'
        read_obj = file if hasattr(file, 'read') else open_for_read(file)
        with open_for_write(write_file) as write_obj:
            for line in read_obj:
                write_obj.write(line)
        return write_file