from typing import Optional, Sequence
from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def process_document_ocr_sample(project_id: str, location: str, processor_id: str, processor_version: str, file_path: str, mime_type: str) -> None:
    if False:
        i = 10
        return i + 15
    process_options = documentai.ProcessOptions(ocr_config=documentai.OcrConfig(enable_native_pdf_parsing=True, enable_image_quality_scores=True, enable_symbol=True, premium_features=documentai.OcrConfig.PremiumFeatures(compute_style_info=True, enable_math_ocr=False, enable_selection_mark_detection=True)))
    document = process_document(project_id, location, processor_id, processor_version, file_path, mime_type, process_options=process_options)
    text = document.text
    print(f'Full document text: {text}\n')
    print(f'There are {len(document.pages)} page(s) in this document.\n')
    for page in document.pages:
        print(f'Page {page.page_number}:')
        print_page_dimensions(page.dimension)
        print_detected_langauges(page.detected_languages)
        print_blocks(page.blocks, text)
        print_paragraphs(page.paragraphs, text)
        print_lines(page.lines, text)
        print_tokens(page.tokens, text)
        if page.symbols:
            print_symbols(page.symbols, text)
        if page.image_quality_scores:
            print_image_quality_scores(page.image_quality_scores)
        if page.visual_elements:
            print_visual_elements(page.visual_elements, text)

def print_page_dimensions(dimension: documentai.Document.Page.Dimension) -> None:
    if False:
        while True:
            i = 10
    print(f'    Width: {str(dimension.width)}')
    print(f'    Height: {str(dimension.height)}')

def print_detected_langauges(detected_languages: Sequence[documentai.Document.Page.DetectedLanguage]) -> None:
    if False:
        while True:
            i = 10
    print('    Detected languages:')
    for lang in detected_languages:
        print(f'        {lang.language_code} ({lang.confidence:.1%} confidence)')

def print_blocks(blocks: Sequence[documentai.Document.Page.Block], text: str) -> None:
    if False:
        i = 10
        return i + 15
    print(f'    {len(blocks)} blocks detected:')
    first_block_text = layout_to_text(blocks[0].layout, text)
    print(f'        First text block: {repr(first_block_text)}')
    last_block_text = layout_to_text(blocks[-1].layout, text)
    print(f'        Last text block: {repr(last_block_text)}')

def print_paragraphs(paragraphs: Sequence[documentai.Document.Page.Paragraph], text: str) -> None:
    if False:
        print('Hello World!')
    print(f'    {len(paragraphs)} paragraphs detected:')
    first_paragraph_text = layout_to_text(paragraphs[0].layout, text)
    print(f'        First paragraph text: {repr(first_paragraph_text)}')
    last_paragraph_text = layout_to_text(paragraphs[-1].layout, text)
    print(f'        Last paragraph text: {repr(last_paragraph_text)}')

def print_lines(lines: Sequence[documentai.Document.Page.Line], text: str) -> None:
    if False:
        i = 10
        return i + 15
    print(f'    {len(lines)} lines detected:')
    first_line_text = layout_to_text(lines[0].layout, text)
    print(f'        First line text: {repr(first_line_text)}')
    last_line_text = layout_to_text(lines[-1].layout, text)
    print(f'        Last line text: {repr(last_line_text)}')

def print_tokens(tokens: Sequence[documentai.Document.Page.Token], text: str) -> None:
    if False:
        i = 10
        return i + 15
    print(f'    {len(tokens)} tokens detected:')
    first_token_text = layout_to_text(tokens[0].layout, text)
    first_token_break_type = tokens[0].detected_break.type_.name
    print(f'        First token text: {repr(first_token_text)}')
    print(f'        First token break type: {repr(first_token_break_type)}')
    if tokens[0].style_info:
        print_style_info(tokens[0].style_info)
    last_token_text = layout_to_text(tokens[-1].layout, text)
    last_token_break_type = tokens[-1].detected_break.type_.name
    print(f'        Last token text: {repr(last_token_text)}')
    print(f'        Last token break type: {repr(last_token_break_type)}')
    if tokens[-1].style_info:
        print_style_info(tokens[-1].style_info)

def print_symbols(symbols: Sequence[documentai.Document.Page.Symbol], text: str) -> None:
    if False:
        return 10
    print(f'    {len(symbols)} symbols detected:')
    first_symbol_text = layout_to_text(symbols[0].layout, text)
    print(f'        First symbol text: {repr(first_symbol_text)}')
    last_symbol_text = layout_to_text(symbols[-1].layout, text)
    print(f'        Last symbol text: {repr(last_symbol_text)}')

def print_image_quality_scores(image_quality_scores: documentai.Document.Page.ImageQualityScores) -> None:
    if False:
        for i in range(10):
            print('nop')
    print(f'    Quality score: {image_quality_scores.quality_score:.1%}')
    print('    Detected defects:')
    for detected_defect in image_quality_scores.detected_defects:
        print(f'        {detected_defect.type_}: {detected_defect.confidence:.1%}')

def print_style_info(style_info: documentai.Document.Page.Token.StyleInfo) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Only supported in version `pretrained-ocr-v2.0-2023-06-02`\n    '
    print(f'           Font Size: {style_info.font_size}pt')
    print(f'           Font Type: {style_info.font_type}')
    print(f'           Bold: {style_info.bold}')
    print(f'           Italic: {style_info.italic}')
    print(f'           Underlined: {style_info.underlined}')
    print(f'           Handwritten: {style_info.handwritten}')
    print(f'           Text Color (RGBa): {style_info.text_color.red}, {style_info.text_color.green}, {style_info.text_color.blue}, {style_info.text_color.alpha}')

def print_visual_elements(visual_elements: Sequence[documentai.Document.Page.VisualElement], text: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Only supported in version `pretrained-ocr-v2.0-2023-06-02`\n    '
    checkboxes = [x for x in visual_elements if 'checkbox' in x.type]
    math_symbols = [x for x in visual_elements if x.type == 'math_formula']
    if checkboxes:
        print(f'    {len(checkboxes)} checkboxes detected:')
        print(f'        First checkbox: {repr(checkboxes[0].type)}')
        print(f'        Last checkbox: {repr(checkboxes[-1].type)}')
    if math_symbols:
        print(f'    {len(math_symbols)} math symbols detected:')
        first_math_symbol_text = layout_to_text(math_symbols[0].layout, text)
        print(f'        First math symbol: {repr(first_math_symbol_text)}')

def process_document_form_sample(project_id: str, location: str, processor_id: str, processor_version: str, file_path: str, mime_type: str) -> documentai.Document:
    if False:
        while True:
            i = 10
    document = process_document(project_id, location, processor_id, processor_version, file_path, mime_type)
    text = document.text
    print(f'Full document text: {repr(text)}\n')
    print(f'There are {len(document.pages)} page(s) in this document.')
    for page in document.pages:
        print(f'\n\n**** Page {page.page_number} ****')
        print(f'\nFound {len(page.tables)} table(s):')
        for table in page.tables:
            num_columns = len(table.header_rows[0].cells)
            num_rows = len(table.body_rows)
            print(f'Table with {num_columns} columns and {num_rows} rows:')
            print('Columns:')
            print_table_rows(table.header_rows, text)
            print('Table body data:')
            print_table_rows(table.body_rows, text)
        print(f'\nFound {len(page.form_fields)} form field(s):')
        for field in page.form_fields:
            name = layout_to_text(field.field_name, text)
            value = layout_to_text(field.field_value, text)
            print(f'    * {repr(name.strip())}: {repr(value.strip())}')
    if document.entities:
        print(f'Found {len(document.entities)} generic entities:')
        for entity in document.entities:
            print_entity(entity)
            for prop in entity.properties:
                print_entity(prop)
    return document

def print_table_rows(table_rows: Sequence[documentai.Document.Page.Table.TableRow], text: str) -> None:
    if False:
        print('Hello World!')
    for table_row in table_rows:
        row_text = ''
        for cell in table_row.cells:
            cell_text = layout_to_text(cell.layout, text)
            row_text += f'{repr(cell_text.strip())} | '
        print(row_text)

def process_document_entity_extraction_sample(project_id: str, location: str, processor_id: str, processor_version: str, file_path: str, mime_type: str) -> None:
    if False:
        while True:
            i = 10
    document = process_document(project_id, location, processor_id, processor_version, file_path, mime_type)
    print(f'Found {len(document.entities)} entities:')
    for entity in document.entities:
        print_entity(entity)
        for prop in entity.properties:
            print_entity(prop)

def print_entity(entity: documentai.Document.Entity) -> None:
    if False:
        for i in range(10):
            print('nop')
    key = entity.type_
    text_value = entity.text_anchor.content
    confidence = entity.confidence
    normalized_value = entity.normalized_value.text
    print(f'    * {repr(key)}: {repr(text_value)}({confidence:.1%} confident)')
    if normalized_value:
        print(f'    * Normalized Value: {repr(normalized_value)}')

def process_document_splitter_sample(project_id: str, location: str, processor_id: str, processor_version: str, file_path: str, mime_type: str) -> None:
    if False:
        print('Hello World!')
    document = process_document(project_id, location, processor_id, processor_version, file_path, mime_type)
    print(f'Found {len(document.entities)} subdocuments:')
    for entity in document.entities:
        conf_percent = f'{entity.confidence:.1%}'
        pages_range = page_refs_to_string(entity.page_anchor.page_refs)
        if entity.type_:
            print(f"{conf_percent} confident that {pages_range} a '{entity.type_}' subdocument.")
        else:
            print(f'{conf_percent} confident that {pages_range} a subdocument.')

def page_refs_to_string(page_refs: Sequence[documentai.Document.PageAnchor.PageRef]) -> str:
    if False:
        return 10
    'Converts a page ref to a string describing the page or page range.'
    pages = [str(int(page_ref.page) + 1) for page_ref in page_refs]
    if len(pages) == 1:
        return f'page {pages[0]} is'
    else:
        return f"pages {', '.join(pages)} are"

def process_document(project_id: str, location: str, processor_id: str, processor_version: str, file_path: str, mime_type: str, process_options: Optional[documentai.ProcessOptions]=None) -> documentai.Document:
    if False:
        for i in range(10):
            print('nop')
    client = documentai.DocumentProcessorServiceClient(client_options=ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com'))
    name = client.processor_version_path(project_id, location, processor_id, processor_version)
    with open(file_path, 'rb') as image:
        image_content = image.read()
    request = documentai.ProcessRequest(name=name, raw_document=documentai.RawDocument(content=image_content, mime_type=mime_type), process_options=process_options)
    result = client.process_document(request=request)
    return result.document

def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    if False:
        print('Hello World!')
    '\n    Document AI identifies text in different parts of the document by their\n    offsets in the entirety of the document"s text. This function converts\n    offsets to a string.\n    '
    return ''.join((text[int(segment.start_index):int(segment.end_index)] for segment in layout.text_anchor.text_segments))