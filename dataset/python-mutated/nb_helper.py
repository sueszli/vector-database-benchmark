def get_output_stream_from_cell(cell, stream_name='stdout'):
    if False:
        return 10
    return '\n'.join([output.text for output in cell.get('outputs', []) if output.output_type == 'stream' and output.name == stream_name])