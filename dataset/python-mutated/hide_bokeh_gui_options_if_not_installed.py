def hide_bokeh_gui_options_if_not_installed(options_blk):
    if False:
        for i in range(10):
            print('nop')
    try:
        import bokehgui
    except ImportError:
        for param in options_blk.parameters_data:
            if param['id'] == 'generate_options':
                ind = param['options'].index('bokeh_gui')
                del param['options'][ind]
                del param['option_labels'][ind]