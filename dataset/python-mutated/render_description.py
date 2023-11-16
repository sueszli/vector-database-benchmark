def render_description(description, format):
    if False:
        print('Hello World!')
    if format is None:
        return None
    if format == 'markdown':
        import markdown
        return markdown.markdown(description)
    raise RuntimeError(f'Unsupported description format {format}')