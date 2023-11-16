def truncate_output(data, max_output_chars=2000):
    if False:
        i = 10
        return i + 15
    needs_truncation = False
    message = f'Output truncated. Showing the last {max_output_chars} characters.\n\n'
    if data.startswith(message):
        data = data[len(message):]
        needs_truncation = True
    if len(data) > max_output_chars or needs_truncation:
        data = message + data[-max_output_chars:]
    return data