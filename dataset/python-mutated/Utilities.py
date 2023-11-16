"""Generic functions which are useful for working with HMMs.

This just collects general functions which you might like to use in
dealing with HMMs.
"""

def pretty_print_prediction(emissions, real_state, predicted_state, emission_title='Emissions', real_title='Real State', predicted_title='Predicted State', line_width=75):
    if False:
        return 10
    'Print out a state sequence prediction in a nice manner.\n\n    Arguments:\n     - emissions -- The sequence of emissions of the sequence you are\n       dealing with.\n     - real_state -- The actual state path that generated the emissions.\n     - predicted_state -- A state path predicted by some kind of HMM model.\n\n    '
    title_length = max(len(emission_title), len(real_title), len(predicted_title)) + 1
    seq_length = line_width - title_length
    emission_title = emission_title.ljust(title_length)
    real_title = real_title.ljust(title_length)
    predicted_title = predicted_title.ljust(title_length)
    cur_position = 0
    while True:
        if cur_position + seq_length < len(emissions):
            extension = seq_length
        else:
            extension = len(emissions) - cur_position
        print(f'{emission_title}{emissions[cur_position:cur_position + seq_length]}')
        print(f'{real_title}{real_state[cur_position:cur_position + seq_length]}')
        print('%s%s\n' % (predicted_title, predicted_state[cur_position:cur_position + seq_length]))
        if len(emissions) < cur_position + seq_length:
            break
        cur_position += seq_length