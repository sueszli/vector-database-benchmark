"""
Run-length encoding (RLE) is a simple compression algorithm 
that gets a stream of data as the input and returns a
sequence of counts of consecutive data values in a row.
When decompressed the data will be fully recovered as RLE
is a lossless data compression.
"""

def encode_rle(input):
    if False:
        while True:
            i = 10
    '\n    Gets a stream of data and compresses it\n    under a Run-Length Encoding.\n    :param input: The data to be encoded.\n    :return: The encoded string.\n    '
    if not input:
        return ''
    encoded_str = ''
    prev_ch = ''
    count = 1
    for ch in input:
        if ch != prev_ch:
            if prev_ch:
                encoded_str += str(count) + prev_ch
            count = 1
            prev_ch = ch
        else:
            count += 1
    else:
        return encoded_str + (str(count) + prev_ch)

def decode_rle(input):
    if False:
        return 10
    '\n    Gets a stream of data and decompresses it\n    under a Run-Length Decoding.\n    :param input: The data to be decoded.\n    :return: The decoded string.\n    '
    decode_str = ''
    count = ''
    for ch in input:
        if not ch.isdigit():
            decode_str += ch * int(count)
            count = ''
        else:
            count += ch
    return decode_str