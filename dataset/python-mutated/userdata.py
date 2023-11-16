from troposphere import Base64, Join, Sub

def from_file(filepath, delimiter='', blanklines=False):
    if False:
        i = 10
        return i + 15
    'Imports userdata from a file.\n\n    :type filepath: string\n\n    :param filepath  The absolute path to the file.\n\n    :type delimiter: string\n\n    :param: delimiter  Delimiter to use with the troposphere.Join().\n\n    :type blanklines: boolean\n\n    :param blanklines  If blank lines should be ignored\n\n    rtype: troposphere.Base64\n    :return The base64 representation of the file.\n    '
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if blanklines and line.strip('\n\r ') == '':
                    continue
                data.append(line)
    except IOError:
        raise IOError('Error opening or reading file: {}'.format(filepath))
    return Base64(Join(delimiter, data))

def from_file_sub(filepath):
    if False:
        print('Hello World!')
    'Imports userdata from a file, using Sub for replacing inline variables such as ${AWS::Region}\n\n    :type filepath: string\n\n    :param filepath  The absolute path to the file.\n\n    rtype: troposphere.Base64\n    :return The base64 representation of the file.\n    '
    try:
        with open(filepath, 'rt') as f:
            data = f.read()
            return Base64(Sub(data))
    except IOError:
        raise IOError('Error opening or reading file: {}'.format(filepath))