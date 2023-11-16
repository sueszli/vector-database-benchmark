import re

def ptb_detokenizer(string):
    if False:
        i = 10
        return i + 15
    string = string.replace(" '", "'")
    string = string.replace(' \n', '\n')
    string = string.replace('\n ', '\n')
    string = string.replace(" n't", "n't")
    string = string.replace(' N ', '1 ')
    string = string.replace('$ 1', '$1')
    string = string.replace('# 1', '#1')
    return string

def wikitext_detokenizer(string):
    if False:
        for i in range(10):
            print('nop')
    string = string.replace("s '", "s'")
    string = re.sub("/' [0-9]/", "/'[0-9]/", string)
    string = string.replace(' @-@ ', '-')
    string = string.replace(' @,@ ', ',')
    string = string.replace(' @.@ ', '.')
    string = string.replace(' : ', ': ')
    string = string.replace(' ; ', '; ')
    string = string.replace(' . ', '. ')
    string = string.replace(' ! ', '! ')
    string = string.replace(' ? ', '? ')
    string = string.replace(' , ', ', ')
    string = re.sub('\\(\\s*([^\\)]*?)\\s*\\)', '(\\1)', string)
    string = re.sub('\\[\\s*([^\\]]*?)\\s*\\]', '[\\1]', string)
    string = re.sub('{\\s*([^}]*?)\\s*}', '{\\1}', string)
    string = re.sub('\\"\\s*([^\\"]*?)\\s*\\"', '"\\1"', string)
    string = re.sub("'\\s*([^']*?)\\s*'", "'\\1'", string)
    string = string.replace('= = = =', '====')
    string = string.replace('= = =', '===')
    string = string.replace('= =', '==')
    string = string.replace(' ' + chr(176) + ' ', chr(176))
    string = string.replace(' \n', '\n')
    string = string.replace('\n ', '\n')
    string = string.replace(' N ', ' 1 ')
    string = string.replace(" 's", "'s")
    return string

def lambada_detokenizer(string):
    if False:
        return 10
    return string

def get_detokenizer(dataset):
    if False:
        print('Hello World!')
    return DETOKENIZERS[dataset]
DETOKENIZERS = {'ptb': ptb_detokenizer, 'wikitext': wikitext_detokenizer, 'lambada': lambada_detokenizer}