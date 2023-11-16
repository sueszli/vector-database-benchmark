import random
import cipheydists

class galactic_encode:
    """
    (Attempts to) encode an input string with the Standard Galactic Alphabet.
    """

    def __init__(self, text: str):
        if False:
            for i in range(10):
                print('nop')
        self.text = text.lower()
        self.ctext = ''
        imported = dict(cipheydists.get_translate('galactic'))
        self.galactic_dict = {value: key for (key, value) in imported.items()}

    def encode(self):
        if False:
            return 10
        for char in self.text:
            if char in self.galactic_dict.keys():
                self.ctext += self.galactic_dict[char]
            else:
                self.ctext += char
        return self.ctext

class atbash_encode:
    """
    Encodes an input string with the Atbash cipher.
    """

    def __init__(self, text: str):
        if False:
            i = 10
            return i + 15
        self.text = text.lower()
        self.letters = list('abcdefghijklmnopqrstuvwxyz')
        self.atbash_dict = {self.letters[::-1][i]: self.letters[i] for i in range(26)}
        self.ctext = ''

    def encode(self):
        if False:
            return 10
        for letter in self.text:
            if letter in self.atbash_dict.keys():
                self.ctext += self.atbash_dict[letter]
            else:
                self.ctext += letter
        return self.ctext

class XY_encrypt:
    """
    Encrypts an input string using binary substitution (called XandY in Ciphey) in which
    first, the input string is converted to its binary representation and then the 0s and 1s
    of the binary string are replaced with any two characters.
    - flip: Which of the two possible rotations of the substitute characters is used?
    - randomize: If True, random spaces are inserted into the cstring, which Ciphey can handle.
    - key: Which two characters are used to represent the 0s and 1s?
    """

    def __init__(self, text: str, flip: bool=bool(random.randint(0, 1)), randomize: bool=True, key: list=None):
        if False:
            print('Hello World!')
        self.ASCII = cipheydists.get_charset('asciiTable')
        self.text = text.lower()
        self.ctext = ''
        self.flip = flip
        self.randomize = randomize
        self.key = key

    def randomizer(self):
        if False:
            for i in range(10):
                print('nop')
        s = list(self.ctext)
        for i in range(len(s) - 1):
            while random.randrange(2):
                s[i] = s[i] + ' '
        return ''.join(s)

    def to_binary(self):
        if False:
            return 10
        return ' '.join((f'{ord(i):08b}' for i in self.text))

    def encrypt(self):
        if False:
            for i in range(10):
                print('nop')
        self.ctext = self.to_binary().replace(' ', '')
        if self.key:
            (one, two) = (self.key[0], self.key[1])
        else:
            (one, two) = (random.choice(self.ASCII), random.choice(self.ASCII))
        self.ctext = self.ctext.replace(str(int(self.flip)), one).replace(str(int(not self.flip)), two)
        self.ctext = self.randomizer() if self.randomize is True else self.ctext
        return self.ctext