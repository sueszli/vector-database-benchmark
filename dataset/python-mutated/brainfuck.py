import re
import time
from typing import Dict, Optional, Tuple
import logging
from rich.logging import RichHandler
from ciphey.iface import Config, Decoder, ParamSpec, T, U, WordList, registry

@registry.register
class Brainfuck(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            return 10
        '\n        Takes a ciphertext and treats it as a Brainfuck program,\n        interpreting it and saving the output as a string to return.\n\n        Brainfuck is a very simple, Turing-complete esoteric language.\n        Below is a simplified interpreter that attempts to check whether a\n        given ciphertext is a brainfuck program that would output a string.\n\n        A program that can be "decoded" like this is one that:\n            * Does not require user input ("," instruction)\n            * Includes at least one putchar instruction (".")\n            * Does not contain anything but the main 7 instructions,\n                (excluding ",") and whitespace\n\n        Details:\n            * This implementation wraps the memory pointer for ">" and "<"\n            * It is time-limited to 60 seconds, to prevent hangups\n            * The program starts with 100 memory cells, chosen arbitrarily\n        '
        logging.debug('Attempting brainfuck')
        result = ''
        memory = [0] * 100
        (codeptr, memptr) = (0, 0)
        timelimit = 60
        (bracemap, isbf) = self.bracemap_and_check(ctext)
        if not isbf:
            logging.debug('Failed to interpret brainfuck due to invalid characters')
            return None
        start = time.time()
        while codeptr < len(ctext):
            current = time.time()
            if current - start > timelimit:
                logging.debug('Failed to interpret brainfuck due to timing out')
                return None
            cmd = ctext[codeptr]
            if cmd == '+':
                if memory[memptr] < 255:
                    memory[memptr] = memory[memptr] + 1
                else:
                    memory[memptr] = 0
            elif cmd == '-':
                if memory[memptr] > 0:
                    memory[memptr] = memory[memptr] - 1
                else:
                    memory[memptr] = 255
            elif cmd == '>':
                if memptr == len(memory) - 1:
                    memory.append(0)
                memptr += 1
            elif cmd == '<':
                if memptr == 0:
                    memptr = len(memory) - 1
                else:
                    memptr -= 1
            elif cmd == '[' and memory[memptr] == 0:
                codeptr = bracemap[codeptr]
            elif cmd == ']' and memory[memptr]:
                codeptr = bracemap[codeptr]
            elif cmd == '.':
                result += chr(memory[memptr])
            codeptr += 1
        logging.info(f"Brainfuck successful, returning '{result}'")
        return result

    def bracemap_and_check(self, program: str) -> Tuple[Optional[Dict], bool]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Create a bracemap of brackets in the program, to compute jmps.\n        Maps open -> close brackets as well as close -> open brackets.\n\n        Also returns True if the program is valid Brainfuck code. If False, we\n        won't even try to run it.\n        "
        open_stack = []
        bracemap = dict()
        legal_instructions = {'+', '-', '>', '<', '[', ']', '.'}
        legal_count = 0
        prints = False
        for (idx, instruction) in enumerate(program):
            if instruction in legal_instructions or re.match('\\s', instruction):
                legal_count += 1
            if not prints and instruction == '.':
                prints = True
            elif instruction == '[':
                open_stack.append(idx)
            elif instruction == ']':
                try:
                    opbracket = open_stack.pop()
                    bracemap[opbracket] = idx
                    bracemap[idx] = opbracket
                except IndexError:
                    return (None, False)
        is_brainfuck = legal_count == len(program) and len(open_stack) == 0 and prints
        return (bracemap, is_brainfuck)

    @staticmethod
    def priority() -> float:
        if False:
            return 10
        return 0.08

    def __init__(self, config: Config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.ALPHABET = config.get_resource(self._params()['dict'], WordList)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            i = 10
            return i + 15
        return {'dict': ParamSpec(desc='Brainfuck alphabet (default English)', req=False, default='cipheydists::list::englishAlphabet')}

    @staticmethod
    def getTarget() -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'brainfuck'