"""
 ██████╗██╗██████╗ ██╗  ██╗███████╗██╗   ██╗
██╔════╝██║██╔══██╗██║  ██║██╔════╝╚██╗ ██╔╝
██║     ██║██████╔╝███████║█████╗   ╚████╔╝
██║     ██║██╔═══╝ ██╔══██║██╔══╝    ╚██╔╝
╚██████╗██║██║     ██║  ██║███████╗   ██║
© Brandon Skerritt
Github: brandonskerritt

Class to provide helper functions for mathematics
(oh, not entirely mathematics either. Some NLP stuff and sorting dicts. It's just a helper class
)
"""
from collections import OrderedDict
from string import punctuation
from typing import Optional
import logging
from rich.logging import RichHandler

class mathsHelper:
    """Class to provide helper functions for mathematics and other small things"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.ETAOIN = 'ETAOINSHRDLCUMWFGYPBVKJXQZ'
        self.LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    @staticmethod
    def gcd(a, b) -> int:
        if False:
            while True:
                i = 10
        "Greatest common divisor.\n\n        The Greatest Common Divisor of a and b using Euclid's Algorithm.\n\n        Args:\n            a -> num 1\n            b -> num 2\n\n        Returns:\n            Returns  GCD(a, b)\n\n        "
        while a != 0:
            (a, b) = (b % a, a)
        return b

    @staticmethod
    def mod_inv(a: int, m: int) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the modular inverse of a mod m, or None if it does not exist.\n\n        The modular inverse of a is the number a_inv that satisfies the equation\n        a_inv * a mod m === 1 mod m\n\n        Note: This is a naive implementation, and runtime may be improved in several ways.\n        For instance by checking if m is prime to perform a different calculation,\n        or by using the extended euclidean algorithm.\n        '
        for i in range(1, m):
            if (m * i + 1) % a == 0:
                return (m * i + 1) // a
        return None

    @staticmethod
    def percentage(part: float, whole: float) -> float:
        if False:
            return 10
        'Returns percentage.\n\n        Just a normal algorithm to return the percent.\n\n        Args:\n            part -> part of the whole number\n            whole -> the whole number\n\n        Returns:\n            Returns the percentage of part to whole.\n\n        '
        if part <= 0 or whole <= 0:
            return 0
        return 100 * float(part) / float(whole)

    def sort_prob_table(self, prob_table: dict) -> dict:
        if False:
            print('Hello World!')
        'Sorts the probability table.\n\n        Sorts a dictionary of dictionaries (and all the sub-dictionaries).\n\n        Args:\n            prob_table -> The probability table returned by the neural network to sort.\n\n        Returns:\n            Returns the prob_table, but sorted.\n\n        '
        max_overall: int = 0
        max_dict_pair: dict = {}
        highest_key = None
        empty_dict: dict = {}
        for (key, value) in prob_table.items():
            prob_table[key] = self.new_sort(value)
            prob_table[key] = dict(prob_table[key])
        counter_max: int = 0
        counter_prob: int = len(prob_table)
        while counter_max < counter_prob:
            max_overall = 0
            highest_key = None
            logging.debug(f'Running while loop in sort_prob_table, counterMax is {counter_max}')
            for (key, value) in prob_table.items():
                logging.debug(f'Sorting {key}')
                maxLocal = 0
                for (key2, value2) in value.items():
                    logging.debug(f'Running key2 {key2}, value2 {value2} for loop for {value.items()}')
                    maxLocal = maxLocal + value2
                    logging.debug(f'MaxLocal is {maxLocal} and maxOverall is {max_overall}')
                    if maxLocal > max_overall:
                        logging.debug(f'New max local found {maxLocal}')
                        max_dict_pair = {}
                        max_overall = maxLocal
                        max_dict_pair[key] = value
                        highest_key = key
                        logging.debug(f'Highest key is {highest_key}')
            logging.debug(f'Prob table is {prob_table} and highest key is {highest_key}')
            logging.debug(f'Removing {prob_table[highest_key]}')
            del prob_table[highest_key]
            logging.debug(f'Prob table after deletion is {prob_table}')
            counter_max += 1
            empty_dict = {**empty_dict, **max_dict_pair}
        logging.debug(f'The prob table is {prob_table} and the maxDictPair is {max_dict_pair}')
        logging.debug(f'The new sorted prob table is {empty_dict}')
        return empty_dict

    @staticmethod
    def new_sort(new_dict: dict) -> dict:
        if False:
            return 10
        "Uses OrderedDict to sort a dictionary.\n\n        I think it's faster than my implementation.\n\n        Args:\n            new_dict -> the dictionary to sort\n\n        Returns:\n            Returns the dict, but sorted.\n\n        "
        logging.debug(f'The old dictionary before new_sort() is {new_dict}')
        sorted_i = OrderedDict(sorted(new_dict.items(), key=lambda x: x[1], reverse=True))
        logging.debug(f'The dictionary after new_sort() is {sorted_i}')
        return sorted_i

    @staticmethod
    def is_ascii(s: str) -> bool:
        if False:
            print('Hello World!')
        'Returns the boolean value if is_ascii is an ascii char.\n\n        Does what it says on the tree. Stolen from\n        https://stackoverflow.com/questions/196345/how-to-check-if-a-string-in-python-is-in-ascii\n\n        Args:\n            s -> the char to check.\n\n        Returns:\n            Returns the boolean of the char.\n\n        '
        return bool(lambda s: len(s) == len(s.encode()))

    @staticmethod
    def strip_punctuation(text: str) -> str:
        if False:
            print('Hello World!')
        'Strips punctuation from a given string.\n\n        Uses string.punctuation.\n\n        Args:\n            text -> the text to strip punctuation from.\n\n        Returns:\n            Returns string without punctuation.\n        '
        text: str = str(text).translate(str.maketrans('', '', punctuation)).strip('\n')
        return text