"""
It stores all maximal matches from the given matches obtained by the template
matching algorithm.
"""

class Match:
    """
    Class Match is an object to store a list of match with its qubits and
    clbits configuration.
    """

    def __init__(self, match, qubit, clbit):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a Match with necessary arguments.\n        Args:\n            match (list): list of a match.\n            qubit (list): list of qubits configuration.\n            clbit (list): list of clbits configuration.\n        '
        self.match = match
        self.qubit = qubit
        self.clbit = clbit

class MaximalMatches:
    """
    Class MaximalMatches allows to sort and store the maximal matches from the list
    of matches obtained with the template matching algorithm.
    """

    def __init__(self, template_matches):
        if False:
            i = 10
            return i + 15
        '\n        Initialize MaximalMatches with the necessary arguments.\n        Args:\n            template_matches (list): list of matches obtained from running the algorithm.\n        '
        self.template_matches = template_matches
        self.max_match_list = []

    def run_maximal_matches(self):
        if False:
            return 10
        '\n        Method that extracts and stores maximal matches in decreasing length order.\n        '
        self.max_match_list = [Match(sorted(self.template_matches[0].match), self.template_matches[0].qubit, self.template_matches[0].clbit)]
        for matches in self.template_matches[1:]:
            present = False
            for max_match in self.max_match_list:
                for elem in matches.match:
                    if elem in max_match.match and len(matches.match) <= len(max_match.match):
                        present = True
            if not present:
                self.max_match_list.append(Match(sorted(matches.match), matches.qubit, matches.clbit))