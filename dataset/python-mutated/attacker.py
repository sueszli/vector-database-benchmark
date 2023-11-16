from typing import List
from allennlp.common import Registrable
from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor

class Attacker(Registrable):
    """
    An `Attacker` will modify an input (e.g., add or delete tokens) to try to change an AllenNLP
    Predictor's output in a desired manner (e.g., make it incorrect).
    """

    def __init__(self, predictor: Predictor) -> None:
        if False:
            while True:
                i = 10
        self.predictor = predictor

    def initialize(self):
        if False:
            return 10
        '\n        Initializes any components of the Attacker that are expensive to compute, so that they are\n        not created on __init__().  Default implementation is `pass`.\n        '
        pass

    def attack_from_json(self, inputs: JsonDict, input_field_to_attack: str, grad_input_field: str, ignore_tokens: List[str], target: JsonDict) -> JsonDict:
        if False:
            while True:
                i = 10
        "\n        This function finds a modification to the input text that would change the model's\n        prediction in some desired manner (e.g., an adversarial attack).\n\n        # Parameters\n\n        inputs : `JsonDict`\n            The input you want to attack (the same as the argument to a Predictor, e.g.,\n            predict_json()).\n        input_field_to_attack : `str`\n            The key in the inputs JsonDict you want to attack, e.g., `tokens`.\n        grad_input_field : `str`\n            The field in the gradients dictionary that contains the input gradients.  For example,\n            `grad_input_1` will be the field for single input tasks. See get_gradients() in\n            `Predictor` for more information on field names.\n        target : `JsonDict`\n            If given, this is a `targeted` attack, trying to change the prediction to a particular\n            value, instead of just changing it from its original prediction.  Subclasses are not\n            required to accept this argument, as not all attacks make sense as targeted attacks.\n            Perhaps that means we should make the API more crisp, but adding another class is not\n            worth it.\n\n        # Returns\n\n        reduced_input : `JsonDict`\n            Contains the final, sanitized input after adversarial modification.\n        "
        raise NotImplementedError()