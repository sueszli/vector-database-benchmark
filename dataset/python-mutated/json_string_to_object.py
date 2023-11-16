import json
from st2common.runners.base_action import Action

class JsonStringToObject(Action):

    def run(self, json_str):
        if False:
            i = 10
            return i + 15
        return json.loads(json_str)