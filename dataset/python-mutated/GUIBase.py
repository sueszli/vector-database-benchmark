from ryvencore.Base import Base

class GUIBase:
    """Base class for GUI items that represent specific core components"""
    FRONTEND_COMPONENT_ASSIGNMENTS = {}

    @staticmethod
    def get_complete_data_function(session):
        if False:
            i = 10
            return i + 15
        '\n        generates a function that searches through generated data by the core and calls\n        complete_data() on frontend components that represent them to add frontend data\n        '

        def analyze(obj):
            if False:
                i = 10
                return i + 15
            'Searches recursively through obj and calls complete_data(obj) on associated\n            frontend components (instances of GUIBase)'
            if isinstance(obj, dict):
                GID = obj.get('GID')
                if GID is not None:
                    comp = GUIBase.FRONTEND_COMPONENT_ASSIGNMENTS.get(GID)
                    if comp:
                        obj = comp.complete_data(obj)
                for (key, value) in obj.items():
                    obj[key] = analyze(value)
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    item = obj[i]
                    item = analyze(item)
                    obj[i] = item
            return obj
        return analyze

    def __init__(self, representing_component: Base=None):
        if False:
            i = 10
            return i + 15
        'parameter `representing` indicates representation of a specific core component'
        if representing_component is not None:
            GUIBase.FRONTEND_COMPONENT_ASSIGNMENTS[representing_component.global_id] = self

    def complete_data(self, data: dict) -> dict:
        if False:
            for i in range(10):
                print('nop')
        'completes the data dict of the represented core component by adding all frontend data'
        return data