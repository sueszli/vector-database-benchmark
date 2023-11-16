"""
A ResultAction is an action that is applicable to at least some results. This
file serves the base class for all result actions, thus providing a unified
interface for all actions.
"""
from coala_utils.decorators import enforce_signature
from coalib.settings.FunctionMetadata import FunctionMetadata
from coalib.settings.Section import Section

class ResultAction:
    SUCCESS_MESSAGE = 'The action was executed successfully.'

    @staticmethod
    def is_applicable(result, original_file_dict, file_diff_dict, applied_actions=()):
        if False:
            print('Hello World!')
        '\n        Checks whether the Action is valid for the result type.\n\n        Returns ``True`` or a string containing the not_applicable message.\n\n        :param result:             The result from the coala run to check if an\n                                   Action is applicable.\n        :param original_file_dict: A dictionary containing the files in the\n                                   state where the result was generated.\n        :param file_diff_dict:     A dictionary containing a diff for every\n                                   file from the state in the\n                                   original_file_dict to the current state.\n                                   This dict will be altered so you do not\n                                   need to use the return value.\n        :applied_actions:          List of actions names that have already been\n                                   applied for the current result. Action names\n                                   are stored in order of application.\n        '
        return True

    def apply(self, result, original_file_dict, file_diff_dict, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        No description. Something went wrong.\n        '
        raise NotImplementedError

    @enforce_signature
    def apply_from_section(self, result, original_file_dict: dict, file_diff_dict: dict, section: Section):
        if False:
            print('Hello World!')
        '\n        Applies this action to the given results with all additional options\n        given as a section. The file dictionaries\n        are needed for differential results.\n\n        :param result:             The result to apply.\n        :param original_file_dict: A dictionary containing the files in the\n                                   state where the result was generated.\n        :param file_diff_dict:     A dictionary containing a diff for every\n                                   file from the state in the\n                                   original_file_dict to the current state.\n                                   This dict will be altered so you do not\n                                   need to use the return value.\n        :param section:            The section where to retrieve the additional\n                                   information.\n        :return:                   The modified file_diff_dict.\n        '
        params = self.get_metadata().create_params_from_section(section)
        return self.apply(result, original_file_dict, file_diff_dict, **params)

    def get_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieves metadata for the apply function. The description may be used\n        to advertise this action to the user. The parameters and their help\n        texts are additional information that are needed from the user. You can\n        create a section out of the inputs from the user and use\n        apply_from_section to apply\n\n        :return: A FunctionMetadata object.\n        '
        data = FunctionMetadata.from_function(self.apply, omit={'self', 'result', 'original_file_dict', 'file_diff_dict'})
        if hasattr(self, 'description'):
            data.desc = self.description
        data.name = self.__class__.__name__
        data.id = id(self)
        return data