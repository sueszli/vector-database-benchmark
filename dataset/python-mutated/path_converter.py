"""
Converter class that handles the conversion of paths from Api Gateway to Flask and back.
"""
import re
PROXY_PATH_PARAMS_ESCAPED = '(.*/){(.*)\\+}'
FLASK_CAPTURE_ALL_PATH = '\\g<1><path:\\g<2>>'
PROXY_PATH_PARAMS = '/{\\g<1>+}'
FLASK_CAPTURE_ALL_PATH_REGEX = '/<path:(.*)>'
LEFT_BRACKET = '{'
RIGHT_BRACKET = '}'
LEFT_ANGLE_BRACKET = '<'
RIGHT_ANGLE_BRACKET = '>'
APIGW_TO_FLASK_REGEX = re.compile(PROXY_PATH_PARAMS_ESCAPED)
FLASK_TO_APIGW_REGEX = re.compile(FLASK_CAPTURE_ALL_PATH_REGEX)

class PathConverter:

    @staticmethod
    def convert_path_to_flask(path):
        if False:
            while True:
                i = 10
        "\n        Converts a Path from an Api Gateway defined path to one that is accepted by Flask\n\n        Examples:\n\n        '/id/{id}' => '/id/<id>'\n        '/{proxy+}' => '/<path:proxy>'\n\n        :param str path: Path to convert to Flask defined path\n        :return str: Path representing a Flask path\n        "
        proxy_sub_path = APIGW_TO_FLASK_REGEX.sub(FLASK_CAPTURE_ALL_PATH, path)
        return proxy_sub_path.replace(LEFT_BRACKET, LEFT_ANGLE_BRACKET).replace(RIGHT_BRACKET, RIGHT_ANGLE_BRACKET)

    @staticmethod
    def convert_path_to_api_gateway(path):
        if False:
            print('Hello World!')
        "\n        Converts a Path from a Flask defined path to one that is accepted by Api Gateway\n\n        Examples:\n\n        '/id/<id>' => '/id/{id}'\n        '/<path:proxy>' => '/{proxy+}'\n\n        :param str path: Path to convert to Api Gateway defined path\n        :return str: Path representing an Api Gateway path\n        "
        proxy_sub_path = FLASK_TO_APIGW_REGEX.sub(PROXY_PATH_PARAMS, path)
        return proxy_sub_path.replace(LEFT_ANGLE_BRACKET, LEFT_BRACKET).replace(RIGHT_ANGLE_BRACKET, RIGHT_BRACKET)