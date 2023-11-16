import textwrap
from vertexai.language_models import CodeGenerationModel

def generate_unittest(temperature: float=0.5) -> object:
    if False:
        print('Hello World!')
    'Example of using Codey for Code Generation to write a unit test.'
    parameters = {'temperature': temperature, 'max_output_tokens': 256}
    code_generation_model = CodeGenerationModel.from_pretrained('code-bison@001')
    response = code_generation_model.predict(prefix=textwrap.dedent('    Write a unit test for this function:\n    def is_leap_year(year):\n        if year % 4 == 0:\n            if year % 100 == 0:\n                if year % 400 == 0:\n                    return True\n                else:\n                    return False\n            else:\n                return True\n        else:\n            return False\n    '), **parameters)
    print(f'Response from Model: {response.text}')
    return response
if __name__ == '__main__':
    generate_unittest()