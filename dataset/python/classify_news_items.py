# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START aiplatform_sdk_classify_news_items]
from vertexai.language_models import TextGenerationModel


def classify_news_items(temperature: float = 0.2) -> None:
    """Text Classification Example with a Large Language Model"""

    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 5,  # Token limit determines the maximum amount of text output.
        "top_p": 0,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 1,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        """What is the topic for a given news headline?
- business
- entertainment
- health
- sports
- technology

Text: Pixel 7 Pro Expert Hands On Review, the Most Helpful Google Phones.
The answer is: technology

Text: Quit smoking?
The answer is: health

Text: Roger Federer reveals why he touched Rafael Nadals hand while they were crying
The answer is: sports

Text: Business relief from Arizona minimum-wage hike looking more remote
The answer is: business

Text: #TomCruise has arrived in Bari, Italy for #MissionImpossible.
The answer is: entertainment

Text: CNBC Reports Rising Digital Profit as Print Advertising Falls
The answer is:
""",
        **parameters,
    )

    print(f"Response from Model: {response.text}")

    return response


if __name__ == "__main__":
    classify_news_items()
# [END aiplatform_sdk_classify_news_items]
