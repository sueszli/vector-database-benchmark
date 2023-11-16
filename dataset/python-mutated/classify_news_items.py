from vertexai.language_models import TextGenerationModel

def classify_news_items(temperature: float=0.2) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Text Classification Example with a Large Language Model'
    parameters = {'temperature': temperature, 'max_output_tokens': 5, 'top_p': 0, 'top_k': 1}
    model = TextGenerationModel.from_pretrained('text-bison@001')
    response = model.predict('What is the topic for a given news headline?\n- business\n- entertainment\n- health\n- sports\n- technology\n\nText: Pixel 7 Pro Expert Hands On Review, the Most Helpful Google Phones.\nThe answer is: technology\n\nText: Quit smoking?\nThe answer is: health\n\nText: Roger Federer reveals why he touched Rafael Nadals hand while they were crying\nThe answer is: sports\n\nText: Business relief from Arizona minimum-wage hike looking more remote\nThe answer is: business\n\nText: #TomCruise has arrived in Bari, Italy for #MissionImpossible.\nThe answer is: entertainment\n\nText: CNBC Reports Rising Digital Profit as Print Advertising Falls\nThe answer is:\n', **parameters)
    print(f'Response from Model: {response.text}')
    return response
if __name__ == '__main__':
    classify_news_items()