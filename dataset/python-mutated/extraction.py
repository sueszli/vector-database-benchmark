import vertexai
from vertexai.language_models import TextGenerationModel

def extractive_question_answering(temperature: float, project_id: str, location: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Extractive Question Answering with a Large Language Model.'
    vertexai.init(project=project_id, location=location)
    parameters = {'temperature': temperature, 'max_output_tokens': 256, 'top_p': 0, 'top_k': 1}
    model = TextGenerationModel.from_pretrained('text-bison@001')
    response = model.predict('Background: There is evidence that there have been significant changes in Amazon rainforest vegetation over the last 21,000 years through the Last Glacial Maximum (LGM) and subsequent deglaciation. Analyses of sediment deposits from Amazon basin paleo lakes and from the Amazon Fan indicate that rainfall in the basin during the LGM was lower than for the present, and this was almost certainly associated with reduced moist tropical vegetation cover in the basin. There is debate, however, over how extensive this reduction was. Some scientists argue that the rainforest was reduced to small, isolated refugia separated by open forest and grassland; other scientists argue that the rainforest remained largely intact but extended less far to the north, south, and east than is seen today. This debate has proved difficult to resolve because the practical limitations of working in the rainforest mean that data sampling is biased away from the center of the Amazon basin, and both explanations are reasonably well supported by the available data.\n\nQ: What does LGM stands for?\nA: Last Glacial Maximum.\n\nQ: What did the analysis from the sediment deposits indicate?\nA: Rainfall in the basin during the LGM was lower than for the present.\n\nQ: What are some of scientists arguments?\nA: The rainforest was reduced to small, isolated refugia separated by open forest and grassland.\n\nQ: There have been major changes in Amazon rainforest vegetation over the last how many years?\nA: 21,000.\n\nQ: What caused changes in the Amazon rainforest vegetation?\nA: The Last Glacial Maximum (LGM) and subsequent deglaciation\n\nQ: What has been analyzed to compare Amazon rainfall in the past and present?\nA: Sediment deposits.\n\nQ: What has the lower rainfall in the Amazon during the LGM been attributed to?\nA:', **parameters)
    print(f'Response from Model: {response.text}')
    return response.text
if __name__ == '__main__':
    extractive_question_answering()