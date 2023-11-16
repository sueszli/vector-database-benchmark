import vertexai
from vertexai.language_models import TextGenerationModel

def text_summarization(temperature: float, project_id: str, location: str) -> str:
    if False:
        return 10
    'Summarization Example with a Large Language Model'
    vertexai.init(project=project_id, location=location)
    parameters = {'temperature': temperature, 'max_output_tokens': 256, 'top_p': 0.95, 'top_k': 40}
    model = TextGenerationModel.from_pretrained('text-bison@001')
    response = model.predict('Provide a summary with about two sentences for the following article:\nThe efficient-market hypothesis (EMH) is a hypothesis in financial economics that states that asset prices reflect all available information. A direct implication is that it is impossible to "beat the market" consistently on a risk-adjusted basis since market prices should only react to new information. Because the EMH is formulated in terms of risk adjustment, it only makes testable predictions when coupled with a particular model of risk. As a result, research in financial economics since at least the 1990s has focused on market anomalies, that is, deviations from specific models of risk. The idea that financial market returns are difficult to predict goes back to Bachelier, Mandelbrot, and Samuelson, but is closely associated with Eugene Fama, in part due to his influential 1970 review of the theoretical and empirical research. The EMH provides the basic logic for modern risk-based theories of asset prices, and frameworks such as consumption-based asset pricing and intermediary asset pricing can be thought of as the combination of a model of risk with the EMH. Many decades of empirical research on return predictability has found mixed evidence. Research in the 1950s and 1960s often found a lack of predictability (e.g. Ball and Brown 1968; Fama, Fisher, Jensen, and Roll 1969), yet the 1980s-2000s saw an explosion of discovered return predictors (e.g. Rosenberg, Reid, and Lanstein 1985; Campbell and Shiller 1988; Jegadeesh and Titman 1993). Since the 2010s, studies have often found that return predictability has become more elusive, as predictability fails to work out-of-sample (Goyal and Welch 2008), or has been weakened by advances in trading technology and investor learning (Chordia, Subrahmanyam, and Tong 2014; McLean and Pontiff 2016; Martineau 2021).\nSummary:\n', **parameters)
    print(f'Response from Model: {response.text}')
    return response.text
if __name__ == '__main__':
    text_summarization()