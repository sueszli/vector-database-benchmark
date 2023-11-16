from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

class IsQuestion:

    def __init__(self):
        if False:
            print('Hello World!')
        self.tokenizer = AutoTokenizer.from_pretrained('shahrukhx01/question-vs-statement-classifier')
        self.model = AutoModelForSequenceClassification.from_pretrained('shahrukhx01/question-vs-statement-classifier')
        self.classifier = pipeline('sentiment-analysis', model=self.model, tokenizer=self.tokenizer)
        self.labels = {'LABEL_0': False, 'LABEL_1': True}

    def __call__(self, text: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.labels[self.classifier(text)[0]['label']]
is_question = IsQuestion()