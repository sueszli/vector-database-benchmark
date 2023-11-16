"""
FILE: ml_samples_evaluate.py

DESCRIPTION:
    These samples demonstrate evaluation api usage.

USAGE:
    python ml_samples_evaluate.py

"""

class AIEvaluateSamples(object):

    def ai_evaluate_qa(self):
        if False:
            while True:
                i = 10
        import os
        from azure.ai.generative import evaluate
        from azure.ai.resources.client import AIClient
        from azure.identity import DefaultAzureCredential
        data_location = '<path_to_data_in_jsonl_format>'

        def sample_chat(question):
            if False:
                while True:
                    i = 10
            return question
        client = AIClient.from_config(DefaultAzureCredential())
        result = evaluate(evaluation_name='my-evaluation', target=sample_chat, data=data_location, task_type='qa', data_mapping={'questions': 'question', 'contexts': 'context', 'y_pred': 'answer', 'y_test': 'truth'}, model_config={'api_version': '2023-05-15', 'api_base': os.getenv('OPENAI_API_BASE'), 'api_type': 'azure', 'api_key': os.getenv('OPENAI_API_KEY'), 'deployment_id': os.getenv('AZURE_OPENAI_EVALUATION_DEPLOYMENT')}, tracking_uri=client.tracking_uri)
if __name__ == '__main__':
    sample = AIEvaluateSamples()
    sample.ai_evaluate_qa()