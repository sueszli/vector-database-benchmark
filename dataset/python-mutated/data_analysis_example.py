import logging
import os
from os import getenv
import requests
from dotenv import load_dotenv
from e2b.templates.data_analysis import DataAnalysis
load_dotenv()
E2B_API_KEY = getenv('E2B_API_KEY')
logging.basicConfig(level=logging.ERROR)

def main():
    if False:
        while True:
            i = 10
    s = DataAnalysis(api_key=E2B_API_KEY)
    os.makedirs('data', exist_ok=True)
    with open('data/netflix.csv', 'wb') as f:
        response = requests.get('https://storage.googleapis.com/e2b-examples/netflix.csv')
        f.write(response.content)
    with open('data/netflix.csv', 'rb') as f:
        path = s.upload_file(file=f)
    (stdout, stderr, artifacts) = s.run_python(f"\nimport pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\ndata = pd.read_csv('{path}')\ntop_countries = data['country'].value_counts().head(10)\n\nplt.figure(figsize=(10, 6))\ntop_countries.plot(kind='bar', color='skyblue')\nplt.title('Number of content')\nplt.xlabel('Country')\nplt.ylabel('Count')\nplt.xticks(rotation=45)\nplt.show()\n")
    for artifact in artifacts:
        content = artifact.download()
        with open(os.path.split(artifact.name)[-1], 'wb') as f:
            f.write(content)
    s.close()
if __name__ == '__main__':
    main()