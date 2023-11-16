import re
import urllib.request
import spacy
from .utils import TextCleaner
nlp = spacy.load('en_core_web_sm')
RESUME_SECTIONS = ['Contact Information', 'Objective', 'Summary', 'Education', 'Experience', 'Skills', 'Projects', 'Certifications', 'Licenses', 'Awards', 'Honors', 'Publications', 'References', 'Technical Skills', 'Computer Skills', 'Programming Languages', 'Software Skills', 'Soft Skills', 'Language Skills', 'Professional Skills', 'Transferable Skills', 'Work Experience', 'Professional Experience', 'Employment History', 'Internship Experience', 'Volunteer Experience', 'Leadership Experience', 'Research Experience', 'Teaching Experience']

class DataExtractor:
    """
    A class for extracting various types of data from text.
    """

    def __init__(self, raw_text: str):
        if False:
            i = 10
            return i + 15
        '\n        Initialize the DataExtractor object.\n\n        Args:\n            raw_text (str): The raw input text.\n        '
        self.text = raw_text
        self.clean_text = TextCleaner.clean_text(self.text)
        self.doc = nlp(self.clean_text)

    def extract_links(self):
        if False:
            i = 10
            return i + 15
        '\n        Find links of any type in a given string.\n\n        Args:\n            text (str): The string to search for links.\n\n        Returns:\n            list: A list containing all the found links.\n        '
        link_pattern = '\\b(?:https?://|www\\.)\\S+\\b'
        links = re.findall(link_pattern, self.text)
        return links

    def extract_links_extended(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract links of all kinds (HTTP, HTTPS, FTP, email, www.linkedin.com,\n          and github.com/user_name) from a webpage.\n\n        Args:\n            url (str): The URL of the webpage.\n\n        Returns:\n            list: A list containing all the extracted links.\n        '
        links = []
        try:
            response = urllib.request.urlopen(self.text)
            html_content = response.read().decode('utf-8')
            pattern = 'href=[\\\'"]?([^\\\'" >]+)'
            raw_links = re.findall(pattern, html_content)
            for link in raw_links:
                if link.startswith(('http://', 'https://', 'ftp://', 'mailto:', 'www.linkedin.com', 'github.com/', 'twitter.com')):
                    links.append(link)
        except Exception as e:
            print(f'Error extracting links: {str(e)}')
        return links

    def extract_names(self):
        if False:
            i = 10
            return i + 15
        "Extracts and returns a list of names from the given \n        text using spaCy's named entity recognition.\n\n        Args:\n            text (str): The text to extract names from.\n\n        Returns:\n            list: A list of strings representing the names extracted from the text.\n        "
        names = [ent.text for ent in self.doc.ents if ent.label_ == 'PERSON']
        return names

    def extract_emails(self):
        if False:
            i = 10
            return i + 15
        '\n        Extract email addresses from a given string.\n\n        Args:\n            text (str): The string from which to extract email addresses.\n\n        Returns:\n            list: A list containing all the extracted email addresses.\n        '
        email_pattern = '\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b'
        emails = re.findall(email_pattern, self.text)
        return emails

    def extract_phone_numbers(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract phone numbers from a given string.\n\n        Args:\n            text (str): The string from which to extract phone numbers.\n\n        Returns:\n            list: A list containing all the extracted phone numbers.\n        '
        phone_number_pattern = '^(\\+\\d{1,3})?[-.\\s]?\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}$'
        phone_numbers = re.findall(phone_number_pattern, self.text)
        return phone_numbers

    def extract_experience(self):
        if False:
            return 10
        '\n        Extract experience from a given string. It does so by using the Spacy module.\n\n        Args:\n            text (str): The string from which to extract experience.\n\n        Returns:\n            str: A string containing all the extracted experience.\n        '
        experience_section = []
        in_experience_section = False
        for token in self.doc:
            if token.text in RESUME_SECTIONS:
                if token.text == 'Experience' or 'EXPERIENCE' or 'experience':
                    in_experience_section = True
                else:
                    in_experience_section = False
            if in_experience_section:
                experience_section.append(token.text)
        return ' '.join(experience_section)

    def extract_position_year(self):
        if False:
            return 10
        '\n            Extract position and year from a given string.\n\n            Args:\n                text (str): The string from which to extract position and year.\n\n            Returns:\n                list: A list containing the extracted position and year.\n        '
        position_year_search_pattern = '(\\b\\w+\\b\\s+\\b\\w+\\b),\\s+(\\d{4})\\s*-\\s*(\\d{4}|\\bpresent\\b)'
        position_year = re.findall(position_year_search_pattern, self.text)
        return position_year

    def extract_particular_words(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract nouns and proper nouns from the given text.\n\n        Args:\n            text (str): The input text to extract nouns from.\n\n        Returns:\n            list: A list of extracted nouns.\n        '
        pos_tags = ['NOUN', 'PROPN']
        nouns = [token.text for token in self.doc if token.pos_ in pos_tags]
        return nouns

    def extract_entities(self):
        if False:
            return 10
        "\n        Extract named entities of types 'GPE' (geopolitical entity) and 'ORG' (organization) from the given text.\n\n        Args:\n            text (str): The input text to extract entities from.\n\n        Returns:\n            list: A list of extracted entities.\n        "
        entity_labels = ['GPE', 'ORG']
        entities = [token.text for token in self.doc.ents if token.label_ in entity_labels]
        return list(set(entities))