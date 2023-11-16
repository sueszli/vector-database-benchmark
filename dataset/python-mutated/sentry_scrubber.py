import re
from typing import Any, Dict, List, Union
from tribler.core.sentry_reporter.sentry_reporter import BREADCRUMBS, RELEASE
from tribler.core.sentry_reporter.sentry_tools import delete_item, format_version, modify_value, obfuscate_string

class SentryScrubber:
    """This class has been created to be responsible for scrubbing all sensitive
    and unnecessary information from Sentry event.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.home_folders = ['users', 'usr', 'home', 'u01', 'var', 'data\\/media', 'WINNT\\\\Profiles', 'Documents and Settings', 'Users']
        self.dict_keys_for_scrub = ['USERNAME', 'USERDOMAIN', 'server_name', 'COMPUTERNAME']
        self.event_fields_to_cut = []
        self.exclusions = ['local', '127.0.0.1']
        self.sensitive_occurrences = {}
        self.create_placeholder = lambda text: f'<{text}>'
        self.hash_placeholder = self.create_placeholder('hash')
        self.ip_placeholder = self.create_placeholder('IP')
        self.re_folders = []
        self.re_ip = None
        self.re_hash = None
        self._compile_re()

    @staticmethod
    def remove_breadcrumbs(event: Dict) -> Dict:
        if False:
            i = 10
            return i + 15
        return delete_item(event, BREADCRUMBS)

    def _compile_re(self):
        if False:
            for i in range(10):
                print('nop')
        'Compile all regular expressions.'
        slash = '[/\\\\]'
        for folder in self.home_folders:
            for separator in [slash, slash * 2]:
                folder_pattern = f'(?<={folder}{separator})[\\w\\s~]+(?={separator})'
                self.re_folders.append(re.compile(folder_pattern, re.I))
        self.re_ip = re.compile('(?<!\\.)\\b(\\d{1,3}\\.){3}\\d{1,3}\\b(?!\\.)', re.I)
        self.re_hash = re.compile('\\b[0-9a-f]{40}\\b', re.I)

    def scrub_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Main method. Removes all sensitive and unnecessary information.\n\n        Args:\n            event: a Sentry event.\n\n        Returns:\n            Scrubbed the Sentry event.\n        '
        if not event:
            return event
        for field_name in self.event_fields_to_cut:
            delete_item(event, field_name)
        modify_value(event, RELEASE, format_version)
        scrubbed_event = self.scrub_entity_recursively(event)
        scrubbed_event = self.scrub_entity_recursively(scrubbed_event)
        return scrubbed_event

    def scrub_text(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Replace all sensitive information from `text` by corresponding\n        placeholders.\n\n        Sensitive information:\n            * IP\n            * User Name\n            * 40-symbols-long hash\n\n        A found user name will be stored and used for further replacements.\n        Args:\n            text:\n\n        Returns:\n            The text with removed sensitive information.\n        '
        if text is None:
            return text

        def scrub_username(m):
            if False:
                print('Hello World!')
            user_name = m.group(0)
            if user_name in self.exclusions:
                return user_name
            fake_username = obfuscate_string(user_name)
            placeholder = self.create_placeholder(fake_username)
            self.add_sensitive_pair(user_name, placeholder)
            return placeholder
        for regex in self.re_folders:
            text = regex.sub(scrub_username, text)

        def scrub_ip(m):
            if False:
                for i in range(10):
                    print('nop')
            return self.ip_placeholder if m.group(0) not in self.exclusions else m.group(0)
        text = self.re_ip.sub(scrub_ip, text)
        text = self.re_hash.sub(self.hash_placeholder, text)
        if self.sensitive_occurrences:
            escaped_sensitive_occurrences = [re.escape(user_name) for user_name in self.sensitive_occurrences]
            pattern = '([^<]|^)\\b(' + '|'.join(escaped_sensitive_occurrences) + ')\\b'

            def scrub_value(m):
                if False:
                    i = 10
                    return i + 15
                if m.group(2) not in self.sensitive_occurrences:
                    return m.group(0)
                return m.group(1) + self.sensitive_occurrences[m.group(2)]
            text = re.sub(pattern, scrub_value, text)
        return text

    def scrub_entity_recursively(self, entity: Union[str, Dict, List, Any], depth=10):
        if False:
            while True:
                i = 10
        'Recursively traverses entity and remove all sensitive information.\n\n        Can work with:\n            1. Text\n            2. Dictionaries\n            3. Lists\n\n        All other fields just will be skipped.\n\n        Args:\n            entity: an entity to process.\n            depth: depth of recursion.\n\n        Returns:\n            The entity with removed sensitive information.\n        '
        if depth < 0 or not entity:
            return entity
        depth -= 1
        if isinstance(entity, str):
            return self.scrub_text(entity)
        if isinstance(entity, list):
            return [self.scrub_entity_recursively(item, depth) for item in entity]
        if isinstance(entity, dict):
            result = {}
            for (key, value) in entity.items():
                if key in self.dict_keys_for_scrub:
                    value = value.strip()
                    fake_value = obfuscate_string(value)
                    placeholder = self.create_placeholder(fake_value)
                    self.add_sensitive_pair(value, placeholder)
                result[key] = self.scrub_entity_recursively(value, depth)
            return result
        return entity

    def add_sensitive_pair(self, text, placeholder):
        if False:
            print('Hello World!')
        if text in self.sensitive_occurrences:
            return
        self.sensitive_occurrences[text] = placeholder