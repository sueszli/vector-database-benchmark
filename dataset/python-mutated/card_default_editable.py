from metaflow_test import MetaflowTest, ExpectationFailed, steps, tag

class DefaultEditableCardTest(MetaflowTest):
    """
    `current.card.append` works for one decorator as default editable cards
        - adding arbitrary information to `current.card.append` should not break user code.
        - If a single @card decorator is present with `id` then it `current.card.append` should still work
        - Only cards with `ALLOW_USER_COMPONENTS=True` are considered default editable.
    """
    HEADER = '\nclass MyNativeType:\n    at = 0\n    def get(self):\n        return self.at\n    '
    PRIORITY = 3

    @tag('card(type="test_editable_card")')
    @steps(0, ['start'])
    def step_start(self):
        if False:
            while True:
                i = 10
        from metaflow import current
        from metaflow.plugins.cards.card_modules.test_cards import TestStringComponent
        import random
        self.random_number = random.randint(0, 100)
        current.card.append(current.pathspec)
        current.card.append(TestStringComponent(str(self.random_number)))
        empty_list = current.card.get(type='nonexistingtype')
        current.card.append(MyNativeType())

    @tag('card(type="test_editable_card", id="xyz")')
    @steps(0, ['foreach-nested-inner'])
    def step_foreach_inner(self):
        if False:
            print('Hello World!')
        from metaflow import current
        from metaflow.plugins.cards.card_modules.test_cards import TestStringComponent
        import random
        self.random_number = random.randint(0, 100)
        current.card.append(current.pathspec)
        current.card.append(TestStringComponent(str(self.random_number)))

    @tag('card(type="taskspec_card")')
    @tag('card(type="test_editable_card")')
    @steps(0, ['join'])
    def step_join(self):
        if False:
            i = 10
            return i + 15
        from metaflow import current
        from metaflow.plugins.cards.card_modules.test_cards import TestStringComponent
        import random
        self.random_number = random.randint(0, 100)
        current.card.append(current.pathspec)
        current.card.append(TestStringComponent(str(self.random_number)))

    @tag('card(type="test_editable_card")')
    @steps(1, ['all'])
    def step_all(self):
        if False:
            for i in range(10):
                print('nop')
        from metaflow import current
        from metaflow.plugins.cards.card_modules.test_cards import TestStringComponent
        import random
        self.random_number = random.randint(0, 100)
        current.card.append(current.pathspec)
        current.card.append(TestStringComponent(str(self.random_number)))

    def check_results(self, flow, checker):
        if False:
            while True:
                i = 10
        run = checker.get_run()
        card_type = 'test_editable_card'
        if run is None:
            for step in flow:
                cli_check_dict = checker.artifact_dict(step.name, 'random_number')
                for task_pathspec in cli_check_dict:
                    task_id = task_pathspec.split('/')[-1]
                    cards_info = checker.list_cards(step.name, task_id, card_type)
                    number = cli_check_dict[task_pathspec]['random_number']
                    assert_equals(cards_info is not None and 'cards' in cards_info and (len(cards_info['cards']) == 1), True)
                    card = cards_info['cards'][0]
                    checker.assert_card(step.name, task_id, card_type, '%d' % number, card_hash=card['hash'], exact_match=True)
        else:
            for step in flow:
                meta_check_dict = checker.artifact_dict(step.name, 'random_number')
                for task_id in meta_check_dict:
                    random_number = meta_check_dict[task_id]['random_number']
                    cards_info = checker.list_cards(step.name, task_id, card_type)
                    assert_equals(cards_info is not None and 'cards' in cards_info and (len(cards_info['cards']) == 1), True)
                    for card in cards_info['cards']:
                        checker.assert_card(step.name, task_id, card_type, '%d' % random_number, card_hash=card['hash'], exact_match=False)