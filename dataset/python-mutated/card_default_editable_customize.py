from metaflow_test import MetaflowTest, ExpectationFailed, steps, tag

class DefaultEditableCardWithCustomizeTest(MetaflowTest):
    """
    `current.card.append` should be accessible to the card with `customize=True`.
        - Even if there are other editable cards without `id` and with `id`
    """
    PRIORITY = 3

    @tag('card(type="test_editable_card",customize=True)')
    @tag('card(type="test_editable_card",id="abc")')
    @tag('card(type="taskspec_card")')
    @tag('card(type="test_editable_card_2")')
    @steps(0, ['start'])
    def step_start(self):
        if False:
            for i in range(10):
                print('nop')
        from metaflow import current
        from metaflow.plugins.cards.card_modules.test_cards import TestStringComponent
        import random
        self.random_number = random.randint(0, 100)
        current.card.append(TestStringComponent(str(self.random_number)))

    @steps(1, ['all'])
    def step_all(self):
        if False:
            while True:
                i = 10
        pass

    def check_results(self, flow, checker):
        if False:
            print('Hello World!')
        run = checker.get_run()
        card_type = 'test_editable_card'
        if run is None:
            for step in flow:
                if step.name != 'start':
                    continue
                cli_check_dict = checker.artifact_dict(step.name, 'random_number')
                for task_pathspec in cli_check_dict:
                    task_id = task_pathspec.split('/')[-1]
                    cards_info = checker.list_cards(step.name, task_id, card_type)
                    assert_equals(cards_info is not None and 'cards' in cards_info and (len(cards_info['cards']) == 2), True)
                    default_editable_cards = [c for c in cards_info['cards'] if c['id'] is None]
                    assert_equals(len(default_editable_cards) == 1, True)
                    card = default_editable_cards[0]
                    number = cli_check_dict[task_pathspec]['random_number']
                    checker.assert_card(step.name, task_id, card_type, '%d' % number, card_hash=card['hash'], exact_match=True)
        else:
            for step in flow:
                if step.name != 'start':
                    continue
                meta_check_dict = checker.artifact_dict(step.name, 'random_number')
                for task_id in meta_check_dict:
                    cards_info = checker.list_cards(step.name, task_id, card_type)
                    assert_equals(cards_info is not None and 'cards' in cards_info and (len(cards_info['cards']) == 2), True)
                    default_editable_cards = [c for c in cards_info['cards'] if c['id'] is None]
                    assert_equals(len(default_editable_cards) == 1, True)
                    card = default_editable_cards[0]
                    random_number = meta_check_dict[task_id]['random_number']
                    checker.assert_card(step.name, task_id, card_type, '%d' % random_number, card_hash=card['hash'], exact_match=True)