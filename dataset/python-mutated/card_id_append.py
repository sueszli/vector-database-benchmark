from metaflow_test import MetaflowTest, ExpectationFailed, steps, tag

class CardsWithIdTest(MetaflowTest):
    """
    `current.card['myid']` should be accessible when cards have an `id` argument in decorator
    - `current.card.append` should not work when there are no single default editable card.
    - if a card has `ALLOW_USER_COMPONENTS=False` then it can still be edited via accessing it with `id` property.
    """
    PRIORITY = 3

    @tag('environment(vars={"METAFLOW_CARD_NO_WARNING": "True"})')
    @tag('card(type="test_editable_card",id="xyz")')
    @tag('card(type="test_editable_card",id="abc")')
    @steps(0, ['start'])
    def step_start(self):
        if False:
            while True:
                i = 10
        from metaflow import current
        from metaflow.plugins.cards.card_modules.test_cards import TestStringComponent
        import random
        self.random_number = random.randint(0, 100)
        self.random_number_2 = random.randint(0, 100)
        current.card['abc'].append(TestStringComponent(str(self.random_number)))
        current.card.append(TestStringComponent(str(self.random_number_2)))

    @tag('card(type="test_non_editable_card",id="abc")')
    @steps(0, ['end'])
    def step_end(self):
        if False:
            for i in range(10):
                print('nop')
        from metaflow import current
        from metaflow.plugins.cards.card_modules.test_cards import TestStringComponent
        import random
        self.random_number = random.randint(0, 100)
        self.random_number_2 = random.randint(0, 100)
        current.card['abc'].append(TestStringComponent(str(self.random_number)))

    @steps(1, ['all'])
    def step_all(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def check_results(self, flow, checker):
        if False:
            for i in range(10):
                print('nop')
        run = checker.get_run()
        if run is None:
            for step in flow:
                if step.name != 'start' or step.name != 'end':
                    continue
                cli_check_dict = checker.artifact_dict(step.name, 'random_number')
                for task_pathspec in cli_check_dict:
                    task_id = task_pathspec.split('/')[-1]
                    number = cli_check_dict[task_pathspec]['random_number']
                    checker.assert_card(step.name, task_id, 'test_editable_card' if step.name == 'start' else 'test_non_editable_card', '%d' % number, card_id='abc', exact_match=True)
        else:
            for step in flow:
                if step.name != 'start' or step.name != 'end':
                    continue
                meta_check_dict = checker.artifact_dict(step.name, 'random_number')
                for task_id in meta_check_dict:
                    random_number = meta_check_dict[task_id]['random_number']
                    checker.assert_card(step.name, task_id, 'test_editable_card' if step.name == 'start' else 'test_non_editable_card', '%d' % random_number, card_id='abc', exact_match=True)