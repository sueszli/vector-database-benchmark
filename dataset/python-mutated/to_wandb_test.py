"""to wandb tests"""
import wandb
from hamcrest import assert_that, equal_to, not_none
from deepchecks.tabular.suites import full_suite
wandb.setup(wandb.Settings(mode='disabled', program=__name__, program_relpath=__name__, disable_code=True))

def test_check_full_suite_not_failing(iris_split_dataset_and_model):
    if False:
        print('Hello World!')
    (train, test, model) = iris_split_dataset_and_model
    suite_res = full_suite().run(train, test, model)
    suite_res.to_wandb()
    assert_that(wandb.run, equal_to(None))

def test_check_full_suite_init_before(iris_split_dataset_and_model):
    if False:
        i = 10
        return i + 15
    (train, test, model) = iris_split_dataset_and_model
    suite_res = full_suite().run(train, test, model)
    wandb.init()
    suite_res.to_wandb()
    assert_that(wandb.run, not_none())

def test_check_full_suite_deticated_false(iris_split_dataset_and_model):
    if False:
        while True:
            i = 10
    (train, test, model) = iris_split_dataset_and_model
    suite_res = full_suite().run(train, test, model)
    suite_res.to_wandb()
    assert_that(wandb.run, not_none())
    wandb.finish()
    assert_that(wandb.run, equal_to(None))

def test_check_full_suite_kwargs(iris_split_dataset_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train, test, model) = iris_split_dataset_and_model
    suite_res = full_suite().run(train, test, model)
    suite_res.to_wandb(project='ahh', config={'ahh': 'oh'})
    assert_that(wandb.run, equal_to(None))

def test_check_plotly(iris_split_dataset_and_model, simple_custom_plt_check):
    if False:
        return 10
    (train, test, _) = iris_split_dataset_and_model
    simple_custom_plt_check.run(train, test).to_wandb()
    assert_that(wandb.run, equal_to(None))