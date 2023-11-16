"""Testing invalid cookiecutter template repositories."""
import pytest
from cookiecutter import exceptions, main

def test_should_raise_error_if_repo_does_not_exist():
    if False:
        return 10
    'Cookiecutter invocation with non-exist repository should raise error.'
    with pytest.raises(exceptions.RepositoryNotFound):
        main.cookiecutter('definitely-not-a-valid-repo-dir')