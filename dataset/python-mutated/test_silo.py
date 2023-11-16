import pytest
from sentry.testutils.silo import validate_protected_queries

def test_validate_protected_queries__no_queries():
    if False:
        i = 10
        return i + 15
    validate_protected_queries([])

def test_validate_protected_queries__ok():
    if False:
        i = 10
        return i + 15
    queries = [{'sql': 'SELECT * FROM sentry_organization'}, {'sql': "UPDATE sentry_project SET slug = 'best-team' WHERE id = 1"}]
    validate_protected_queries(queries)

def test_validate_protected_queries__missing_fences():
    if False:
        return 10
    queries = [{'sql': 'SAVEPOINT "s123abc"'}, {'sql': 'UPDATE "sentry_useremail" SET "is_verified" = true WHERE "id" = 1'}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'oops\' WHERE "id" = 1'}, {'sql': 'UPDATE "sentry_project" SET "slug" = \'frontend\' WHERE "id" = 3'}]
    with pytest.raises(AssertionError):
        validate_protected_queries(queries)

def test_validate_protected_queries__with_single_fence():
    if False:
        while True:
            i = 10
    queries = [{'sql': 'SAVEPOINT "s123abc"'}, {'sql': 'UPDATE "sentry_useremail" SET "is_verified" = true WHERE "id" = 1'}, {'sql': "SELECT 'start_role_override_1'"}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'oops\' WHERE "id" = 1'}, {'sql': "SELECT 'end_role_override_1'"}, {'sql': 'UPDATE "sentry_project" SET "slug" = \'frontend\' WHERE "id" = 3'}]
    validate_protected_queries(queries)

def test_validate_protected_queries__multiple_fences():
    if False:
        print('Hello World!')
    queries = [{'sql': 'SAVEPOINT "s123abc"'}, {'sql': 'UPDATE "sentry_useremail" SET "is_verified" = true WHERE "id" = 1'}, {'sql': "SELECT 'start_role_override_1'"}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'oops\' WHERE "id" = 1'}, {'sql': "SELECT 'end_role_override_1'"}, {'sql': 'UPDATE "sentry_project" SET "slug" = \'frontend\' WHERE "id" = 3'}, {'sql': "SELECT 'start_role_override_2'"}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'another-oops\' WHERE "id" = 1'}, {'sql': "SELECT 'end_role_override_2'"}]
    validate_protected_queries(queries)

def test_validate_protected_queries__nested_fences():
    if False:
        while True:
            i = 10
    queries = [{'sql': 'SAVEPOINT "s123abc"'}, {'sql': 'UPDATE "sentry_useremail" SET "is_verified" = true WHERE "id" = 1'}, {'sql': "SELECT 'start_role_override_1'"}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'safe\' WHERE "id" = 1'}, {'sql': "SELECT 'start_role_override_2'"}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'also-safe\' WHERE "id" = 1'}, {'sql': "SELECT 'end_role_override_2'"}, {'sql': "SELECT 'end_role_override_1'"}, {'sql': 'UPDATE "sentry_project" SET "slug" = \'frontend\' WHERE "id" = 3'}, {'sql': 'UPDATE "sentry_organizationmemberteam" SET "role" = \'member\' WHERE "id" = 3'}]
    validate_protected_queries(queries)
    queries = [{'sql': 'SAVEPOINT "s123abc"'}, {'sql': 'UPDATE "sentry_useremail" SET "is_verified" = true WHERE "id" = 1'}, {'sql': "SELECT 'start_role_override_1'"}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'safe\' WHERE "id" = 1'}, {'sql': "SELECT 'start_role_override_2'"}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'also-safe\' WHERE "id" = 1'}, {'sql': "SELECT 'end_role_override_2'"}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'still-safe\' WHERE "id" = 1'}, {'sql': "SELECT 'end_role_override_1'"}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'not-safe\' WHERE "id" = 1'}]
    with pytest.raises(AssertionError):
        validate_protected_queries(queries)

def test_validate_protected_queries__fenced_and_not():
    if False:
        for i in range(10):
            print('nop')
    queries = [{'sql': 'SAVEPOINT "s123abc"'}, {'sql': 'UPDATE "sentry_useremail" SET "is_verified" = true WHERE "id" = 1'}, {'sql': "SELECT 'start_role_override_1'"}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'oops\' WHERE "id" = 1'}, {'sql': "SELECT 'end_role_override_1'"}, {'sql': 'UPDATE "sentry_project" SET "slug" = \'frontend\' WHERE "id" = 3'}, {'sql': 'UPDATE "sentry_organization" SET "slug" = \'another-oops\' WHERE "id" = 1'}]
    with pytest.raises(AssertionError):
        validate_protected_queries(queries)