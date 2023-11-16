from posthog.management.commands.test_migrations_are_safe import validate_migration_sql

def test_new_tables_must_not_have_int_32_ids() -> None:
    if False:
        for i in range(10):
            print('nop')
    sql_for_model_with_int32 = '\nBEGIN;\n--\n-- Create model StrawMan\n--\nCREATE TABLE "posthog_strawman" ("id" serial NOT NULL PRIMARY KEY, "name" varchar(400) NULL);\nCOMMIT;\nBEGIN;\n--\n-- Create model StrawMan\n--\nCREATE TABLE "posthog_strawman" ("id" serial NOT NULL PRIMARY KEY, "name" varchar(400) NULL);\nCOMMIT;\n'
    should_fail = validate_migration_sql(sql_for_model_with_int32)
    assert should_fail is True

def test_new_tables_can_have_int64_ids() -> None:
    if False:
        print('Hello World!')
    sql_for_model_with_int64 = '\nBEGIN;\n--\n-- Create model StrawMan\n--\nCREATE TABLE "posthog_strawman" ("id" bigserial NOT NULL PRIMARY KEY, "name" varchar(400) NULL);\nCOMMIT;\nBEGIN;\n--\n-- Create model StrawMan\n--\nCREATE TABLE "posthog_strawman" ("id" bigserial NOT NULL PRIMARY KEY, "name" varchar(400) NULL);\nCOMMIT;\n    '
    should_fail = validate_migration_sql(sql_for_model_with_int64)
    assert should_fail is False