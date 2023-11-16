import pytest
from django.core.management.base import CommandError
from awx.main.management.commands.inventory_import import Command

@pytest.mark.inventory_import
class TestInvalidOptions:

    def test_invalid_options_no_options_specified(self):
        if False:
            i = 10
            return i + 15
        cmd = Command()
        with pytest.raises(CommandError) as err:
            cmd.handle()
        assert 'inventory-id' in str(err.value)
        assert 'required' in str(err.value)

    def test_invalid_options_name_and_id(self):
        if False:
            i = 10
            return i + 15
        cmd = Command()
        with pytest.raises(CommandError) as err:
            cmd.handle(inventory_id=42, inventory_name='my-inventory')
        assert 'inventory-id' in str(err.value)
        assert 'exclusive' in str(err.value)

    def test_invalid_options_missing_source(self):
        if False:
            print('Hello World!')
        cmd = Command()
        with pytest.raises(CommandError) as err:
            cmd.handle(inventory_id=42)
        assert '--source' in str(err.value)
        assert 'required' in str(err.value)