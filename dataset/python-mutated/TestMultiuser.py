import pytest
import json
from Config import config
from User import UserManager

@pytest.mark.usefixtures('resetSettings')
@pytest.mark.usefixtures('resetTempSettings')
class TestMultiuser:

    def testMemorySave(self, user):
        if False:
            while True:
                i = 10
        users_before = open('%s/users.json' % config.data_dir).read()
        user = UserManager.user_manager.create()
        user.save()
        assert open('%s/users.json' % config.data_dir).read() == users_before