import os
import time
import pytest
import testinfra
SKIP_REASON = 'unimplemented, see GH#3689'

class TestBase:

    @pytest.fixture(autouse=True)
    def only_mon_staging_sudo(self, host):
        if False:
            print('Hello World!')
        if host.backend.host != 'mon-staging':
            pytest.skip()
        with host.sudo():
            yield

    def ansible(self, host, module, parameters):
        if False:
            while True:
                i = 10
        r = host.ansible(module, parameters, check=False)
        assert 'exception' not in r

    def run(self, host, cmd):
        if False:
            return 10
        print(host.backend.host + ' running: ' + cmd)
        r = host.run(cmd)
        print(r.stdout)
        print(r.stderr)
        return r.rc == 0

    def wait_for(self, fun):
        if False:
            return 10
        success = False
        for d in (1, 2, 4, 8, 16, 32, 64):
            if fun():
                success = True
                break
            time.sleep(d)
        return success

    def wait_for_command(self, host, cmd):
        if False:
            print('Hello World!')
        return self.wait_for(lambda : self.run(host, cmd))

    def service_started(self, host, name):
        if False:
            return 10
        assert self.run(host, f'service {name} start')
        assert self.wait_for_command(host, f"service {name} status | grep -q 'is running'")

    def service_restarted(self, host, name):
        if False:
            while True:
                i = 10
        assert self.run(host, f'service {name} restart')
        assert self.wait_for_command(host, f"service {name} status | grep -q 'is running'")

    def service_stopped(self, host, name):
        if False:
            while True:
                i = 10
        assert self.run(host, f'service {name} stop')
        assert self.wait_for_command(host, f"service {name} status | grep -q 'not running'")

class TestJournalistMail(TestBase):

    @pytest.mark.skip(reason=SKIP_REASON)
    def test_procmail(self, host):
        if False:
            print('Hello World!')
        self.service_started(host, 'postfix')
        for (destination, f) in (('journalist', 'alert-journalist-one.txt'), ('journalist', 'alert-journalist-two.txt'), ('ossec', 'alert-ossec.txt')):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.ansible(host, 'copy', 'dest=/tmp/{f} src={d}/{f}'.format(f=f, d=current_dir))
            assert self.run(host, '/var/ossec/process_submissions_today.sh forget')
            assert self.run(host, 'postsuper -d ALL')
            assert self.run(host, f"cat /tmp/{f} | mail -s 'abc' root@localhost")
            assert self.wait_for_command(host, f'mailq | grep -q {destination}@ossec.test')
        self.service_stopped(host, 'postfix')

    @pytest.mark.skip(reason=SKIP_REASON)
    def test_process_submissions_today(self, host):
        if False:
            print('Hello World!')
        assert self.run(host, '/var/ossec/process_submissions_today.sh test_handle_notification')
        assert self.run(host, '/var/ossec/process_submissions_today.sh test_modified_in_the_past_24h')

    @pytest.mark.skip(reason=SKIP_REASON)
    def test_send_encrypted_alert(self, host):
        if False:
            for i in range(10):
                print('nop')
        self.service_started(host, 'postfix')
        src = '../../install_files/ansible-base/roles/ossec/files/test_admin_key.sec'
        self.ansible(host, 'copy', f'dest=/tmp/test_admin_key.sec src={src}')
        self.run(host, 'gpg  --homedir /var/ossec/.gnupg --import /tmp/test_admin_key.sec')

        def trigger(who, payload):
            if False:
                i = 10
                return i + 15
            assert self.run(host, f'! mailq | grep -q {who}@ossec.test')
            assert self.run(host, "\n                ( echo 'Subject: TEST' ; echo ; echo -e '{payload}' ) |                 /var/ossec/send_encrypted_alarm.sh {who}\n                ".format(who=who, payload=payload))
            assert self.wait_for_command(host, f'mailq | grep -q {who}@ossec.test')
        for (who, payload, expected) in (('journalist', 'JOURNALISTPAYLOAD', 'JOURNALISTPAYLOAD'), ('ossec', 'OSSECPAYLOAD', 'OSSECPAYLOAD')):
            assert self.run(host, 'postsuper -d ALL')
            trigger(who, payload)
            assert self.run(host, "\n                job=$(mailq | sed -n -e '2p' | cut -f1 -d ' ')\n                postcat -q $job | tee /dev/stderr |                    gpg --homedir /var/ossec/.gnupg --decrypt 2>&1 |                    grep -q {expected}\n                ".format(expected=expected))
        try:
            assert self.run(host, 'postsuper -d ALL')
            assert self.run(host, 'mv /usr/bin/gpg /usr/bin/gpg.save')
            trigger(who, 'MYGREATPAYLOAD')
            assert self.run(host, "\n                job=$(mailq | sed -n -e '2p' | cut -f1 -d ' ')\n                postcat -q $job | grep -q 'Failed to encrypt OSSEC alert'\n                ")
        finally:
            assert self.run(host, 'mv /usr/bin/gpg.save /usr/bin/gpg')
        self.service_stopped(host, 'postfix')

    @pytest.mark.skip(reason=SKIP_REASON)
    def test_missing_journalist_alert(self, host):
        if False:
            print('Hello World!')
        assert self.run(host, "\n            JOURNALIST_EMAIL=                bash -x /var/ossec/send_encrypted_alarm.sh journalist |                tee /dev/stderr |                grep -q 'no notification sent'\n            ")

    @pytest.mark.skip(reason=SKIP_REASON)
    def test_ossec_rule_journalist(self, host):
        if False:
            return 10
        assert self.run(host, '\n        set -ex\n        l="ossec: output: \'head -1 /var/lib/securedrop/submissions_today.txt"\n        echo "$l" | /var/ossec/bin/ossec-logtest\n        echo "$l" | /var/ossec/bin/ossec-logtest -U \'400600:1:ossec\'\n        ')

    @pytest.mark.skip(reason=SKIP_REASON)
    def test_journalist_mail_notification(self, host):
        if False:
            return 10
        mon = host
        app = testinfra.host.Host.get_host('ansible://app-staging', ansible_inventory=host.backend.ansible_inventory)
        self.service_started(mon, 'postfix')
        self.service_started(mon, 'ossec')
        with app.sudo():
            assert self.run(app, '\n            cd /var/www/securedrop\n            ./manage.py were-there-submissions-today\n            test -f /var/lib/securedrop/submissions_today.txt\n            ')
        assert self.run(mon, 'postsuper -d ALL')
        assert self.run(mon, '/var/ossec/process_submissions_today.sh forget')
        with app.sudo():
            self.service_restarted(app, 'ossec')
        assert self.wait_for_command(mon, 'mailq | grep -q journalist@ossec.test')
        assert self.run(mon, 'test 1 = $(mailq | grep journalist@ossec.test | wc -l)')
        assert self.run(mon, "grep --count 'notification suppressed' /var/log/syslog > /tmp/before")
        with app.sudo():
            self.service_restarted(app, 'ossec')
        assert self.wait_for_command(mon, "\n        grep --count 'notification suppressed' /var/log/syslog > /tmp/after\n        test $(cat /tmp/before) -lt $(cat /tmp/after)\n        ")
        self.service_stopped(mon, 'postfix')
        self.service_stopped(mon, 'ossec')
        with app.sudo():
            self.service_stopped(app, 'ossec')