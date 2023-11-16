"""Tests for signing and verifying blobs of data via gpg."""
import sys
from bzrlib import config, errors, gpg, tests, trace, ui
from bzrlib.tests import TestCase, features

class FakeConfig(config.MemoryStack):

    def __init__(self, content=None):
        if False:
            return 10
        if content is None:
            content = '\ngpg_signing_key=amy@example.com\ngpg_signing_command=false'
        super(FakeConfig, self).__init__(content)

class TestCommandLine(tests.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestCommandLine, self).setUp()
        self.my_gpg = gpg.GPGStrategy(FakeConfig())

    def test_signing_command_line(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(['false', '--clearsign', '-u', 'amy@example.com'], self.my_gpg._command_line())

    def test_signing_command_line_from_default(self):
        if False:
            i = 10
            return i + 15
        my_gpg = gpg.GPGStrategy(FakeConfig('\nemail=Amy <amy@example.com>\ngpg_signing_key=default\ngpg_signing_command=false'))
        self.assertEqual(['false', '--clearsign', '-u', 'amy@example.com'], my_gpg._command_line())

    def test_signing_command_line_from_email(self):
        if False:
            for i in range(10):
                print('nop')
        my_gpg = gpg.GPGStrategy(FakeConfig('\nemail=Amy <amy@example.com>\ngpg_signing_command=false'))
        self.assertEqual(['false', '--clearsign', '-u', 'amy@example.com'], my_gpg._command_line())

    def test_checks_return_code(self):
        if False:
            return 10
        self.assertRaises(errors.SigningFailed, self.my_gpg.sign, 'content')

    def assertProduces(self, content):
        if False:
            return 10
        if sys.platform == 'win32':
            self.my_gpg._command_line = lambda : [sys.executable, '-c', 'import sys; sys.stdout.write(sys.stdin.read())']
            new_content = content.replace('\n', '\r\n')
            self.assertEqual(new_content, self.my_gpg.sign(content))
        else:
            self.my_gpg._command_line = lambda : ['cat', '-']
            self.assertEqual(content, self.my_gpg.sign(content))

    def test_returns_output(self):
        if False:
            while True:
                i = 10
        content = 'some content\nwith newlines\n'
        self.assertProduces(content)

    def test_clears_progress(self):
        if False:
            for i in range(10):
                print('nop')
        content = 'some content\nwith newlines\n'
        old_clear_term = ui.ui_factory.clear_term
        clear_term_called = []

        def clear_term():
            if False:
                for i in range(10):
                    print('nop')
            old_clear_term()
            clear_term_called.append(True)
        ui.ui_factory.clear_term = clear_term
        try:
            self.assertProduces(content)
        finally:
            ui.ui_factory.clear_term = old_clear_term
        self.assertEqual([True], clear_term_called)

    def test_aborts_on_unicode(self):
        if False:
            return 10
        "You can't sign Unicode text; it must be encoded first."
        self.assertRaises(errors.BzrBadParameterUnicode, self.assertProduces, u'foo')

class TestVerify(TestCase):

    def import_keys(self):
        if False:
            return 10
        from StringIO import StringIO
        import gpgme
        context = gpgme.Context()
        key = StringIO('-----BEGIN PGP PUBLIC KEY BLOCK-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\nmQENBE343IgBCADwzPW7kmKb2bjB+UU+1ER/ABMZspvtoZMPusUw7bk6coXHF/0W\nu1K/hSYeX9xaGOfOQw41r/g13MoR9dsL6L84RLiisf38rRoBZt+d5bCbZA5Xo801\n2PeoBoGo6u5oOYKAFLMvrUitPiiE0IT/oQTfC4YUrLN4A+9W0QZruPGIpIXwmZXr\nL0zsqYfNqIN0ompeJenVpKpvm3loJ/zfK7R3EJ3hsv6nkUmWCFsP1Pw3UV1YuCmw\nMkdn1U7DaOql1WjXgj9ABQDJrun2TGsqrSRzBODtHKA/uOX0K3VfKBU8VZo3dXUm\n1Q4ZeZC39L9qJGTH8TQYlwBLe1yAOp+vx7QJABEBAAG0JEJhemFhciBUZXN0IEtl\neSA8YmF6YWFyQGV4YW1wbGUuY29tPokBOAQTAQIAIgUCTfjciAIbAwYLCQgHAwIG\nFQgCCQoLBBYCAwECHgECF4AACgkQh2gbHuMIDkWJUggAwj537fH6WW+GGLA5onys\n2hZmXUq/tU+L92bjQoRY4fmsQpk/FUVPUf+NQ0v1gkxx4BTfyYewaj5G6L8cvqW2\njj7UiJd8z9gTRxWTnYwfR/w5PGmxfJsBfEUKWsccrPQdOXAhwu0fjYIVk4nqgswa\nIOAZIwe5Vsfs36uSS7p8RQHAZXLXtTOn3KcXHaxu83w6nc4zkWRovGJ9isBN3haO\n2qEa0mYiAfDpz40CGtb8N/TQHF3Xcw8rJcxpg6RF3jMtWQnzbVJFp13it00R3LqW\no/r3RII3Ii3z2yARlg6D+5hVOrFBV8jFLkff1R2ZnVu+7WOrnbpmt3OiMkSeZrtB\nOrkBDQRN+NyIAQgArRZ2YGzUj5dXOVIWgZ1/QFpyfx/cG/293WjRE4Wt2e4SxMf2\nV0dcVCqWwT0+a79Wbausv4bStD4SkwDmu0Jf3z5ERzrr7oZwP0PMsIlM5zT6XSsr\n6UUneB3UXX7MrEqVogVhRM0ORIaK/oRwMXr7K6xVT+bCBP3/p66kHtY1ZpfEzTEX\nimBsN3GqoewBHYIneJKBtHE7uzdzw3O5p5dXqoj5foxGi9R1J15vAmt5pI68HJeX\nP6ktvXbX2Iu7VDNoCvRXM9+ntyJtsXCjNXg4pTGHS/XO4nm2db4FUZOBcVMb1vCc\nVtFjLTcbCqJqpoJWUtsLcNqDqMHOQDpe6KTNTQARAQABiQEfBBgBAgAJBQJN+NyI\nAhsMAAoJEIdoGx7jCA5FrR8IANnOF3PUj1TbRcwV6RoWmHsFQHrPmM8ogXia1Lsv\njE1iEWoC+muvKh6Oydf90k6ZslS7rdDnp2qzYY8W/TiDkxP+fvsZ4mMi1Y0F+3ty\n1jzWhcsnB2VrJSiavxEXk0tKPrNv4EUGWG6wHsC9TBj37If+nrMyim94VHvI0eHm\nX8yMlN4O3HfmgD9CbJdUxueP3e31OIYuwh/6F7GII8TNEVHU/8vh/mQcCxppNbc+\nboff+kIsoa/TAMLwtJoSrX1nXm0K3vZePRLnIgmwVzdkOIkaRJUG2tSQFvkfhvtE\nLhnkL5l4MO0wrUds0UWRwa3d7j/P2ExrqXdlLmEzrifWyEQ=\n=hUJn\n-----END PGP PUBLIC KEY BLOCK-----\n')
        secret_key = StringIO('-----BEGIN PGP PRIVATE KEY BLOCK-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\nlQOYBE343IgBCADwzPW7kmKb2bjB+UU+1ER/ABMZspvtoZMPusUw7bk6coXHF/0W\nu1K/hSYeX9xaGOfOQw41r/g13MoR9dsL6L84RLiisf38rRoBZt+d5bCbZA5Xo801\n2PeoBoGo6u5oOYKAFLMvrUitPiiE0IT/oQTfC4YUrLN4A+9W0QZruPGIpIXwmZXr\nL0zsqYfNqIN0ompeJenVpKpvm3loJ/zfK7R3EJ3hsv6nkUmWCFsP1Pw3UV1YuCmw\nMkdn1U7DaOql1WjXgj9ABQDJrun2TGsqrSRzBODtHKA/uOX0K3VfKBU8VZo3dXUm\n1Q4ZeZC39L9qJGTH8TQYlwBLe1yAOp+vx7QJABEBAAEAB/0RJTbV991SOtVfPQVu\nLM+tD0SiOXJwIBIINlngsFHWVIiBSDb6uF8dneMR70IRnuEFHFyAUXA7PZDxvcSu\nphAqIdKCWxQPkAULAS0o4U2K3ZFGh4uOqvfZ8eSnh1rETFv7Yf3u23K89cZiy99n\nEtWgSqzC/2z5PaZ7/alsYCBqhHuyd4Phaud7qv7FTz8mFrCf+CCY+D08wbnZBu4g\nN9tBwoxT/UKRfv3nghIh9v+3qWfBEFGhrYbt92XKFbHOQeATZz8AGIv1eqN/+ZQY\noYmvVfO3GkrWaRoPeJNLqSDEn/45O1Uh9MJ4mQclXqB0QzMShle8uusHxIeJSQsR\nz//VBAD11WS7qSgCeiHR+4jDzrrlb2snnA2bfDToEomDxd/n8xm7nJWdkNfJ2BCw\nKvnxYVxjFNAwkKJGRajzALBLzRVO+K9NtSLiddv5zv+UNdgsKuE8tD7Jqxd/IbWw\nAimCtL8osnJ+r9dvL+NyjkAT6l/NdEbLXGrBaMeTfSgl2cBOOwQA+sJIh1R5PiCK\nnLIs9pm3PSy3w92Peelq/x/+0aebTZaJUk2ou3oCvB3druDqrUeaopuuCc0drV7C\nLdoey8x/T2ZGzmT2af9qNaD6ScTimDodXcJdwlpobhZTKpsE4EyywpLXtlWte1x0\n1Mq3llQsIdRdf3GLS+L207hWgKDiDosD/0SyOBO/IBDteeEzeN2hNE3A8oeVbvRS\nXrS/3uj6oKmlWUBORYP8ptUrXPoVPmNz2y4GO+OysFtfct3Yqb+Sb/52SXMOHTox\n2oLW08tkzfkDArU5aauMEPmyutGyJ+hGo7fsuLXzXR8OPw4yZJdzG1tRlP2TTKmq\nFx8G/Ik6bN4zTYK0JEJhemFhciBUZXN0IEtleSA8YmF6YWFyQGV4YW1wbGUuY29t\nPokBOAQTAQIAIgUCTfjciAIbAwYLCQgHAwIGFQgCCQoLBBYCAwECHgECF4AACgkQ\nh2gbHuMIDkWJUggAwj537fH6WW+GGLA5onys2hZmXUq/tU+L92bjQoRY4fmsQpk/\nFUVPUf+NQ0v1gkxx4BTfyYewaj5G6L8cvqW2jj7UiJd8z9gTRxWTnYwfR/w5PGmx\nfJsBfEUKWsccrPQdOXAhwu0fjYIVk4nqgswaIOAZIwe5Vsfs36uSS7p8RQHAZXLX\ntTOn3KcXHaxu83w6nc4zkWRovGJ9isBN3haO2qEa0mYiAfDpz40CGtb8N/TQHF3X\ncw8rJcxpg6RF3jMtWQnzbVJFp13it00R3LqWo/r3RII3Ii3z2yARlg6D+5hVOrFB\nV8jFLkff1R2ZnVu+7WOrnbpmt3OiMkSeZrtBOp0DlwRN+NyIAQgArRZ2YGzUj5dX\nOVIWgZ1/QFpyfx/cG/293WjRE4Wt2e4SxMf2V0dcVCqWwT0+a79Wbausv4bStD4S\nkwDmu0Jf3z5ERzrr7oZwP0PMsIlM5zT6XSsr6UUneB3UXX7MrEqVogVhRM0ORIaK\n/oRwMXr7K6xVT+bCBP3/p66kHtY1ZpfEzTEXimBsN3GqoewBHYIneJKBtHE7uzdz\nw3O5p5dXqoj5foxGi9R1J15vAmt5pI68HJeXP6ktvXbX2Iu7VDNoCvRXM9+ntyJt\nsXCjNXg4pTGHS/XO4nm2db4FUZOBcVMb1vCcVtFjLTcbCqJqpoJWUtsLcNqDqMHO\nQDpe6KTNTQARAQABAAf1EfceUlGLvoA/+yDTNTMjuPfzfKwbB/FOVfX44g3Za1eT\nv7RvSuj4rFYIdE9UvZEei/pqPOSc+hhSsKZCulGXD5TUpf3AyG7ipWU/kID46Csp\n0V08DPpFHnuw/N6+qNo5iSnhN9U1XMLjYT5d1HvKur26r2vWbmUTSJ1qIluHL2fT\nR1pKYYLuoff4MIjZ01Hawq72jjor+dLBmMWveHpq4XNp+vQ4x8aFnY9ozufon0nM\nuRSJRlQjDNB274tvUbmDFP+nzNbqF1nBTZ6FTdH/iKVNbytiYF7Hbat8GWVZqY1u\nCZr7BklpIVWlk62ll0psMIPVyANi7YT332LLqYmBBADJKTx2dariG/kWU2W/9VEO\n2VZpqsqazAxOoFEIOpcOlByhhyw5g0IKu0UyzHkhoCje0cWxpdSBFG432b8zL0AT\nZ0RycfUG7Sgp9CpY1h8Cc/HbBa8xo1fSM7zplPQrHBqUzlVVBq6HOkUq+7qsPFWc\nRRie95VsDmIMKQKPJHeYHQQA3EYGit+QHV0dccAInghEsf/mq8Gfnvo6HPYhWcDC\nDTM39NhNlnl1WkTFCd2TWc+TWQ4KlRsh6bMjUpNa2qjrUl90fLekbogcxxMhcwa6\nxgzEANZfwqdY0u3aB/CyZ6odfThwcAoeqoMpw34CfeKEroubpi2n8wKByrN2MQXJ\n4vEEAJbXZOqgAcFAFBUVb5mVT0s2lJMagZFPdhRJz2bttz01s/B8aca6CrDpFRjT\n03zRFUZjwDYqZDWBC181dCE9yla4OkWd5QyRKSS2EE02KEYqRzT0RngQn7s4AW2r\n326up3Jhleln3hgD4Kk3V3KHmyK8zqZA0qWzry4Vl2jjkbnAPB2JAR8EGAECAAkF\nAk343IgCGwwACgkQh2gbHuMIDkWtHwgA2c4Xc9SPVNtFzBXpGhaYewVAes+YzyiB\neJrUuy+MTWIRagL6a68qHo7J1/3STpmyVLut0OenarNhjxb9OIOTE/5++xniYyLV\njQX7e3LWPNaFyycHZWslKJq/EReTS0o+s2/gRQZYbrAewL1MGPfsh/6eszKKb3hU\ne8jR4eZfzIyU3g7cd+aAP0Jsl1TG54/d7fU4hi7CH/oXsYgjxM0RUdT/y+H+ZBwL\nGmk1tz5uh9/6Qiyhr9MAwvC0mhKtfWdebQre9l49EuciCbBXN2Q4iRpElQba1JAW\n+R+G+0QuGeQvmXgw7TCtR2zRRZHBrd3uP8/YTGupd2UuYTOuJ9bIRA==\n=LXn0\n-----END PGP PRIVATE KEY BLOCK-----\n')
        revoked_key = StringIO('-----BEGIN PGP PUBLIC KEY BLOCK-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\nmI0ETjlW5gEEAOb/6P+TVM59E897wRtatxys2BhsHCXM4T7xjIiANfDwejDdifqh\ntluTfSJLLxPembtrrEjux1C0AJgc+f0MIfsc3Pr3eFJzKB2ot/1IVG1/1KnA0zt3\nW2xPT3lRib27WJ9Fag+dMtQaIzgJ7/n2DFxsFZ33FD2kxrEXB2exGg6FABEBAAGI\npgQgAQIAEAUCTjlXkAkdAHJldm9rZWQACgkQjs6dvEpb0cQPHAP/Wi9rbx0e+1Sf\nziGgyVdr3m3A6uvze5oXKVgFRbGRUYSH4/I8GW0W9x4TcRg9h+YaQ8NUdADr9kNE\ntKAljLqYA5qdqSfYuaij1M++Xj+KUZ359R74sHuQqwnRy1XXQNfRs/QpXA7vLdds\nrjg+pbWuXO92TZJUdnqtWW+VEyZBsPy0G3Rlc3Qga2V5IDx0ZXN0QGV4YW1wbGUu\nY29tPoi4BBMBAgAiBQJOOVbmAhsDBgsJCAcDAgYVCAIJCgsEFgIDAQIeAQIXgAAK\nCRCOzp28SlvRxNWzA/42WVmI0b+6mF/imEOlY1TiyvrcpK250rkSDsCtL4lOwy7G\nantZhpgNfnXRd/ySfsS3EB6dpOWgOSxGRvWQhA+vxBT9BYNk49qd3JIrSaSWpR12\nrET8qO1rEQQFWsw03CxTGujxGlmEO+a1yguRXp2UWaY7FngcQmD+8q7BUIVm7riN\nBE45VuYBBADTEH2jHTjNCc5CMOhea6EJTrkx3upcEqB2oyhWeSWJiBGOxlcddsjo\n3J3/EmBB8kK1hM9TidD3SG64x1N287lg8ELJBlKv+pQVyxohGJ1u/THgpTDMMQcL\nluG5rAHQGSfyzKTiOnaTyBYg3M/nzgUOU9dKEFB0EA3tjUXFOT+r3wARAQABiJ8E\nGAECAAkFAk45VuYCGwwACgkQjs6dvEpb0cRSLQP/fzCWX2lXwlwWiVF8BOPF7o9z\nicHErc7/X17RGb4qj1kVf+UkRdUWJrbEVh4h6MncBIuA70WsYogiw+Kz/0LCtQAR\nYUJsPy/EL++OKPH1aFasOdTxwkTka85+RdYqhP1+z/aYLFMWq6mRFI+o6x2k5mGi\n7dMv2kKTJPoXUpiXJbg=\n=hLYO\n-----END PGP PUBLIC KEY BLOCK-----\n')
        expired_key = StringIO('-----BEGIN PGP PUBLIC KEY BLOCK-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\nmI0ETjZ6PAEEALkR4GcFQidCCxV7pgQwQd5MZua0YO2l92fVqHX+PhnZ6egCLKdD\n2bWlMUd6MLPF3FlRL7BBAxvW/DazkBOp7ljsnpMpptEzY49Uem1irYLYiVb9zK96\n0sQZzFxFkfEYetQEXC68mIck8tbySOX5NAOw++3jFm3J7dsU1R3XtYzRABEBAAG0\nG3Rlc3Qga2V5IDx0ZXN0QGV4YW1wbGUuY29tPoi+BBMBAgAoBQJONno8AhsDBQkA\nAVGABgsJCAcDAgYVCAIJCgsEFgIDAQIeAQIXgAAKCRAc4m97T40VEz+DA/9PBphG\nYp9cHVaHSfTUKGTGgIbvRe60sFNpDCYZeAGDrygOMuI8MNzbVpwefRBFHVPx7jWd\nrrYMsLkcsNUS9D0baU+0D/qp7JVg7ZSQtG0O6IG4eTZhibteY1fu0+unlXmg9NHx\n5VvhwzBiJDYji00M2p/CZEMiYFUuy76CsxUpN7iNBE42ejwBBACkv2/mX7IPQg0C\nA3KSrJsJv+sdvKm4b4xuI4OwagwTIVz4KlTqV4IBrVjSBfwyMXucXz0bTW85qjgA\n+n67td8vyjYYZUEz1uY9lSquQQDnAN0txL3cLHZXWiWOkmzZVddQtlflK2a/J9o0\nQkHPVUm+hc4l64dIzStrNl2S66fAvQARAQABiKUEGAECAA8FAk42ejwCGwwFCQAB\nUYAACgkQHOJve0+NFROEYQP/epg+o8iBs31hkSERyZjrRR66LpywezWj30Rn/3mX\nFzi9HkF4xLemWOzdNt9C5PYrOep85PQg8haEjknxVjZFS0ikT1h3OWk/TF1ZrLVm\nWzyX8DaHQEjKpLJJjXcAbTiZBNMk0QaVC9RvIeHpCf3n3DC49DdjsPJRMKOn8KDi\nkRk=\n=p0gt\n-----END PGP PUBLIC KEY BLOCK-----\n')
        context.import_(key)
        context.import_(secret_key)
        context.import_(revoked_key)
        context.import_(expired_key)

    def test_verify_untrusted_but_accepted(self):
        if False:
            return 10
        self.requireFeature(features.gpgme)
        self.import_keys()
        content = '-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\nbazaar-ng testament short form 1\nrevision-id: amy@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\niQEcBAEBAgAGBQJN+ekFAAoJEIdoGx7jCA5FGtEH/i+XxJRvqU6wdBtLVrGBMAGk\nFZ5VP+KyXYtymSbgSstj/vM12NeMIeFs3xGnNnYuX1MIcY6We5TKtCH0epY6ym5+\n6g2Q2QpQ5/sT2d0mWzR0K4uVngmxVQaXTdk5PdZ40O7ULeDLW6CxzxMHyUL1rsIx\n7UBUTBh1O/1n3ZfD99hUkm3hVcnsN90uTKH59zV9NWwArU0cug60+5eDKJhSJDbG\nrIwlqbFAjDZ7L/48e+IaYIJwBZFzMBpJKdCxzALLtauMf+KK8hGiL2hrRbWm7ty6\nNgxfkMYOB4rDPdSstT35N+5uBG3n/UzjxHssi0svMfVETYYX40y57dm2eZQXFp8=\n=iwsn\n-----END PGP SIGNATURE-----\n'
        plain = 'bazaar-ng testament short form 1\nrevision-id: amy@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n'
        my_gpg = gpg.GPGStrategy(FakeConfig())
        my_gpg.set_acceptable_keys('bazaar@example.com')
        self.assertEqual((gpg.SIGNATURE_VALID, None), my_gpg.verify(content, plain))

    def test_verify_unacceptable_key(self):
        if False:
            return 10
        self.requireFeature(features.gpgme)
        self.import_keys()
        content = '-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\nbazaar-ng testament short form 1\nrevision-id: amy@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\niQEcBAEBAgAGBQJN+ekFAAoJEIdoGx7jCA5FGtEH/i+XxJRvqU6wdBtLVrGBMAGk\nFZ5VP+KyXYtymSbgSstj/vM12NeMIeFs3xGnNnYuX1MIcY6We5TKtCH0epY6ym5+\n6g2Q2QpQ5/sT2d0mWzR0K4uVngmxVQaXTdk5PdZ40O7ULeDLW6CxzxMHyUL1rsIx\n7UBUTBh1O/1n3ZfD99hUkm3hVcnsN90uTKH59zV9NWwArU0cug60+5eDKJhSJDbG\nrIwlqbFAjDZ7L/48e+IaYIJwBZFzMBpJKdCxzALLtauMf+KK8hGiL2hrRbWm7ty6\nNgxfkMYOB4rDPdSstT35N+5uBG3n/UzjxHssi0svMfVETYYX40y57dm2eZQXFp8=\n=iwsn\n-----END PGP SIGNATURE-----\n'
        plain = 'bazaar-ng testament short form 1\nrevision-id: amy@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n'
        my_gpg = gpg.GPGStrategy(FakeConfig())
        my_gpg.set_acceptable_keys('foo@example.com')
        self.assertEqual((gpg.SIGNATURE_KEY_MISSING, u'E3080E45'), my_gpg.verify(content, plain))

    def test_verify_valid_but_untrusted(self):
        if False:
            for i in range(10):
                print('nop')
        self.requireFeature(features.gpgme)
        self.import_keys()
        content = '-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\nbazaar-ng testament short form 1\nrevision-id: amy@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\niQEcBAEBAgAGBQJN+ekFAAoJEIdoGx7jCA5FGtEH/i+XxJRvqU6wdBtLVrGBMAGk\nFZ5VP+KyXYtymSbgSstj/vM12NeMIeFs3xGnNnYuX1MIcY6We5TKtCH0epY6ym5+\n6g2Q2QpQ5/sT2d0mWzR0K4uVngmxVQaXTdk5PdZ40O7ULeDLW6CxzxMHyUL1rsIx\n7UBUTBh1O/1n3ZfD99hUkm3hVcnsN90uTKH59zV9NWwArU0cug60+5eDKJhSJDbG\nrIwlqbFAjDZ7L/48e+IaYIJwBZFzMBpJKdCxzALLtauMf+KK8hGiL2hrRbWm7ty6\nNgxfkMYOB4rDPdSstT35N+5uBG3n/UzjxHssi0svMfVETYYX40y57dm2eZQXFp8=\n=iwsn\n-----END PGP SIGNATURE-----\n'
        plain = 'bazaar-ng testament short form 1\nrevision-id: amy@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n'
        my_gpg = gpg.GPGStrategy(FakeConfig())
        self.assertEqual((gpg.SIGNATURE_NOT_VALID, None), my_gpg.verify(content, plain))

    def test_verify_bad_testament(self):
        if False:
            for i in range(10):
                print('nop')
        self.requireFeature(features.gpgme)
        self.import_keys()
        content = '-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\nbazaar-ng testament short form 1\nrevision-id: amy@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\niQEcBAEBAgAGBQJN+ekFAAoJEIdoGx7jCA5FGtEH/i+XxJRvqU6wdBtLVrGBMAGk\nFZ5VP+KyXYtymSbgSstj/vM12NeMIeFs3xGnNnYuX1MIcY6We5TKtCH0epY6ym5+\n6g2Q2QpQ5/sT2d0mWzR0K4uVngmxVQaXTdk5PdZ40O7ULeDLW6CxzxMHyUL1rsIx\n7UBUTBh1O/1n3ZfD99hUkm3hVcnsN90uTKH59zV9NWwArU0cug60+5eDKJhSJDbG\nrIwlqbFAjDZ7L/48e+IaYIJwBZFzMBpJKdCxzALLtauMf+KK8hGiL2hrRbWm7ty6\nNgxfkMYOB4rDPdSstT35N+5uBG3n/UzjxHssi0svMfVETYYX40y57dm2eZQXFp8=\n=iwsn\n-----END PGP SIGNATURE-----\n'
        plain = 'bazaar-ng testament short form 1\nrevision-id: doctor@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n'
        my_gpg = gpg.GPGStrategy(FakeConfig())
        my_gpg.set_acceptable_keys('bazaar@example.com')
        self.assertEqual((gpg.SIGNATURE_NOT_VALID, None), my_gpg.verify(content, plain))

    def test_verify_revoked_signature(self):
        if False:
            for i in range(10):
                print('nop')
        self.requireFeature(features.gpgme)
        self.import_keys()
        content = '-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\nasdf\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\niJwEAQECAAYFAk45V18ACgkQjs6dvEpb0cSIZQP/eOGTXGPlrNwvDkcX2d8O///I\necB4sUIUEpv1XAk1MkNu58lsjjK72lRaLusEGqd7HwrFmpxVeVs0oWLg23PNPCFs\nyJBID9ma+VxFVPtkEFnrc1R72sBJLfBcTxMkwVTC8eeznjdtn+cg+aLkxbPdrGnr\nJFA6kUIJU2w9LU/b88Y=\n=UuRX\n-----END PGP SIGNATURE-----\n'
        plain = 'asdf\n'
        my_gpg = gpg.GPGStrategy(FakeConfig())
        my_gpg.set_acceptable_keys('test@example.com')
        self.assertEqual((gpg.SIGNATURE_NOT_VALID, None), my_gpg.verify(content, plain))

    def test_verify_invalid(self):
        if False:
            while True:
                i = 10
        self.requireFeature(features.gpgme)
        self.import_keys()
        content = '-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\nbazaar-ng testament short form 1\nrevision-id: amy@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\niEYEARECAAYFAk33gYsACgkQpQbm1N1NUIhiDACglOuQDlnSF4NxfHSkN/zrmFy8\nnswAoNGXAVuR9ONasAKIGBNUE0b+lols\n=SOuC\n-----END PGP SIGNATURE-----\n'
        plain = 'bazaar-ng testament short form 1\nrevision-id: amy@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n'
        my_gpg = gpg.GPGStrategy(FakeConfig())
        self.assertEqual((gpg.SIGNATURE_NOT_VALID, None), my_gpg.verify(content, plain))

    def test_verify_expired_but_valid(self):
        if False:
            while True:
                i = 10
        self.requireFeature(features.gpgme)
        self.import_keys()
        content = '-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n \nbazaar-ng testament short form 1\nrevision-id: test@example.com-20110801100657-f1dr1nompeex723z\nsha1: 59ab434be4c2d5d646dee84f514aa09e1b72feeb\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.10 (GNU/Linux)\n \niJwEAQECAAYFAk42esUACgkQHOJve0+NFRPc5wP7BoZkzBU8JaHMLv/LmqLr0sUz\nzuE51ofZZ19L7KVtQWsOi4jFy0fi4A5TFwO8u9SOfoREGvkw292Uty9subSouK5/\nmFmDOYPQ+O83zWgYZsBmMJWYDZ+X9I6XXZSbPtV/7XyTjaxtl5uRnDVJjg+AzKvD\ndTp8VatVVrwuvzOPDVc=\n=uHen\n-----END PGP SIGNATURE-----\n'
        plain = 'bazaar-ng testament short form 1\nrevision-id: test@example.com-20110801100657-f1dr1nompeex723z\nsha1: 59ab434be4c2d5d646dee84f514aa09e1b72feeb\n'
        my_gpg = gpg.GPGStrategy(FakeConfig())
        self.assertEqual((gpg.SIGNATURE_EXPIRED, u'4F8D1513'), my_gpg.verify(content, plain))

    def test_verify_unknown_key(self):
        if False:
            return 10
        self.requireFeature(features.gpgme)
        self.import_keys()
        content = '-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\nasdf\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\niQEcBAEBAgAGBQJOORKwAAoJENf6AkFdUeVvJDYH/1Cz+AJn1Jvy5n64o+0fZ5Ow\nY7UQb4QQTIOV7jI7n4hv/yBzuHrtImFzYvQl/o2Ezzi8B8L5gZtQy+xCUF+Q8iWs\ngytZ5JUtSze7hDZo1NUl4etjoRGYqRfrUcvE2LkVH2dFbDGyyQfVmoeSHa5akuuP\nQZmyg2F983rACVIpGvsqTH6RcBdvE9vx68lugeKQA8ArDn39/74FBFipFzrXSPij\neKFpl+yZmIb3g6HkPIC8o4j/tMvc37xF1OG5sBu8FT0+FC+VgY7vAblneDftAbyP\nsIODx4WcfJtjLG/qkRYqJ4gDHo0eMpTJSk2CWebajdm4b+JBrM1F9mgKuZFLruE=\n=RNR5\n-----END PGP SIGNATURE-----\n'
        plain = 'asdf\n'
        my_gpg = gpg.GPGStrategy(FakeConfig())
        self.assertEqual((gpg.SIGNATURE_KEY_MISSING, u'5D51E56F'), my_gpg.verify(content, plain))

    def test_set_acceptable_keys(self):
        if False:
            return 10
        self.requireFeature(features.gpgme)
        self.import_keys()
        my_gpg = gpg.GPGStrategy(FakeConfig())
        my_gpg.set_acceptable_keys('bazaar@example.com')
        self.assertEqual(my_gpg.acceptable_keys, [u'B5DEED5FCB15DAE6ECEF919587681B1EE3080E45'])

    def test_set_acceptable_keys_from_config(self):
        if False:
            return 10
        self.requireFeature(features.gpgme)
        self.import_keys()
        my_gpg = gpg.GPGStrategy(FakeConfig('acceptable_keys=bazaar@example.com'))
        my_gpg.set_acceptable_keys(None)
        self.assertEqual(my_gpg.acceptable_keys, [u'B5DEED5FCB15DAE6ECEF919587681B1EE3080E45'])

    def test_set_acceptable_keys_unknown(self):
        if False:
            return 10
        self.requireFeature(features.gpgme)
        my_gpg = gpg.GPGStrategy(FakeConfig())
        self.notes = []

        def note(*args):
            if False:
                print('Hello World!')
            self.notes.append(args[0] % args[1:])
        self.overrideAttr(trace, 'note', note)
        my_gpg.set_acceptable_keys('unknown')
        self.assertEqual(my_gpg.acceptable_keys, [])
        self.assertEqual(self.notes, ['No GnuPG key results for pattern: unknown'])

class TestDisabled(TestCase):

    def test_sign(self):
        if False:
            while True:
                i = 10
        self.assertRaises(errors.SigningFailed, gpg.DisabledGPGStrategy(None).sign, 'content')

    def test_verify(self):
        if False:
            return 10
        self.assertRaises(errors.SignatureVerificationFailed, gpg.DisabledGPGStrategy(None).verify, 'content', 'testament')