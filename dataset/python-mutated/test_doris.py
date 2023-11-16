from tests.dialects.test_dialect import Validator

class TestDoris(Validator):
    dialect = 'doris'

    def test_identity(self):
        if False:
            for i in range(10):
                print('nop')
        self.validate_identity('COALECSE(a, b, c, d)')
        self.validate_identity('SELECT CAST(`a`.`b` AS INT) FROM foo')
        self.validate_identity('SELECT APPROX_COUNT_DISTINCT(a) FROM x')

    def test_time(self):
        if False:
            i = 10
            return i + 15
        self.validate_identity("TIMESTAMP('2022-01-01')")

    def test_regex(self):
        if False:
            i = 10
            return i + 15
        self.validate_all("SELECT REGEXP_LIKE(abc, '%foo%')", write={'doris': "SELECT REGEXP(abc, '%foo%')"})