def test_issue69(db):
    if False:
        for i in range(10):
            print('nop')
    db.query('CREATE table users (id text)')
    db.query('SELECT * FROM users WHERE id = :user', user="Te'ArnaLambert")