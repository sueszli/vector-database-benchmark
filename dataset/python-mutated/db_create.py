import sqlalchemy
from nyaa import create_app, models
from nyaa.extensions import db
app = create_app('config')
NYAA_CATEGORIES = [('Anime', ['Anime Music Video', 'English-translated', 'Non-English-translated', 'Raw']), ('Audio', ['Lossless', 'Lossy']), ('Literature', ['English-translated', 'Non-English-translated', 'Raw']), ('Live Action', ['English-translated', 'Idol/Promotional Video', 'Non-English-translated', 'Raw']), ('Pictures', ['Graphics', 'Photos']), ('Software', ['Applications', 'Games'])]
SUKEBEI_CATEGORIES = [('Art', ['Anime', 'Doujinshi', 'Games', 'Manga', 'Pictures']), ('Real Life', ['Photobooks / Pictures', 'Videos'])]

def add_categories(categories, main_class, sub_class):
    if False:
        print('Hello World!')
    for (main_cat_name, sub_cat_names) in categories:
        main_cat = main_class(name=main_cat_name)
        for (i, sub_cat_name) in enumerate(sub_cat_names):
            sub_cat = sub_class(id=i + 1, name=sub_cat_name, main_category=main_cat)
        db.session.add(main_cat)
if __name__ == '__main__':
    with app.app_context():
        database_empty = False
        try:
            models.User.query.first()
        except (sqlalchemy.exc.ProgrammingError, sqlalchemy.exc.OperationalError):
            database_empty = True
        print('Creating all tables...')
        db.create_all()
        nyaa_category_test = models.NyaaMainCategory.query.first()
        if not nyaa_category_test:
            print('Adding Nyaa categories...')
            add_categories(NYAA_CATEGORIES, models.NyaaMainCategory, models.NyaaSubCategory)
        sukebei_category_test = models.SukebeiMainCategory.query.first()
        if not sukebei_category_test:
            print('Adding Sukebei categories...')
            add_categories(SUKEBEI_CATEGORIES, models.SukebeiMainCategory, models.SukebeiSubCategory)
        db.session.commit()
        if database_empty:
            print('Remember to run the following to mark the database up-to-date for Alembic:')
            print('./db_migrate.py stamp head')