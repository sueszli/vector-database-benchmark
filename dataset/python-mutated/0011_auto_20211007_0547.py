from django.db import migrations

def move_gift_card_tags_to_objects(apps, schema_editor):
    if False:
        while True:
            i = 10
    GiftCard = apps.get_model('giftcard', 'GiftCard')
    GiftCardTag = apps.get_model('giftcard', 'GiftCardTag')
    for card in GiftCard.objects.iterator():
        if card.tag:
            (tag, _created) = GiftCardTag.objects.get_or_create(name=card.tag)
            card.tags.add(tag)

class Migration(migrations.Migration):
    dependencies = [('giftcard', '0010_auto_20211007_0546')]
    operations = [migrations.RunPython(move_gift_card_tags_to_objects, migrations.RunPython.noop)]