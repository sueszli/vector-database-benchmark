# Generated by Django 3.1.8 on 2021-08-06 11:32

from django.db import migrations


def favorited_to_rating(apps, schema_editor):
    Photo = apps.get_model("api", "Photo")
    for photo in Photo.objects.all():
        photo.rating = 4 if photo.favorited else 0
        photo.save()


def rating_to_favorited(apps, schema_editor):
    Photo = apps.get_model("api", "Photo")
    for photo in Photo.objects.all():
        photo.favorited = photo.rating >= 4
        photo.save()


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0011_a_add_rating"),
    ]

    run_before = [("api", "0011_c_remove_favorited")]

    operations = [migrations.RunPython(favorited_to_rating, rating_to_favorited)]
