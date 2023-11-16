# Generated by Django 3.1.6 on 2021-02-16 18:04

from django.db import migrations, models
import django.db.models.deletion
import modelcluster.fields
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ("wagtailcore", "0059_apply_collection_ordering"),
        ("i18n", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="ClusterableTestModel",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "translation_key",
                    models.UUIDField(default=uuid.uuid4, editable=False),
                ),
                ("title", models.CharField(max_length=255)),
                (
                    "locale",
                    models.ForeignKey(
                        editable=False,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="+",
                        to="wagtailcore.locale",
                    ),
                ),
            ],
            options={
                "abstract": False,
                "unique_together": {("translation_key", "locale")},
            },
        ),
        migrations.CreateModel(
            name="ClusterableTestModelChild",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "sort_order",
                    models.IntegerField(blank=True, editable=False, null=True),
                ),
                ("field", models.TextField()),
                (
                    "parent",
                    modelcluster.fields.ParentalKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="children",
                        to="i18n.clusterabletestmodel",
                    ),
                ),
            ],
            options={
                "ordering": ["sort_order"],
                "abstract": False,
            },
        ),
        migrations.CreateModel(
            name="ClusterableTestModelTranslatableChild",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "translation_key",
                    models.UUIDField(default=uuid.uuid4, editable=False),
                ),
                (
                    "sort_order",
                    models.IntegerField(blank=True, editable=False, null=True),
                ),
                ("field", models.TextField()),
                (
                    "locale",
                    models.ForeignKey(
                        editable=False,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="+",
                        to="wagtailcore.locale",
                    ),
                ),
                (
                    "parent",
                    modelcluster.fields.ParentalKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="translatable_children",
                        to="i18n.clusterabletestmodel",
                    ),
                ),
            ],
            options={
                "ordering": ["sort_order"],
                "abstract": False,
                "unique_together": {("translation_key", "locale")},
            },
        ),
    ]
