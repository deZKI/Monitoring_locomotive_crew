# Generated by Django 4.2.5 on 2023-09-22 10:37

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=100)),
                ('description', models.TextField()),
                ('image', models.ImageField(upload_to='image/')),
                ('file', models.FileField(upload_to='video/', validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['mp4'])])),
                ('create_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='TimeCodes',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start', models.PositiveIntegerField(verbose_name='Начало')),
                ('end', models.PositiveIntegerField(verbose_name='Конец')),
                ('type', models.PositiveIntegerField(choices=[(1, 'Сидел в телефоне'), (2, 'Другая ошибка')])),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='video.video')),
            ],
        ),
    ]