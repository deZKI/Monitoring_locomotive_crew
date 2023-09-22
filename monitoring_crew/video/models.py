from django.core.validators import FileExtensionValidator
from django.db import models

ERROR_CHOICES = [
    (1, 'Сидел в телефоне'),
    (2, 'Другая ошибка'),
]


class Video(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField()
    image = models.ImageField(upload_to='image/')
    file = models.FileField(
        upload_to='video/',
        validators=[FileExtensionValidator(allowed_extensions=['mp4'])]
    )
    create_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class TimeCodes(models.Model):
    video = models.ForeignKey(to=Video, on_delete=models.CASCADE)
    start = models.PositiveIntegerField(verbose_name='Начало в секундах')
    end = models.PositiveIntegerField(verbose_name='Конец в секундах')
    type = models.PositiveIntegerField(choices=ERROR_CHOICES)
