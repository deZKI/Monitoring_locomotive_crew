from django.core.validators import FileExtensionValidator
from django.db import models

from video.ai import detect_video

ERROR_CHOICES = [
    (1, 'Сидел в телефоне'),
    (2, 'Другая ошибка'),
]

class Video(models.Model):
    title = models.CharField(max_length=100)
    file = models.FileField(
        upload_to='video/',
        validators=[FileExtensionValidator(allowed_extensions=['mp4'])]
    )
    image = models.ImageField(upload_to='image/', blank=False)
    create_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        super().save(force_insert, force_update, using, update_fields)
        detect_video(self.file.path)




class TimeCodes(models.Model):
    video = models.ForeignKey(to=Video, on_delete=models.CASCADE)
    start = models.PositiveIntegerField(verbose_name='Начало в секундах')
    end = models.PositiveIntegerField(verbose_name='Конец в секундах')
    type = models.PositiveIntegerField(choices=ERROR_CHOICES)
