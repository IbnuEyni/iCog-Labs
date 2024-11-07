from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=255)
    file_type = models.CharField(max_length=50)
    content = models.TextField()

    def __str__(self):
        return self.title
