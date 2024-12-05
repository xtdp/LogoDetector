from django.db import models

class LogoImage(models.Model):
    image = models.ImageField(upload_to='static/')
    is_fake = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)


