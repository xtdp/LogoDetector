from django import forms
from .models import LogoImage

class LogoImageForm(forms.ModelForm):
    class Meta:
        model = LogoImage
        fields = ['image']
