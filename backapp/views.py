from django.shortcuts import render, redirect
from .forms import LogoImageForm
from .models import LogoImage
from .utils import preprocess_image 

def index(request):
    return render(request, 'index.html')

def upload_image(request):
    if request.method == 'POST':
        form = LogoImageForm(request.POST, request.FILES)
        if form.is_valid():
            logo_image = form.save()
            is_fake, confidence = preprocess_image(logo_image.image.path)
            logo_image.is_fake = is_fake
            logo_image.save()
            request.session['confidence'] = confidence
            return redirect('result', pk=logo_image.pk)
    else:
        form = LogoImageForm()
    return render(request, 'upload.html', {'form': form})

def result(request, pk):
    logo_image = LogoImage.objects.get(pk=pk)
    confidence = request.session.get('confidence', None)
    context = {
        'logo_image': logo_image,
        'is_fake': 'Fake' if logo_image.is_fake else 'Real',
        'confidence': confidence
    }
    return render(request, 'result.html', context)

def contactus(request):
    return render(request, 'contactus.html')

def aboutus(request):
    return render(request, 'aboutus.html')

def terms(request):
    return render(request, 'terms.html')
