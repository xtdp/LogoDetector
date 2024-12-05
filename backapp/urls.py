from django.contrib import admin
from django.urls import path, include
from . import views



urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_image, name='upload_image'),
    path('result/<int:pk>/', views.result, name='result'),
    path('home/', views.index, name='index'),
    path('contactus/', views.contactus, name='contactus'),
    path('aboutus/', views.aboutus, name='aboutus'),
    path('terms/', views.terms, name='terms'),
]