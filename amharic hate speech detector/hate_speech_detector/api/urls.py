from django.urls import path
from .views import PredictView, home

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('', home, name='home'),
]