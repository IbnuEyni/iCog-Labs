
from django.urls import path

from rag_app import views


urlpatterns = [
    path('', views.chat_view, name='chat'),
] 