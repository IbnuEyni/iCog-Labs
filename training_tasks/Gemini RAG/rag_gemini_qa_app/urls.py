
from django.contrib import admin
from django.urls import path

from rag_gemini_qa_app import views

urlpatterns = [
    path("", views.main ,name="upload"),

] 