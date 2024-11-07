from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_document_page, name='upload_document_page'),
    path('upload_and_process_documents/', views.upload_and_process_documents, name='upload_and_process_documents'),
    path('', views.query_page, name='query_page'),
    path('process_query/', views.process_query, name='process_query'),
    path('documents/', views.document_list, name='list_documents_page'),
    path('delete_document/<int:document_id>/', views.delete_document, name='delete_document'),
] 
