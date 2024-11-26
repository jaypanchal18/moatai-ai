from django.urls import path
from .views import upload_with_predefined, query

urlpatterns = [
    path('upload-with-predefined/', upload_with_predefined, name='upload_with_predefined'),
    path('query/', query, name='query'),
]
