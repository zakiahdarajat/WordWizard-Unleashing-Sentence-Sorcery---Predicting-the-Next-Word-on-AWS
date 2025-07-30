from django.urls import path
from django.conf.urls.static import static
from .views import handler

app_name = 'polls'

urlpatterns = [
    path('', handler, name='homepage'),  # Root URL
    path('home/', handler, name='home'),  # Alternative URL
    path('predict/', handler, name='predict'),  # Semantic URL
]