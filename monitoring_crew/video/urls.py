from django.urls import path
from . import views


urlpatterns = [
    path('stream/<int:pk>/', views.get_streaming_video, name='stream'),
    path('<int:pk>/', views.get_video, name='video'),
    path('', views.get_list_video, name='home'),
    path('api/get_timecodes/<int:video_id>', views.get_timecodes)
]
