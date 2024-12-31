from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path('signup/doctor/',  views.doctor_signup, name='doctor_signup'),
    path('signup/patient/',  views.patient_signup, name='patient_signup'),
    path('findbecks/', views.findBecksindex, name='findbecks'),
    path('api/findbecks/', views.findbecks ,name='find_becks'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logoutOut, name='logout'),
    path('home/<str:user>', views.Loginhome, name='Login-home'),
    path("chatBot", views.chatBot, name="chatBot"),
    path("doctor/patient/<str:userId>", views.doctor, name="doctor"),
    path("api/upload", views.upload_image, name="home"),
    path("api/chatBot", views.chat_bot, name="chat_Bot"),
    path("api/doctor", views.GetPatientData, name="doctor"),
    path("api/doctor/video", views.getVideoMood , name="videoMood"),
    path("api/doctor/counts" , views.getVideoMoodCount , name="videoMoodCount"),
    path('api/doctor/sendnote', views.sendNote , name="sendNote"),
    path('api/becks/<str:patient_id>', views.getBecksScores , name="getBecksScores"),
    path('api/changeBeckTest/', views.changeBeckTest , name="changeBeckTest")
]
