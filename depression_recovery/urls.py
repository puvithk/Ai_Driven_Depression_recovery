from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path('signup/doctor/',  views.doctor_signup, name='doctor_signup'),
    path('signup/patient/',  views.patient_signup, name='patient_signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('home/<str:user>', views.Loginhome, name='Login-home'),
    path("chatBot", views.chatBot, name="chatBot"),
    path("doctor", views.doctor, name="doctor"),
    path("api/upload", views.upload_image, name="home"),
    path("api/chatBot", views.chat_bot, name="chat_Bot"),
    path("api/doctor", views.GetPatientData, name="doctor"),

]
