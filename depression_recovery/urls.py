from django.urls import path

from . import views

urlpatterns = [
    path("", views.upload_image, name="home"),
    path("home", views.chatBot, name="chatBot"),
    path("chatBot", views.chat_bot, name="chat_Bot")
]
