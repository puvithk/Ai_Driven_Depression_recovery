from rest_framework import serializers
from .models import *
import pytz
class ChatLogSerializer(serializers.ModelSerializer):


    class Meta:
        model = ChatLog
        fields = '__all__'

class DoctorSerializer(serializers.ModelSerializer):
    class Meta:
        model = DoctorData
        fields = '__all__'

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = PatientData
        fields = '__all__'

class MoodVideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = MoodVideo
        fields = '__all__'

class WeablesDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = WeablesData
        fields = '__all__'


class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = '__all__'


class BecksIndexSerializer(serializers.ModelSerializer):
    class Meta:
        model = BecksIndex
        fields = '__all__'