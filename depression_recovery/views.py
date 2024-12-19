import base64
import datetime
import json
import uuid
import pytz
import os
import pandas as pd
import django.contrib.auth
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableSequence
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils.timezone import now, localtime
import depression_recovery.serializer
from depression_recovery.Mlmodels.AllAiModels import Sentimental, VideoModel
from .models import *
from datetime import date
from django.shortcuts import render, redirect
from django.contrib.auth import login ,authenticate
from .forms import DoctorSignupForm, PatientSignupForm
from .models import DoctorData, PatientData , ChatLog , WeablesData
from datetime import datetime, timedelta
from django.conf import settings
from pathlib import Path
import csv

from django.db.models.expressions import RawSQL
import time
moodByText = [ 'neutral' ]


class AIModel():
    def __init__(self):
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key="AIzaSyBwd6F4ufCjZZBOunKWWflyb94wcyLxD2A"
            , temperature=0.9

        )
        self.sentimental = Sentimental()
        self.videoModel = VideoModel()
        prompt_template = """
        You are a virtual psychiatrist helping users manage their mental health.
        Act as a human dont provide other numbers in text (Dont provide the phone number of anyone )
        You should be able to talk and convence users 
        And Your country is India
        And emergency contact number is 112
        Below is the conversation history and the current input:
        Conversation history:{conversation_history}
        User's input: {user_input}
        Respond with a short and empathetic text message, offering simple and helpful advice and make it realistic.
        """
        prompt = PromptTemplate(
            input_variables=[ "conversation_history", "emotion_state", "user_input" ],
            template=prompt_template
        )
        chain = prompt | llm

        self.conversation_chain = RunnableSequence(
            chain
        )
        self.conversation_history = ""
        self.messaging_history = [ {
            "role": "Assistant", "content": "Hello, how can I help you today?"
        } ]

    def add_to_message_history(self, user_input, response):
        self.messaging_history.append({"role": "user", "content": user_input})
        self.messaging_history.append({"role": "assistant", "content": response})

    def add_to_history(self, user_input, emotion_state, response):
        self.conversation_history += f"User's input: {user_input}\n"
        self.conversation_history += f"Emotion state: {emotion_state}\n"
        self.conversation_history += f"Assistant's response: {response}\n"

    def chat(self, user_input, emotion_state):
        response = self.conversation_chain.invoke({"conversation_history": self.conversation_history,
                                                   "user_input": user_input,
                                                   "emotion_state": emotion_state})
        self.add_to_history(user_input, emotion_state, response.content)
        self.add_to_message_history(user_input, response.content)
        return response.content


global_ai_model = AIModel()

#Upload Images
def upload_image(request):

    if request.method == 'POST':
        try:
            # Extract the base64 image data
            data = json.loads(request.body)
            data = data.get('image')
            if data.startswith('data:image'):
                header, encoded = data.split(',', 1)
                image_data = base64.b64decode(encoded)
                image = Image.open(BytesIO(image_data))
                image_np = np.array(image)
                print(image.mode)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                mood = global_ai_model.videoModel.predict_emotion(image_np)
                print(mood)
                if mood is None:
                    mood = "neutral"
                try:
                    MoodVideo.objects.create(mood=mood, patient=PatientData.objects.get(user=request.user),timestamp = now())
                except Exception as e:
                    print(e)
                    return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
                return JsonResponse({'status': 'success', 'mood': mood})
            else:
                return JsonResponse({'status': 'error', 'message': 'Invalid image format'}, status=400)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return render(request, "upload.html")


def process_image(image):
    cv2.imshow(f"image", image)
    cv2.waitKey(0)

    print(image)
    return "happy"


@csrf_exempt
def chatBot(request):
    # if 'session_id' not in request.session or request.method == "GET":
    #     # Start a new session on refresh or first visit
    #     request.session[ 'session_id' ] = str(uuid.uuid4())  # Unique session ID
    #     request.session[ 'conversation_history' ] = ""
    #     request.session[ 'messaging_history' ] = [ {
    #         "role": "assistant", "content": "Hello, how can I help you today?"
    #     } ]
    #
    # conversation_history = request.session[ 'conversation_history' ]
    # messaging_history = request.session[ 'messaging_history' ]
    # aimodel = global_ai_model
    # request.session.flush()  # Clears session data
    # request.session[ 'session_id' ] = str(uuid.uuid4())  # Generate a new session ID
    request.session[ 'conversation_history' ] = ""
    request.session[ 'messaging_history' ] = [ {
       "role": "assistant", "content": "Hello, how can I help you today?"
    } ]
    # print("hello")
    # for i ,j in request.session.items():
    #     print(i,j)
    # message = aimodel.messaging_history
    # if request.method == "POST":
    #     print("Request " ,request.POST.get("textInput"))
    #     global_ai_model.conversation_history = conversation_history
    #     user_input = request.POST.get("textInput")
    #     response = global_ai_model.chat(user_input, "happy")
    #     request.session[ 'conversation_history' ] = global_ai_model.conversation_history
    #     request.session[ 'messaging_history' ] = global_ai_model.messaging_history
    #     message = aimodel.messaging_history
    #     return Response({"message": response})
    return render(request, "ChatBot.html", )


@csrf_exempt
@api_view([ 'POST' ])
def chat_bot(request):
    if not request.user.is_authenticated:
        return Response({"message": "Please login to continue"})

    if request.method == "POST":
        data = request.data
        user_input = data.get("textInput")
        print("Request received: ", user_input)

        # Check session variables
        conversation_history = request.session.get('conversation_history', [ ])
        messaging_history = request.session.get('messaging_history', [ ])

        aimodel = global_ai_model
        aimodel.conversation_history = conversation_history

        # Generate AI response and mood analysis
        response = aimodel.chat(user_input, moodByText[ -1 ])
        sentiment = aimodel.sentimental.predict_sentiment(user_input)
        moodByText.append(sentiment)
        print("Mood by text: ", moodByText)

        # Update session variables
        request.session[ 'conversation_history' ] = aimodel.conversation_history
        request.session[ 'messaging_history' ] = aimodel.messaging_history

        # Save ChatLog entry
        user = PatientData.objects.get(user=request.user)
        chat_log = ChatLog(
            timestamp=now(),
            user_input=user_input,
            bot_response=response,
            mood=sentiment,
            patient=user
        )
        chat_log.save()

        # Determine mood index for WeablesData
        mood_index = 1 if sentiment == "negative" else 10 if sentiment == "positive" else 5

        # Process CSV file and save WeablesData
        csv_file_path = os.path.join(settings.BASE_DIR,'depression_recovery', 'static', 'wearables.csv')
        try:
            with open(csv_file_path, 'r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    # Save each row as a WeablesData entry
                    weables = WeablesData(
                        timestamp=now(),
                        respiration_rate=float(row['rr']),
                        body_temperature=float(row['t']),
                        blood_oxygen=float(row[ 'bo']),
                        heart_rate=float(row[ 'hr' ]),
                        sleeping_hours=float(row[ 'hr' ]),
                        stress_level=float(row[ 'sl' ]),
                        mood_data=str(mood_index),
                        patient=user
                    )
                    weables.save()
        except FileNotFoundError:
            print(f"CSV file not found at {csv_file_path}")
        except Exception as e:
            print(f"Error processing CSV: {e}")

        # Return AI response
        return Response({"message": response})

    return Response({"message": "Hello, how can I help you today?"})


def getAllChats(request):
    seralizer = depression_recovery.serializer.ChatLogSerializer(ChatLog.objects.all(), many=True)
    return JsonResponse(seralizer.data, safe=False)


def Loginhome(request,user):
    if request.user.is_authenticated:
        user = user.lower()
        if user == "doctor":
            try:
                doctor_data = DoctorData.objects.get(user=request.user)
                patients = PatientData.objects.filter(doctor=doctor_data)
            except DoctorData.DoesNotExist:
                patients = []
            return render(request, "doctorHome.html"  , {'patients': patients})
        elif user == "patient":
            return render(request, "pateintHome.html",{'user': request.user})
    return redirect('/login')


@api_view([ 'GET' ])
@csrf_exempt
def GetPatientData(request):
    filter_date = now().date()
    print(filter_date)

    # Filter messages by today's date
    filtered_messages = ChatLog.objects.filter(timestamp__date=filter_date)
    serializer = depression_recovery.serializer.ChatLogSerializer(filtered_messages, many=True)
    data = serializer.data
    print(data)

    # Convert timestamps to local time and process data
    dataTime = [ ]
    for x in data:
        try:
            # Parse the timestamp including timezone offset
            timestamp = datetime.fromisoformat(x[ 'timestamp' ])
            local_timestamp = localtime(timestamp)  # Convert to local time

            # Extract time components and mood
            time_in_seconds = float(local_timestamp.strftime('%H%M%S')) / 10000
            mood_score = 1 if x[ 'mood' ] == "positive" else 0.5 if x[ 'mood' ] == "negative" else 0

            dataTime.append([ time_in_seconds, mood_score ])
        except ValueError as e:
            print(f"Error parsing timestamp: {x[ 'timestamp' ]}. Error: {e}")
            continue

    print(json.dumps(dataTime))
    return Response(dataTime)

def doctor(request):

    return render(request, "doctor.html")



@csrf_exempt
def doctor_signup(request):

    if request.method == 'POST':
        form = DoctorSignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Create a DoctorData entry for the user
            doctor =  DoctorData.objects.create(user=user)


            seralizer = depression_recovery.serializer.DoctorSerializer(doctor)
            login(request, user)

            return JsonResponse(seralizer.data)

    else:
        form = DoctorSignupForm()
    return render(request, 'doctor_signup.html', {'form': form})

# Patient Signup View
@csrf_exempt
def patient_signup(request):
    if request.method == 'POST':
        form = PatientSignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            doctor_id = form.cleaned_data.get('doctor_id')
            try:
                doctor = DoctorData.objects.get(doctorId=doctor_id)
                # Create a PatientData entry for the user
                patient = PatientData.objects.create(user=user, doctor=doctor)
                doctor.patients.add(patient)  # Link patient to the doctor
                doctor.save()
                login(request, user)  # Automatically log in after signup
                return redirect('/upload')  # Redirect to the dashboard
            except DoctorData.DoesNotExist:
                form.add_error('doctor_id', 'Doctor with this ID does not exist.')
    else:
        form = PatientSignupForm()
    return render(request, 'patient_signup.html', {'form': form})

@csrf_exempt
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username').strip()
        password = request.POST.get('password').strip()

        print(username,password)

        user = authenticate(request, username=username, password=password)
        print(user)
        if user is not None:
            login(request, user)
            if DoctorData.objects.filter(user=user).exists():
                return redirect('/home/Doctor')
            if PatientData.objects.filter(user=user).exists():
                return redirect('/home/Patient')
        else:
            return render(request, 'login.html')
    else:
        return render(request, 'login.html')


def home(request):
    return render(request,"Mainhome.html")