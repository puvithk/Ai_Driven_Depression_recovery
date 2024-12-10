import base64
import json
import uuid
from django.http import JsonResponse
from django.shortcuts import render
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
moodByText =[]
class AIModel():
    def __init__(self):
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key="AIzaSyBwd6F4ufCjZZBOunKWWflyb94wcyLxD2A"
            , temperature=0.9

        )
        prompt_template = """
        You are a virtual psychiatrist helping users manage their mental health.
        Act as a human dont provide other numbers in text (Dont provide the phone number of anyone )
        You should be able to talk and convence users 
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
        self.messaging_history = [{
        "role": "Assistant", "content": "Hello, how can I help you today?"
    }]

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
                mood = process_image(image_np)

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
    request.session.flush()  # Clears session data
    request.session[ 'session_id' ] = str(uuid.uuid4())  # Generate a new session ID
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
    return render(request, "home.html", )
@api_view(['POST'])
def chat_bot(request):
    data = json.loads(request.body)
    conversation_history = request.session[ 'conversation_history' ]
    messaging_history = request.session[ 'messaging_history' ]
    aimodel = global_ai_model
    message = aimodel.messaging_history
    if request.method == "POST":
        print(request.session[ 'conversation_history' ])
        print("Request " ,data.get("textInput"))
        global_ai_model.conversation_history = conversation_history
        user_input = data.get("textInput")
        response = global_ai_model.chat(user_input, moodByText[-1])
        request.session[ 'conversation_history' ] = global_ai_model.conversation_history
        request.session[ 'messaging_history' ] = global_ai_model.messaging_history
        message = aimodel.messaging_history
        return Response({"message": response})
    return Response({"message": "Hello, how can I help you today?"})
