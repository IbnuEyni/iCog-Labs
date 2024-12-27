from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import TextSerializer
from .ml_model import predict_hate_speech

def home(request):
    return render(request, 'index.html')


class PredictView(APIView):
    def post(self, request):
        """
        Handle POST request for predicting hate speech in the provided text.
        """
        # Validate and serialize the input data
        serializer = TextSerializer(data=request.data)
        
        if serializer.is_valid():
            # Get the validated text input
            text = serializer.validated_data['text']
            
            # Get the prediction result from the model
            prediction = predict_hate_speech(text)
            
            # Return the result as a JSON response
            print(f"text: {text}, prediction: {prediction}")
            return Response({
                "text": text, 
                "prediction": prediction
            }, status=status.HTTP_200_OK)
        
        # If input data is invalid, return errors with a bad request status
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
