from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
import numpy as np
from tensorflow.keras.models import load_model

@csrf_exempt 
def home(request):
    context = {'result': 'none'}
    if request.method == 'POST':
        model = load_model('./recommenderApp/MPox_Recommender.keras')
        responses = request.POST.dict()

        answers = [
            1 if 'Fever' in responses.keys() else 0,
            1 if 'Muscle Aches and Pain' in responses.keys() else 0,
            1 if 'Swollen Lymph Nodes' in responses.keys() else 0,
            1 if responses['Rectal Pain Answer'] == 'Yes' else 0,
            1 if responses['Sore Throat Answer'] == 'Yes' else 0,
            1 if responses['Oedema below the Waist Answer'] == 'Yes' else 0,
            1 if responses['Oral Lesions Answer'] == 'Yes' else 0,
            1 if responses['Solitary Oral Lesions Answer'] == 'Yes' else 0,
            1 if responses['Swollen Tonsils Answer'] == 'Yes' else 0,
            1 if responses['HIV Infection Answer'] == 'Yes' else 0,
            1 if responses['Sexually Transmitted Infections Answer'] == 'Yes' else 0,
        ]

        prediction = model.predict(np.array(answers).reshape(1, -1))
        if prediction > 0.35:
            context['result'] = 'positive'
        else:
            context['result'] = 'negative'

    return render(request, 'base.html', context)
