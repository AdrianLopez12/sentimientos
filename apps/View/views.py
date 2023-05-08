from django.shortcuts import render
from apps.Logica import modeloSNN #para utilizar el método inteligente
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
import json
from django.http import JsonResponse

class Clasificacion():
    def determinarSentimiento(request):
        return render(request, 'sentimiento.html')
    @api_view(["GET","POST"])
    def predecir(request):
        try:
            texto=request.POST.get("texto")
            resul=modeloSNN.predecir2(modeloSNN,texto)
        except ValueError as e:
            resul = "Error"
        return render(request, "informe.html", {"e":resul})
    
    @csrf_exempt
    @api_view(["GET","POST"])
    def predecirIOJson(request):
        print(request)
        print("****")
        print(request.body)
        print("****")
        body.json.loads(request.body.decode('utf-8'))
        texto=str(body["texto"])
        print(texto)
        resul=modeloSNN.predecir2(texto)
        data = {"resultado": resul}
        resp = JsonResponse(data)
        resp["Access-Control-Allow-Origin"] = "*"
        return resp
