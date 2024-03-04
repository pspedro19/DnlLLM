from django.shortcuts import render
from django.shortcuts import render
from django.http import JsonResponse

def index(request):
    return render(request, 'index.html')

def api_view(request, mensaje):
    return JsonResponse({'mensaje': f'Hola, soy el asistente virtual. Recibí tu mensaje: {mensaje}'})