
from django.contrib import admin
from django.urls import path
from django.urls import re_path
from apps.View import views

from rest_framework import permissions




urlpatterns = [
    #re_path(r'^swagger(?P<format>.json|.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    #re_path(r'^swagger/$', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    #re_path(r'^redoc/$', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    path("admin/", admin.site.urls),
    re_path(r'^nuevasolicitud/$',views.Clasificacion.determinarSentimiento),
    re_path(r'^predecir/',views.Clasificacion.predecir),
    re_path(r'^predecirIOJson/',views.Clasificacion.predecirIOJson),
]
