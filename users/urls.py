from django.urls import path
from . import views


urlpatterns = [
    path("",views.home),
    path("signin",views.signin),
    path("logout", views.logout_view, name='logout_view_o')
]