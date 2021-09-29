from django.conf.urls import url
from django.urls import path

from . import views
 
urlpatterns = [
    url(r'^$', views.index),
    url('home', views.index),
    url('aicoach', views.aicoach),
    url('search', views.search),
    url(r'^ySearch$', views.searchYear),
    url('correlation', views.searchCorrelation),
    url('cSearch', views.compareSearch)
]