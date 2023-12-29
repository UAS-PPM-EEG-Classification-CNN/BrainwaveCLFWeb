from django.contrib import admin
from django.urls import path
from BrainApp.views import IndexView, PredictView, ResultView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', PredictView.as_view()),
    path('result/', ResultView.as_view()),
    path('', IndexView.as_view()),
]
