from django.urls import path
from .views.selectionsview import SelectionsView
from .views.inputpreferencesview import InputPreferencesView
from .views.showallocationview import ShowAllocationView
from .views.algorithmview import AlgorithmDetailView
from .views.homeview import HomeView  

urlpatterns = [
    path('', HomeView.as_view(), name='home'),  
    path('selections/', SelectionsView.as_view(), name='selections'),
    path('input_preferences/', InputPreferencesView.as_view(), name='input_preferences'),
    path('show_allocation/', ShowAllocationView.as_view(), name='show_allocation'),
    path('algorithm/<str:algorithm_name>/', AlgorithmDetailView.as_view(), name='algorithm_detail'),
]
