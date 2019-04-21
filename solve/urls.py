from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns

from solve.views import index, solve

urlpatterns = [
	path('', index, name="solve_index"),
	path('solve', solve, name="solve_solve"),
]


urlpatterns = format_suffix_patterns(urlpatterns)
