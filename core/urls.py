from django.urls import path
from .views import (
    ResumeUploadView,
    ResumeListView,
    JobDescriptionCreateView,
    MatchCreateView,
    MatchListView,
    TaskStatusView,
    RegisterView,
    WebhookMatchResultView
)
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('resumes/upload/', ResumeUploadView.as_view(), name='resume-upload'),
    path('resumes/', ResumeListView.as_view(), name='resume-list'),
    path('job-descriptions/', JobDescriptionCreateView.as_view(), name='jd-list-create'),
    path('match/', MatchCreateView.as_view(), name='match-create'),
    path('matches/', MatchListView.as_view(), name='match-list'),
    path('tasks/<str:task_id>/', TaskStatusView.as_view(), name='task-status'),
    path('auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/register/', RegisterView.as_view(), name='register'),
    path('webhook/match/<int:match_id>/', WebhookMatchResultView.as_view(), name='webhook-match-result'),
]