from django.db import models
from django.contrib.auth.models import User
from pgvector.django import VectorField

# Create your models here.
class Resume(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='resumes', null=True, blank=True)
    file = models.FileField(upload_to='resumes/')
    original_filename = models.CharField(max_length=255)
    raw_text = models.TextField(blank=True)
    parsed_data = models.JSONField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    embedding = VectorField(dimensions=384, null=True, blank=True)  # 384 = MiniLM output size

    def __str__(self):
        username = self.user.username if self.user else "Anonymous"
        return f"{self.original_filename} ({username})"
        # return f"{self.original_filename} ({self.user.username})"

class JobDescription(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='job_descriptions')
    title = models.CharField(max_length=255, blank=True)
    company = models.CharField(max_length=255, blank=True)
    raw_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    embedding = VectorField(dimensions=384, null=True, blank=True)

    def __str__(self):
        return self.title or f"JD #{self.pk}"

class MatchResult(models.Model):
    resume = models.ForeignKey(Resume, on_delete=models.CASCADE, related_name='matches')
    job_description = models.ForeignKey(JobDescription, on_delete=models.CASCADE, related_name='matches')
    score = models.FloatField(null=True, blank=True)
    method = models.CharField(max_length=50, blank=True)
    suggestions = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Match: {self.resume} <-> {self.job_description} {self.score}"

class Application(models.Model):
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('applied', 'Applied'),
        ('rejected', 'Rejected'),   
        ('interview', 'Interview'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='applications')
    match_result = models.ForeignKey(MatchResult, on_delete=models.SET_NULL, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    applied_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.status}"


