from django.contrib.auth.models import User
from rest_framework.test import APITestCase
from rest_framework import status
from core.models import Resume, JobDescription


class IDORTests(APITestCase):
    def setUp(self):
        self.user_a = User.objects.create_user(username='usera', password='strongpass123')
        self.user_b = User.objects.create_user(username='userb', password='strongpass123')
        self.resume_a = Resume.objects.create(user=self.user_a, original_filename='a.pdf', raw_text='python django')
        self.jd_a = JobDescription.objects.create(user=self.user_a, raw_text='python django role')

    def test_user_cannot_match_another_users_resume(self):
        self.client.force_authenticate(user=self.user_b)
        response = self.client.post('/api/match/', {
            'resume_id': self.resume_a.id,
            'job_description_id': self.jd_a.id
        })
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)