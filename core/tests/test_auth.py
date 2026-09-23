from django.contrib.auth.models import User
from rest_framework.test import APITestCase
from rest_framework import status


class AuthTests(APITestCase):
    def test_register_creates_user(self):
        response = self.client.post('/api/auth/register/', {
            'username': 'testuser', 'password': 'strongpass123'
        })
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(User.objects.filter(username='testuser').exists())

    def test_register_rejects_short_password(self):
        response = self.client.post('/api/auth/register/', {
            'username': 'testuser2', 'password': 'short'
        })
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_register_rejects_duplicate_username(self):
        User.objects.create_user(username='dupe', password='strongpass123')
        response = self.client.post('/api/auth/register/', {
            'username': 'dupe', 'password': 'anotherpass123'
        })
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_login_returns_tokens(self):
        User.objects.create_user(username='loginuser', password='strongpass123')
        response = self.client.post('/api/auth/login/', {
            'username': 'loginuser', 'password': 'strongpass123'
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)

    def test_protected_endpoint_rejects_no_token(self):
        response = self.client.get('/api/job-descriptions/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)