from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APITestCase
from rest_framework import status


class ResumeUploadTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='uploaduser', password='strongpass123')
        self.client.force_authenticate(user=self.user)  # bypasses real JWT for test speed

    def test_rejects_invalid_extension(self):
        fake_exe = SimpleUploadedFile("malware.exe", b"fake content", content_type="application/octet-stream")
        response = self.client.post('/api/resumes/upload/', {'file': fake_exe}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_rejects_oversized_file(self):
        big_content = b"x" * (6 * 1024 * 1024)  # 6MB, over the 5MB limit
        big_file = SimpleUploadedFile("resume.pdf", big_content, content_type="application/pdf")
        response = self.client.post('/api/resumes/upload/', {'file': big_file}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_accepts_valid_pdf(self):
        # Minimal valid-looking PDF bytes — good enough to pass extension/size checks
        pdf_content = b"%PDF-1.4 fake but valid-looking content"
        valid_file = SimpleUploadedFile("resume.pdf", pdf_content, content_type="application/pdf")
        response = self.client.post('/api/resumes/upload/', {'file': valid_file}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)