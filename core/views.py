import logging
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.models import User
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.throttling import AnonRateThrottle
from .webhook_security import verify_signature
from .models import Resume, JobDescription, MatchResult
from .serializers import ResumeUploadSerializer, JobDescriptionSerializer, MatchResultSerializer
from .parsing import extract_resume_text
from .matching import compute_tfidf_score
from .embeddings import compute_embedding, cosine_similarity_score
from .tailoring import generate_tailoring_suggestions

logger = logging.getLogger(__name__)


class ResumeUploadView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = ResumeUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({"error": "No file was attached to the request."}, status=status.HTTP_400_BAD_REQUEST)

        safe_filename = file_obj.name.replace('/', '_').replace('\\', '_')

        # Save record first to preserve file in storage
        resume = serializer.save(
            user=request.user if request.user.is_authenticated else None,
            original_filename=safe_filename
        )

        try:
            raw_text = extract_resume_text(resume.file, safe_filename)
            if not raw_text or len(raw_text.strip()) < 20:
                resume.delete()
                return Response(
                    {"error": "Could not extract sufficient text from this file. Ensure it is not an image-only scan or empty document."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            resume.raw_text = raw_text.strip()
            resume.save(update_fields=['raw_text'])
        except Exception as e:
            logger.error(f"Resume text extraction error for {safe_filename}: {e}")
            resume.delete()
            return Response(
                {"error": f"Failed to parse resume: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        return Response(ResumeUploadSerializer(resume).data, status=status.HTTP_201_CREATED)


class ResumeListView(generics.ListAPIView):
    serializer_class = ResumeUploadSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Resume.objects.filter(user=self.request.user).order_by('-uploaded_at')[:10]


class JobDescriptionCreateView(generics.ListCreateAPIView):
    serializer_class = JobDescriptionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return JobDescription.objects.filter(user=self.request.user).order_by('-created_at')[:10]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class MatchCreateView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        resume_id = request.data.get('resume_id')
        jd_id = request.data.get('job_description_id')
        method = request.data.get('method', 'hybrid')  # hybrid, embedding, or tfidf

        if not resume_id or not jd_id:
            return Response(
                {"error": "Both resume_id and job_description_id are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        resume = get_object_or_404(Resume, pk=resume_id, user=request.user)
        jd = get_object_or_404(JobDescription, pk=jd_id, user=request.user)

        # Self-heal missing resume raw_text if file is present
        if not resume.raw_text and resume.file:
            try:
                resume.raw_text = extract_resume_text(resume.file, resume.original_filename)
                resume.save(update_fields=['raw_text'])
            except Exception as e:
                return Response(
                    {"error": f"Resume text missing and re-extraction failed: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

        if not resume.raw_text or not resume.raw_text.strip():
            return Response(
                {"error": "The selected resume contains no extractable text. Please re-upload your resume."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not jd.raw_text or not jd.raw_text.strip():
            return Response(
                {"error": "The selected job description has no text content."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Compute matching score based on selected method
        score = 0.0
        used_method = method

        if method == 'tfidf':
            score = compute_tfidf_score(resume.raw_text, jd.raw_text)
        elif method == 'embedding':
            try:
                if not resume.embedding:
                    resume.embedding = compute_embedding(resume.raw_text)
                    resume.save(update_fields=['embedding'])
                if not jd.embedding:
                    jd.embedding = compute_embedding(jd.raw_text)
                    jd.save(update_fields=['embedding'])
                score = cosine_similarity_score(resume.embedding, jd.embedding)
            except Exception as e:
                logger.warning(f"Embedding failed, falling back to TF-IDF: {e}")
                score = compute_tfidf_score(resume.raw_text, jd.raw_text)
                used_method = 'tfidf_fallback'
        else:
            # Hybrid: Combines deep semantic similarity (70%) with exact keyword matching (30%)
            tfidf_score = compute_tfidf_score(resume.raw_text, jd.raw_text)
            embedding_score = None
            try:
                if not resume.embedding:
                    resume.embedding = compute_embedding(resume.raw_text)
                    resume.save(update_fields=['embedding'])
                if not jd.embedding:
                    jd.embedding = compute_embedding(jd.raw_text)
                    jd.save(update_fields=['embedding'])
                embedding_score = cosine_similarity_score(resume.embedding, jd.embedding)
            except Exception as e:
                logger.warning(f"Embedding failed during hybrid matching: {e}")

            if embedding_score is not None:
                # 65% semantic + 35% keyword overlap provides a very realistic ATS match
                score = round((embedding_score * 0.65) + (tfidf_score * 0.35), 1)
                used_method = 'hybrid'
            else:
                score = tfidf_score
                used_method = 'tfidf'

        # Ensure score is within valid range
        score = max(0.0, min(100.0, round(float(score), 1)))

        # Generate intelligent tailoring suggestions customized to the scoring method
        suggestions = generate_tailoring_suggestions(
            resume.raw_text,
            jd.raw_text,
            method=used_method,
            score=score
        )

        match = MatchResult.objects.create(
            resume=resume,
            job_description=jd,
            score=score,
            method=used_method,
            suggestions=suggestions
        )

        return Response(MatchResultSerializer(match).data, status=status.HTTP_201_CREATED)


class MatchListView(generics.ListAPIView):
    serializer_class = MatchResultSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return MatchResult.objects.filter(resume__user=self.request.user).order_by('-created_at')[:10]


class TaskStatusView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, task_id):
        from celery.result import AsyncResult
        result = AsyncResult(task_id)
        return Response({
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None
        })


class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]
    throttle_classes = [AnonRateThrottle]

    def post(self, request):
        username = request.data.get('username', '').strip()
        password = request.data.get('password', '')

        if not username or len(password) < 8:
            return Response(
                {"error": "Username and password are required. Password must be at least 8 characters."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if User.objects.filter(username=username).exists():
            return Response(
                {"error": "A user with that username already exists."},
                status=status.HTTP_400_BAD_REQUEST
            )

        User.objects.create_user(username=username, password=password)
        return Response({"message": "User registered successfully."}, status=status.HTTP_201_CREATED)


@method_decorator(csrf_exempt, name='dispatch')
class WebhookMatchResultView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, match_id):
        signature = request.headers.get('X-Webhook-Signature', '')
        payload = str(match_id).encode()

        if not verify_signature(payload, signature):
            return Response({"error": "Invalid signature"}, status=status.HTTP_401_UNAUTHORIZED)

        match = get_object_or_404(MatchResult, pk=match_id)
        return Response(MatchResultSerializer(match).data)