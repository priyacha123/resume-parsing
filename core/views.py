from django.shortcuts import render, get_object_or_404
from django.contrib.auth.models import User
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import generics, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from .webhook_security import verify_signature
from .models import Resume, JobDescription, MatchResult
from .serializers import ResumeUploadSerializer, JobDescriptionSerializer, MatchResultSerializer
from .parsing import extract_resume_text
from .matching import compute_tfidf_score
from .embeddings import compute_embedding, cosine_similarity_score
from .tasks import compute_match_task, parse_resume_task
from celery.result import AsyncResult
from .tailoring import generate_tailoring_suggestions

# Create your views here.
# class ResumeUploadView(APIView):
#     permission_classes = [permissions.IsAuthenticated]  

#     def post(self, request):
#         serializer = ResumeUploadSerializer(data=request.data)
#         serializer.is_valid(raise_exception=True)

#         file_obj = request.FILES['file']
#         # Sanitize filename before storing — no path traversal via ../../
#         safe_filename = file_obj.name.replace('/', '_').replace('\\', '_')

#         # try:
#         #     # Extract text from the uploaded file
#         #     raw_text = extract_resume_text(resume.file, safe_filename)
#         #     resume.raw_text = raw_text
#         #     resume.save()

#         # except Exception as e:
#         #     return Response(
#         #         {"error": f"Upload succeded but parsing failed: {str(e)}"},
#         #         status = 207  # multi-status: file saved, parsing had an issue
#         #     )

#         # Trigger the asynchronous parsing task
#         resume = serializer.save(
#             user=request.user if request.user.is_authenticated else None, original_filename=safe_filename
#         )


#         # Parsing runs async — moved this into a Celery task
#         parse_resume_task.delay(resume.id)

#         # Return the serialized resume data
#         return Response(ResumeUploadSerializer(resume).data, status = 202) # 202 = accepted, processing


class ResumeUploadView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = ResumeUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        file_obj = request.FILES['file']
        # Sanitize filename before storing — no path traversal via ../../
        safe_filename = file_obj.name.replace('/', '_').replace('\\', '_')

        resume = serializer.save(
            user=request.user if request.user.is_authenticated else None,
            original_filename=safe_filename
        )

        # Synchronous parsing — Celery/Redis available locally (see docker-compose.yml),
        # but this deploy runs inline due to Render free-tier memory constraints
        try:
            raw_text = extract_resume_text(resume.file, safe_filename)
            resume.raw_text = raw_text
            resume.save()
        except Exception as e:
            return Response(
                {"error": f"Upload succeeded but parsing failed: {str(e)}"},
                status=207  # multi-status: file saved, parsing had an issue
            )

        return Response(ResumeUploadSerializer(resume).data, status=201)


class JobDescriptionCreateView(generics.ListCreateAPIView):
    serializer_class = JobDescriptionSerializer
    permission_classes = [permissions.IsAuthenticated]  # Adjust permissions as needed

    def get_queryset(self):
        return JobDescription.objects.filter(user=self.request.user) if self.request.user.is_authenticated else JobDescription.objects.none()

    def perform_create(self, serializer):
        from django.contrib.auth.models import User
        user = self.request.user
        serializer.save(user=user)

class MatchCreateView(APIView):
    permission_classes = [permissions.IsAuthenticated] 

    # def post(self, request):
    #     resume_id = request.data.get('resume_id')
    #     jd_id = request.data.get('job_description_id')
    #     method = request.data.get('method', 'embedding')  # Default to embedding if not specified

    #     # resume = get_object_or_404(Resume, pk=resume_id)
    #     # jd = get_object_or_404(JobDescription, pk=jd_id)

    #     # if method == 'tfidf':
    #     #     # Compute the TF-IDF similarity score
    #     #     score = compute_tfidf_score(resume.raw_text, jd.raw_text)
    #     # else:
    #     #     # Compute + cache embeddings so repeat matches don't recompute
    #     #     if not resume.embedding:
    #     #         resume.embedding = compute_embedding(resume.raw_text)
    #     #         resume.save()
    #     #     if not jd.embedding:
    #     #         jd.embedding = compute_embedding(jd.raw_text)
    #     #         jd.save()
    #     #     # Compute the cosine similarity score using embeddings
    #     #     score = cosine_similarity_score(resume.embedding, jd.embedding)

    #     # Create a MatchResult instance
    #     # match = MatchResult.objects.create(
    #     #     resume=resume,
    #     #     job_description=jd,
    #     #     score=score,
    #     #     method=method
    #     # )


    #     # Ownership check — not just existence check. This is the IDOR fix.
    #     resume = get_object_or_404(Resume, pk=resume_id, user=request.user)
    #     jd = get_object_or_404(JobDescription, pk=jd_id, user=request.user)

    #     task = compute_match_task.delay(resume.id, jd.id, method)

    #     return Response({
    #         "task_id": task.id,
    #         "status": "processing",
    #     }, status=202)  # 202 = accepted, processing

    def post(self, request):
        resume_id = request.data.get('resume_id')
        jd_id = request.data.get('job_description_id')
        method = request.data.get('method', 'embedding')

        resume = get_object_or_404(Resume, pk=resume_id, user=request.user)
        jd = get_object_or_404(JobDescription, pk=jd_id, user=request.user)

        if not resume.embedding:
            resume.embedding = compute_embedding(resume.raw_text)
            resume.save()
        if not jd.embedding:
            jd.embedding = compute_embedding(jd.raw_text)
            jd.save()

        score = cosine_similarity_score(resume.embedding, jd.embedding)
        match = MatchResult.objects.create(resume=resume, job_description=jd, score=score, method=method)

        suggestions = generate_tailoring_suggestions(resume.raw_text, jd.raw_text)
        match.suggestions = suggestions
        match.save()

        return Response(MatchResultSerializer(match).data, status=201)


class TaskStatusView(APIView):
    permission_classes = [permissions.IsAuthenticated] 

    def get(self, request, task_id):
        result = AsyncResult(task_id)
        return Response({
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None
        })


class RegisterView(APIView):
    permission_classes = [permissions.AllowAny] # must be open — this IS the signup endpoint
    throttle_classes = [AnonRateThrottle]  # Optional: throttle registration attempts

    def post(self, request):
        username = request.data.get('username', '').strip()
        password = request.data.get('password', '')

        if not username or len(password) < 8:
            return Response({"error": "Username and password are required. Password must be at least 8 characters."}, status=400)
        if User.objects.filter(username=username).exists():
            return Response({"error": "Username already exists."}, status=400)

        # Django's create_user hashes the password (PBKDF2 by default) — never store plaintext
        user = User.objects.create_user(username=username, password=password)
        return Response({"message": "User registered successfully."}, status=201)

@method_decorator(csrf_exempt, name='dispatch')
class WebhookMatchResultView(APIView):
    permission_classes = [permissions.AllowAny]  # Webhook endpoint should be open, but secured via signature

    def get(self, request, match_id):
        signature = request.headers.get('X-Webhook-Signature', '')
        # Signature is computed over the match_id as the payload, since GET has no body
        payload = str(match_id).encode()

        if not verify_signature(payload, signature):
            return Response({"error": "Invalid signature"}, status=401)

        match = get_object_or_404(MatchResult, pk=match_id)
        return Response(MatchResultSerializer(match).data)