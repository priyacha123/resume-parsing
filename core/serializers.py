from rest_framework import serializers
from .models import Resume, MatchResult, JobDescription
import bleach

ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MAX_FILE_SIZE_MB = 5
ALLOWED_TAGS = [] # strip ALL HTML — JD text should be plain text only

class ResumeUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Resume
        fields = ['id', 'file', 'original_filename', 'uploaded_at']
        read_only_fields = ['id', 'original_filename', 'uploaded_at']

    def validate_file(self, value):
        # Extension whitelist — never trust the extension alone, but it's the first gate
        ext = value.name.lower().rsplit('.', 1)[-1]
        if ext not in ALLOWED_EXTENSIONS:
            raise serializers.ValidationError("Only PDF and DOCX files are allowed.")

        # Size limit — prevents disk-fill / DoS via huge uploads
        if value.size > MAX_FILE_SIZE_MB * 1024 *1024:
            raise serializers.ValidationError(f"File must be under {MAX_FILE_SIZE_MB} MB.")

        return value

class JobDescriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobDescription
        fields = ['id', 'title', 'company', 'raw_text', 'created_at']
        read_only_fields = ['id', 'created_at']

    def validate_raw_text(self, value):
        if not value.strip():
            raise serializers.ValidationError("Job description text cannot be empty.")
        if len(value) > 30000:
            raise serializers.ValidationError("Job description text cannot exceed 30,000 characters.")
        # Strip any HTML/script content — defense against stored XSS
        cleaned = bleach.clean(value, tags=ALLOWED_TAGS, strip=True)
        return cleaned


    def validate_title(self, value):
        return bleach.clean(value, tags=[], strip=True) if value else value

    def validate_company(self, value):
        return bleach.clean(value, tags=[], strip=True) if value else value

class MatchResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatchResult
        fields = ['id', 'resume', 'job_description', 'score', 'method', 'suggestions', 'created_at']