from rest_framework import serializers
from .models import Resume, MatchResult, JobDescription
import bleach

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_FILE_SIZE_MB = 10
ALLOWED_TAGS = []  # strip ALL HTML — JD text should be plain text only

class ResumeUploadSerializer(serializers.ModelSerializer):
    word_count = serializers.SerializerMethodField()
    preview = serializers.SerializerMethodField()

    class Meta:
        model = Resume
        fields = ['id', 'file', 'original_filename', 'raw_text', 'word_count', 'preview', 'uploaded_at']
        read_only_fields = ['id', 'original_filename', 'raw_text', 'word_count', 'preview', 'uploaded_at']

    def validate_file(self, value):
        filename = getattr(value, 'name', '')
        if not filename or '.' not in filename:
            raise serializers.ValidationError("File must have a valid extension (.pdf, .docx, or .txt).")

        ext = filename.lower().rsplit('.', 1)[-1]
        if ext not in ALLOWED_EXTENSIONS:
            raise serializers.ValidationError("Only PDF, DOCX, and TXT files are supported.")

        if value.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise serializers.ValidationError(f"File must be under {MAX_FILE_SIZE_MB} MB.")

        return value

    def get_word_count(self, obj):
        return len(obj.raw_text.split()) if obj.raw_text else 0

    def get_preview(self, obj):
        if not obj.raw_text:
            return ""
        return obj.raw_text[:300] + ("..." if len(obj.raw_text) > 300 else "")


class JobDescriptionSerializer(serializers.ModelSerializer):
    word_count = serializers.SerializerMethodField()

    class Meta:
        model = JobDescription
        fields = ['id', 'title', 'company', 'raw_text', 'word_count', 'created_at']
        read_only_fields = ['id', 'word_count', 'created_at']

    def validate_raw_text(self, value):
        if not value or not value.strip():
            raise serializers.ValidationError("Job description text cannot be empty.")
        if len(value) > 30000:
            raise serializers.ValidationError("Job description text cannot exceed 30,000 characters.")
        cleaned = bleach.clean(value, tags=ALLOWED_TAGS, strip=True)
        return cleaned.strip()

    def validate_title(self, value):
        return bleach.clean(value, tags=[], strip=True).strip() if value else ""

    def validate_company(self, value):
        return bleach.clean(value, tags=[], strip=True).strip() if value else ""

    def get_word_count(self, obj):
        return len(obj.raw_text.split()) if obj.raw_text else 0


class MatchResultSerializer(serializers.ModelSerializer):
    job_title = serializers.CharField(source='job_description.title', read_only=True)
    resume_name = serializers.CharField(source='resume.original_filename', read_only=True)

    class Meta:
        model = MatchResult
        fields = [
            'id', 'resume', 'job_description', 'resume_name', 'job_title',
            'score', 'method', 'suggestions', 'created_at'
        ]