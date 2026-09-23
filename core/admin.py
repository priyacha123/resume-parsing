from django.contrib import admin
from  .models import Resume, JobDescription, MatchResult, Application

# Register your models here.
admin.site.register(Resume)
admin.site.register(JobDescription)
admin.site.register(MatchResult)
admin.site.register(Application)
