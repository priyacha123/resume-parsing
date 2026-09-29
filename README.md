# ResumeMatch Backend

Django REST API for parsing resumes, matching them with job descriptions, and generating model-specific optimization recommendations.

## Features

- PDF, DOCX, and TXT resume parsing.
- JWT authentication with refresh-token rotation and logout blacklisting.
- TF-IDF keyword matching.
- Hugging Face MiniLM embeddings.
- Hybrid matching with 65% semantic and 35% keyword weighting.
- Gemini-powered tailoring recommendations.
- Missing keywords, matched proficiencies, and exactly two project ideas per match.
- PostgreSQL with `pgvector`.
- Optional Celery tasks using Upstash Redis.

## Requirements

- Python 3.12+
- PostgreSQL with the `vector` extension
- Optional Upstash Redis for Celery
- Optional Gemini API key
- Optional Hugging Face API token

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python manage.py migrate
python manage.py runserver
```

Configure `.env` before starting the server:

```env
DJANGO_SECRET_KEY=replace-with-a-long-random-secret
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
DATABASE_URL=postgresql://username:password@localhost:5432/resume_automation
GEMINI_API_KEY=
HF_API_TOKEN=
WEBHOOK_SECRET=replace-with-a-webhook-secret
```

<!-- ## Email configuration

The project currently does not send registration, password-reset, or notification emails. Django is configured for a generic SMTP provider so email can be enabled later without adding a cloud-specific service.

If you need SMTP email, add these values to `.env`:

```env
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.example.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@example.com
EMAIL_HOST_PASSWORD=your-email-password-or-app-password
EMAIL_USE_TLS=True
EMAIL_USE_SSL=False
DEFAULT_FROM_EMAIL=your-email@example.com
```

Use either TLS on port `587` or SSL on port `465`; do not enable both. For Gmail, use an app password rather than your normal account password. The SMTP provider can be Gmail, Outlook, Brevo, Mailgun, or another non-AWS provider. -->

## API endpoints

All endpoints are prefixed with `/api/`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/auth/register/` | Register a user |
| POST | `/auth/login/` | Obtain access and refresh tokens |
| POST | `/auth/refresh/` | Refresh an access token |
| POST | `/auth/logout/` | Blacklist a refresh token |
| GET | `/health/` | Health check |
| POST | `/resumes/upload/` | Upload and parse a resume |
| GET | `/resumes/` | List the current user's resumes |
| POST | `/job-descriptions/` | Create a job description |
| GET | `/job-descriptions/` | List the current user's job descriptions |
| POST | `/match/` | Run synchronous matching and tailoring |
| GET | `/matches/` | List the current user's match results |

Example match request:

```json
{
  "resume_id": 1,
  "job_description_id": 1,
  "method": "hybrid"
}
```

Supported methods are `tfidf`, `embedding`, and `hybrid`.

