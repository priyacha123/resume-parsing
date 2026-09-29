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

On Render, set `ALLOWED_HOSTS` to the backend hostname, for example:

```env
ALLOWED_HOSTS=resume-parsing-np5i.onrender.com
```

Render also provides `RENDER_EXTERNAL_HOSTNAME`; the settings automatically include that hostname when it is available.

`UPSTASH_REDIS_URL` is required only when running Celery. The main `/api/match/` flow is synchronous.

## Email configuration

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

Use either TLS on port `587` or SSL on port `465`; do not enable both. For Gmail, use an app password rather than your normal account password. The SMTP provider can be Gmail, Outlook, Brevo, Mailgun, or another non-AWS provider.

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

## Gemini integration

The backend uses the current `google-genai` SDK:

```python
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="...",
)
```

The deprecated `google-generativeai` package is not used.

## Docker

```powershell
Copy-Item .env.example .env
docker compose up --build -d
docker compose exec web python manage.py migrate
```

The default stack runs Django and Nginx. It does not start local Redis.

To explicitly run the optional Celery worker:

```powershell
docker compose --profile worker up --build -d
```

The worker uses Upstash through `UPSTASH_REDIS_URL` and is capped at three concurrent processes.

## Render deployment

Create a Render Web Service using this directory as the root:

- Runtime: Docker
- Release command: `python manage.py migrate`
- Health check path: `/api/health/`

Set these environment variables:

```text
DJANGO_SECRET_KEY
DATABASE_URL
ALLOWED_HOSTS
CORS_ALLOWED_ORIGINS
CSRF_TRUSTED_ORIGINS
SECURE_SSL_REDIRECT=True
GEMINI_API_KEY
HF_API_TOKEN
WEBHOOK_SECRET
UPSTASH_REDIS_URL
```

Use deployed origins, not paths, for CORS and CSRF values:

```text
ALLOWED_HOSTS=resume-parsing-np5i.onrender.com
CORS_ALLOWED_ORIGINS=https://your-frontend.onrender.com
CSRF_TRUSTED_ORIGINS=https://your-frontend.onrender.com
```

Render free plans do not provide Background Workers. The current matching endpoint does not require one.

## Validation

```powershell
.\venv\Scripts\python.exe manage.py check
.\venv\Scripts\python.exe manage.py test
```

Uploaded files are stored on the server's local `MEDIA_ROOT` only; no S3 or AWS storage backend is configured. Render disk storage is ephemeral unless you attach a paid persistent disk, so uploaded files can disappear after redeploys or restarts. Never commit `.env`, uploaded resumes, or API keys. Rotate any credentials that have been exposed.
