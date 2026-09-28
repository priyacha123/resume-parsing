import hmac
import hashlib
import os

def verify_signature(payload_body: bytes, signature_header: str) -> bool:
    if not signature_header:
        return False

    secret_value = os.environ.get('WEBHOOK_SECRET')
    if not secret_value:
        return False
    secret = secret_value.encode()
    expected_signature = hmac.new(secret, payload_body, hashlib.sha256).hexdigest()

    # compare_digest prevents timing attacks — never use == for signature comparison
    return hmac.compare_digest(expected_signature, signature_header)
