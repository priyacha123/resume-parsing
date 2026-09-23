import os
import ngrok
from dotenv import load_dotenv

load_dotenv()

def connect_ngrok():
    forwarder = ngrok.forward(
        "localhost:8000",
        authtoken_from_env=True,
        domain="everybody-display-gratitude.ngrok-free.dev"
    )
    print(f"Available at: {forwarder.url()}")
    input("Press Enter to stop the tunnel...\n")  # keeps the script (and tunnel) alive

connect_ngrok()