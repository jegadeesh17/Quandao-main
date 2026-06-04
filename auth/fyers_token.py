"""
backend/auth/fyers_token.py
Fyers API token management.
Automatically resolves the project root to find .env (no hardcoded paths).
"""

import os
import json
import webbrowser
import requests
from hashlib import sha256
from pathlib import Path
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

# Auto-detect project root (two levels up from this file: auth/ -> backend/ -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_PATH)


def load_env_vars() -> dict:
    """Load required Fyers credentials from .env."""
    return {
        "client_id":    os.getenv("APP_ID"),
        "secret_key":   os.getenv("APP_SECRET_ID"),
        "redirect_uri": os.getenv("REDIRECT_URI", "https://sridamul.in/"),
        "pin":          os.getenv("PIN"),
        "refresh_token":os.getenv("REFRESH_TOKEN"),
        "app_id_hash":  os.getenv("APP_ID_HASH"),
    }


def save_env_var(key: str, value: str) -> None:
    """Update or add a key=value pair in the .env file."""
    lines = _ENV_PATH.read_text().splitlines() if _ENV_PATH.exists() else []
    new_lines, updated = [], False

    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{key}={value}")

    _ENV_PATH.write_text("\n".join(new_lines))


def generate_app_id_hash(client_id: str, secret_key: str) -> str:
    """Generate SHA-256 hash of client_id:secret_key and save to .env."""
    raw = f"{client_id}:{secret_key}"
    app_id_hash = sha256(raw.encode()).hexdigest()
    save_env_var("APP_ID_HASH", app_id_hash)
    return app_id_hash


def get_access_token_from_refresh(app_id_hash: str, refresh_token: str, pin: str):
    """Try to refresh the access token using an existing refresh token."""
    url = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
    payload = {
        "grant_type":   "refresh_token",
        "appIdHash":    app_id_hash,
        "refresh_token": refresh_token,
        "pin":          pin,
    }
    resp = requests.post(url, headers={"Content-Type": "application/json"},
                         data=json.dumps(payload))
    if resp.status_code == 200:
        data = resp.json()
        if data.get("s") == "ok" and data.get("access_token"):
            token = data["access_token"]
            save_env_var("ACCESS_TOKEN", token)
            print("Access token refreshed successfully.")
            return token

    print("Refresh token invalid or expired.")
    return None


def generate_new_tokens(client_id: str, secret_key: str, redirect_uri: str):
    """Open browser to authorize, then exchange auth code for tokens."""
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )
    auth_url = session.generate_authcode()
    print(f"\nOpen this URL in your browser to authorize:\n{auth_url}")
    webbrowser.open(auth_url)

    auth_code = input("\nPaste the 'auth_code' from the redirected URL: ").strip()
    session.set_token(auth_code)
    token_response = session.generate_token()

    if token_response.get("access_token"):
        save_env_var("ACCESS_TOKEN", token_response["access_token"])
        print("ACCESS_TOKEN updated.")
    if token_response.get("refresh_token"):
        save_env_var("REFRESH_TOKEN", token_response["refresh_token"])
        print("REFRESH_TOKEN updated.")

    return token_response.get("access_token")


def main():
    env = load_env_vars()
    client_id, secret_key = env["client_id"], env["secret_key"]

    if not env["app_id_hash"]:
        env["app_id_hash"] = generate_app_id_hash(client_id, secret_key)

    access_token = None
    if env["refresh_token"]:
        access_token = get_access_token_from_refresh(
            env["app_id_hash"], env["refresh_token"], env["pin"]
        )

    if not access_token:
        access_token = generate_new_tokens(client_id, secret_key, env["redirect_uri"])

    print("\nFinal Access Token:", access_token)


if __name__ == "__main__":
    main()
