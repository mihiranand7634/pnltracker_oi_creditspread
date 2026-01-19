import os
import json
import webbrowser
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

API_KEY = os.getenv("KITE_API_KEY", "").strip()
API_SECRET = os.getenv("KITE_API_SECRET", "").strip()
TOKEN_PATH = os.getenv("KITE_TOKEN_PATH", "./kite_access_token.json").strip()

if not API_KEY or not API_SECRET:
    raise SystemExit("Missing KITE_API_KEY / KITE_API_SECRET in .env")

def extract_request_token(pasted: str) -> str:
    pasted = pasted.strip()
    if "request_token=" not in pasted and "http" not in pasted:
        if len(pasted) < 10:
            raise RuntimeError("Input doesn't look like a request_token or URL.")
        return pasted
    u = urlparse(pasted)
    q = parse_qs(u.query)
    rt = (q.get("request_token") or [""])[0].strip()
    status = (q.get("status") or [""])[0].strip()
    if status and status != "success":
        raise RuntimeError(f"Login status not success: status={status}")
    if not rt:
        raise RuntimeError("Could not find request_token in pasted URL.")
    return rt

def main():
    kite = KiteConnect(api_key=API_KEY)
    login_url = kite.login_url()
    print("\n[ACTION] Open this URL in your browser and login:\n", login_url)
    try:
        webbrowser.open(login_url, new=1, autoraise=True)
    except Exception:
        pass

    print("\nAfter login you will be redirected to your Redirect URL with ?request_token=....")
    pasted = input("\nPaste FULL redirected URL (or request_token): ").strip()
    request_token = extract_request_token(pasted)

    data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = data["access_token"]

    out = {"access_token": access_token}
    with open(TOKEN_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n[OK] Saved access_token to: {TOKEN_PATH}")
    print("[NOTE] If your token expires daily, rerun this script and restart the service.")

if __name__ == "__main__":
    main()
