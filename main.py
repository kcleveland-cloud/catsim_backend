import os
from datetime import datetime
from functools import lru_cache

import requests
import stripe
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# =========================
# Environment variables
# =========================

DATABASE_URL = os.getenv("DATABASE_URL")

STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")  # your API identifier in Auth0
AUTH0_ISSUER = f"https://{AUTH0_DOMAIN}/" if AUTH0_DOMAIN else None

# URL of your Streamlit FRONTEND (Dev for now)
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://cubesimprov2-noruuoxdtsrjzdskhuobbr.streamlit.app",
)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

if not STRIPE_API_KEY:
    raise RuntimeError("STRIPE_API_KEY is not set")

stripe.api_key = STRIPE_API_KEY

# Stripe price IDs – you set these in Render env
PRICE_MAP = {
    "standard_monthly": os.getenv("STRIPE_PRICE_STANDARD_MONTHLY"),
    "standard_yearly": os.getenv("STRIPE_PRICE_STANDARD_YEARLY"),
    "pro_monthly": os.getenv("STRIPE_PRICE_PRO_MONTHLY"),
    "pro_yearly": os.getenv("STRIPE_PRICE_PRO_YEARLY"),
    "academic_yearly": os.getenv("STRIPE_PRICE_ACAD_YEARLY"),
    "dept_yearly": os.getenv("STRIPE_PRICE_DEPT_YEARLY"),
}

# =========================
# Database setup
# =========================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    auth0_sub = Column(String, unique=True, index=True, nullable=False)
    email = Column(String)
    stripe_customer_id = Column(String)
    stripe_subscription_id = Column(String)
    plan = Column(String)     # Stripe price ID (e.g. price_xxx) or 'standard'/'pro'
    status = Column(String)   # 'trialing', 'active', 'canceled', etc.
    trial_end = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# =========================
# FastAPI app & CORS
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock to Streamlit URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================
# Auth0 helpers (used only by /me/subscription; optional)
# =========================

@lru_cache()
def get_jwks():
    if not AUTH0_DOMAIN:
        raise RuntimeError("AUTH0_DOMAIN is not set")
    jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    resp = requests.get(jwks_url)
    resp.raise_for_status()
    return resp.json()


def verify_token(token: str):
    if not AUTH0_AUDIENCE or not AUTH0_ISSUER:
        raise RuntimeError("AUTH0_AUDIENCE or AUTH0_ISSUER not configured")

    jwks = get_jwks()
    unverified_header = jwt.get_unverified_header(token)

    rsa_key = {}
    for key in jwks["keys"]:
        if key["kid"] == unverified_header["kid"]:
            rsa_key = {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"],
            }
            break

    if not rsa_key:
        raise HTTPException(status_code=401, detail="Invalid token header")

    try:
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=AUTH0_AUDIENCE,
            issuer=AUTH0_ISSUER,
        )
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user_payload(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = parts[1]
    return verify_token(token)

# =========================
# Health check
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# SYNC USER from Streamlit
# Streamlit POSTs here with auth0_sub + email
# =========================

@app.post("/sync-user")
async def sync_user(payload: dict, db: Session = Depends(get_db)):
    auth0_sub = payload.get("auth0_sub")
    email = payload.get("email")
    stripe_customer_id = payload.get("stripe_customer_id")

    if not auth0_sub:
        raise HTTPException(status_code=400, detail="auth0_sub is required")

    user = db.query(User).filter(User.auth0_sub == auth0_sub).first()
    if not user:
        user = User(auth0_sub=auth0_sub, email=email)
        db.add(user)
    else:
        if email:
            user.email = email

    if stripe_customer_id:
        user.stripe_customer_id = stripe_customer_id

    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)

    return {
        "auth0_sub": user.auth0_sub,
        "email": user.email,
        "stripe_customer_id": user.stripe_customer_id,
        "plan": user.plan,
        "status": user.status,
    }

# =========================
# SUBSCRIPTION STATE for Streamlit
# GET /subscription-state?auth0_sub=...
# =========================

@app.get("/subscription-state")
def subscription_state(auth0_sub: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.auth0_sub == auth0_sub).first()
    if not user:
        return {
            "plan": None,
            "status": None,
            "price_id": None,
            "current_period_end": None,
            "customer_id": None,
        }

    return {
        "plan": user.plan,
        "status": user.status,
        "price_id": user.plan,
        "current_period_end": user.trial_end.isoformat() if user.trial_end else None,
        "customer_id": user.stripe_customer_id,
    }

# =========================
# CREATE CHECKOUT SESSION
# POST /create-checkout-session
# body: {"plan_key": "pro_monthly", "email": "x@y.com"}
# =========================

@app.post("/create-checkout-session")
async def create_checkout_session(payload: dict, db: Session = Depends(get_db)):
    plan_key = payload.get("plan_key")
    email = payload.get("email")

    if plan_key not in PRICE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown plan_key '{plan_key}'",
        )

    price_id = PRICE_MAP[plan_key]
    if not price_id:
        raise HTTPException(
            status_code=400,
            detail=f"Stripe price not configured for plan_key '{plan_key}'",
        )

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            customer_email=email,
            success_url=f"{FRONTEND_URL}?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}?checkout=cancel",
        )
    except Exception as e:
        # Bubble up as clean 500 with message
        raise HTTPException(status_code=500, detail=f"Stripe error: {e}")

    return {"url": session.url}

# =========================
# CREATE BILLING PORTAL SESSION
# POST /create-portal-session
# body: {"customer_id": "cus_xxx"}
# =========================

@app.post("/create-portal-session")
async def create_portal_session(payload: dict):
    customer_id = payload.get("customer_id")
    if not customer_id:
        raise HTTPException(status_code=400, detail="customer_id is required")

    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=FRONTEND_URL,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe error: {e}")

    return {"url": portal_session.url}

# =========================
# OPTIONAL: /me/subscription (Auth0-protected)
# =========================

@app.get("/me/subscription")
def get_my_subscription(
    request: Request,
    db: Session = Depends(get_db),
):
    payload = get_current_user_payload(request)
    auth0_sub = payload["sub"]
    email = payload.get("email")

    user = db.query(User).filter(User.auth0_sub == auth0_sub).first()
    if not user:
        user = User(auth0_sub=auth0_sub, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)

    return {
        "auth0_sub": user.auth0_sub,
        "email": user.email,
        "plan": user.plan,
        "status": user.status,
        "trial_end": user.trial_end.isoformat() if user.trial_end else None,
    }

# =========================
# STRIPE WEBHOOK – updates DB
# =========================

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Stripe webhook secret not configured",
        )

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid webhook: {e}")

    event_type = event["type"]
    data = event["data"]["object"]

    def upsert_user_from_stripe(customer_id, subscription_id=None):
        stripe_customer = stripe.Customer.retrieve(customer_id)
        email = stripe_customer.get("email")

        user = None
        if email:
            user = db.query(User).filter(User.email == email).first()

        if not user:
            user = User(
                auth0_sub=email or f"stripe:{customer_id}",
                email=email,
                stripe_customer_id=customer_id,
            )
            db.add(user)
        else:
            user.stripe_customer_id = customer_id

        if subscription_id:
            user.stripe_subscription_id = subscription_id

        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)
        return user

    if event_type in [
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    ]:
        sub = data
        customer_id = sub["customer"]
        subscription_id = sub["id"]
        status = sub["status"]
        plan_id = sub["items"]["data"][0]["plan"]["id"]
        trial_end = (
            datetime.fromtimestamp(sub["trial_end"]) if sub.get("trial_end") else None
        )

        user = upsert_user_from_stripe(customer_id, subscription_id)
        user.status = status
        user.plan = plan_id
        user.trial_end = trial_end
        user.updated_at = datetime.utcnow()
        db.commit()

    return {"received": True}

@app.get("/debug/prices")
def debug_prices():
    return PRICE_MAP
