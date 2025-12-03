import os
from datetime import datetime
from functools import lru_cache

import requests
import stripe
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# =========================
# Environment variables
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")

STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_AUDIENCE = os.getenv("https://catsim-backend-api")  # your API identifier in Auth0
AUTH0_ISSUER = f"https://{AUTH0_DOMAIN}/" if AUTH0_DOMAIN else None

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

if not STRIPE_API_KEY:
    raise RuntimeError("STRIPE_API_KEY is not set")

stripe.api_key = STRIPE_API_KEY

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
    plan = Column(String)     # store Stripe plan/price ID or 'standard'/'pro'
    status = Column(String)   # 'trialing', 'active', 'canceled', etc.
    trial_end = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


# This will auto-create the table in your Render Postgres DB
Base.metadata.create_all(bind=engine)

# =========================
# FastAPI app
# =========================
app = FastAPI()

# For launch, keep CORS simple: allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down later to just Streamlit URL
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
# Auth0 helpers
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
# Simple health check
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# /me/subscription – for Streamlit
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
# Stripe webhook – updates DB
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
        # fetch customer info from Stripe to get email
        stripe_customer = stripe.Customer.retrieve(customer_id)
        email = stripe_customer.get("email")

        user = None
        if email:
            user = db.query(User).filter(User.email == email).first()

        if not user:
            # No existing user; create one keyed on email for now.
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

    # you can add more event handlers later if needed

    return {"received": True}
