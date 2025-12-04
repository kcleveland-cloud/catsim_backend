import os
from datetime import datetime
from functools import lru_cache

import requests
import stripe
from fastapi import FastAPI, Request, HTTPException, Depends, Body
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

# ============================================================
# Environment variables
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")

STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")  # sk_test_xxx or sk_live_xxx
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# Frontend base URL for redirecting back from Stripe
FRONTEND_BASE_URL = os.getenv(
    "FRONTEND_BASE_URL",
    "https://cubesimprov2-noruuoxdtsrjzdskhuobbr.streamlit.app",
)

# Stripe price IDs (set all of these in Render)
PRICE_STANDARD_MONTHLY = os.getenv("STRIPE_PRICE_STANDARD_MONTHLY")
PRICE_STANDARD_YEARLY = os.getenv("STRIPE_PRICE_STANDARD_YEARLY")
PRICE_PRO_MONTHLY = os.getenv("STRIPE_PRICE_PRO_MONTHLY")
PRICE_PRO_YEARLY = os.getenv("STRIPE_PRICE_PRO_YEARLY")
PRICE_ACADEMIC_YEARLY = os.getenv("STRIPE_PRICE_ACADEMIC_YEARLY")
PRICE_DEPT_YEARLY = os.getenv("STRIPE_PRICE_DEPT_YEARLY")

PRICE_IDS = {
    "standard_monthly": PRICE_STANDARD_MONTHLY,
    "standard_yearly": PRICE_STANDARD_YEARLY,
    "pro_monthly":      PRICE_PRO_MONTHLY,
    "pro_yearly":       PRICE_PRO_YEARLY,
    "academic_yearly":  PRICE_ACADEMIC_YEARLY,
    "dept_yearly":      PRICE_DEPT_YEARLY,
}

# Optional: Auth0 (not required by the current Streamlit flow, but available)
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")  # your API identifier in Auth0
AUTH0_ISSUER = f"https://{AUTH0_DOMAIN}/" if AUTH0_DOMAIN else None

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

if not STRIPE_API_KEY:
    raise RuntimeError("STRIPE_API_KEY is not set")

stripe.api_key = STRIPE_API_KEY

# ============================================================
# Database setup (SQLAlchemy)
# ============================================================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    auth0_sub = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, index=True)
    stripe_customer_id = Column(String, index=True)
    stripe_subscription_id = Column(String, index=True)

    # plan can be either a Stripe price ID or a friendly label like 'standard' / 'pro'
    plan = Column(String)
    status = Column(String)  # 'trialing', 'active', 'canceled', etc.
    trial_end = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


# Auto-create tables in Render Postgres
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# Auth0 helpers (optional – not required by current flow)
# ============================================================

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
    """Not used by current endpoints, but handy if you later add /me endpoints."""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = parts[1]
    return verify_token(token)


# ============================================================
# FastAPI app + CORS
# ============================================================

app = FastAPI()

# For launch: allow all origins (you can lock this down later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://cubesimprov2-....streamlit.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# /sync-user – called by Streamlit after Auth0 login
#    body: { "auth0_sub": "...", "email": "...", "stripe_customer_id": "cus_xxx" (optional) }
# ============================================================

@app.post("/sync-user")
def sync_user(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
):
    auth0_sub = payload.get("auth0_sub")
    email = payload.get("email")
    stripe_customer_id = payload.get("stripe_customer_id")

    if not auth0_sub and not email:
        raise HTTPException(status_code=400, detail="Missing auth0_sub or email")

    # Prefer auth0_sub as primary key
    user = None
    if auth0_sub:
        user = db.query(User).filter(User.auth0_sub == auth0_sub).first()

    if not user and email:
        # Fallback: maybe we only had an email-based user before
        user = db.query(User).filter(User.email == email).first()

    if not user:
        user = User(
            auth0_sub=auth0_sub or (email or "unknown"),
            email=email,
            stripe_customer_id=stripe_customer_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(user)
    else:
        # Update fields
        if email and not user.email:
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
        "trial_end": user.trial_end.isoformat() if user.trial_end else None,
    }


# ============================================================
# /subscription-state – called by Streamlit to set plan on load
#    GET /subscription-state?auth0_sub=...
# ============================================================

@app.get("/subscription-state")
def subscription_state(
    auth0_sub: str,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.auth0_sub == auth0_sub).first()
    if not user:
        # No subscription known for this user yet
        return {
            "plan": "none",
            "status": "none",
            "price_id": None,
            "current_period_end": None,
            "customer_id": None,
        }

    return {
        "plan": user.plan or "none",
        "status": user.status or "none",
        "price_id": user.plan,  # you can change this if you prefer friendly names
        "current_period_end": user.trial_end.isoformat() if user.trial_end else None,
        "customer_id": user.stripe_customer_id,
    }


# ============================================================
# Helper: upsert user by Stripe customer (used by webhook)
# ============================================================

def upsert_user_from_stripe(
    db: Session,
    customer_id: str,
    subscription_id: str | None = None,
):
    # Fetch customer info from Stripe to get email
    stripe_customer = stripe.Customer.retrieve(customer_id)
    email = stripe_customer.get("email")

    user = None
    # First, try by Stripe customer ID
    user = db.query(User).filter(User.stripe_customer_id == customer_id).first()

    # Then by email (if present)
    if not user and email:
        user = db.query(User).filter(User.email == email).first()

    # Then by auth0_sub == email (if sync-user created it like that)
    if not user and email:
        user = db.query(User).filter(User.auth0_sub == email).first()

    if not user:
        # Create a new user row if nothing matches
        user = User(
            auth0_sub=email or f"stripe:{customer_id}",
            email=email,
            stripe_customer_id=customer_id,
            created_at=datetime.utcnow(),
        )
        db.add(user)
    else:
        # Just ensure we keep stripe_customer_id synced
        user.stripe_customer_id = customer_id

    if subscription_id:
        user.stripe_subscription_id = subscription_id

    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user


# ============================================================
# /create-checkout-session – called by Streamlit sidebar buttons
#    body: { "plan_key": "...", "email": "..." }
# ============================================================

@app.post("/create-checkout-session")
def create_checkout_session(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
):
    plan_key = payload.get("plan_key")
    email = payload.get("email")

    if not plan_key or not email:
        raise HTTPException(status_code=400, detail="Missing plan_key or email")

    price_id = PRICE_IDS.get(plan_key)
    if not price_id:
        raise HTTPException(status_code=400, detail=f"Unknown plan_key: {plan_key}")

    # Find or create Stripe Customer by email
    try:
        customers = stripe.Customer.list(email=email, limit=1).data
        if customers:
            customer = customers[0]
        else:
            customer = stripe.Customer.create(email=email)
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=500, detail=f"Stripe customer error: {e.user_message or str(e)}")

    customer_id = customer["id"]

    # Upsert user in DB (by email/auth0_sub), store stripe_customer_id
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            auth0_sub=email,  # we don't have sub here; this will be unified by /sync-user
            email=email,
            stripe_customer_id=customer_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(user)
    else:
        user.stripe_customer_id = customer_id
        user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)

    # Create Stripe Checkout Session
    success_url = f"{FRONTEND_BASE_URL}?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = FRONTEND_BASE_URL

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            customer=customer_id,
            line_items=[
                {
                    "price": price_id,
                    "quantity": 1,
                }
            ],
            allow_promotion_codes=True,
            success_url=success_url,
            cancel_url=cancel_url,
        )
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=500, detail=f"Stripe checkout error: {e.user_message or str(e)}")

    return {"url": session.url}


# ============================================================
# /create-portal-session – manage billing & invoices
#    body: { "customer_id": "cus_xxx" }
# ============================================================

@app.post("/create-portal-session")
def create_portal_session(
    payload: dict = Body(...),
):
    customer_id = payload.get("customer_id")
    if not customer_id:
        raise HTTPException(status_code=400, detail="Missing customer_id")

    return_url = FRONTEND_BASE_URL

    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=500, detail=f"Stripe portal error: {e.user_message or str(e)}")

    return {"url": portal_session.url}


# ============================================================
# /stripe/webhook – Stripe → backend (subscription updates)
# ============================================================

@app.post("/stripe/webhook")
async def stripe_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Stripe webhook secret not configured",
        )

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid webhook: {str(e)}")

    event_type = event["type"]
    data = event["data"]["object"]

    # Handle subscription lifecycle events
    if event_type in [
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    ]:
        sub = data
        customer_id = sub["customer"]
        subscription_id = sub["id"]
        status = sub["status"]           # 'active', 'trialing', 'canceled', ...
        plan_id = sub["items"]["data"][0]["plan"]["id"]
        trial_end = (
            datetime.fromtimestamp(sub["trial_end"]) if sub.get("trial_end") else None
        )

        user = upsert_user_from_stripe(db, customer_id, subscription_id=subscription_id)
        user.status = status
        user.plan = plan_id
        user.trial_end = trial_end
        user.updated_at = datetime.utcnow()
        db.commit()

    # You can add more event types here if needed

    return {"received": True}
