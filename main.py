import os
from datetime import datetime
from functools import lru_cache
from typing import Optional

import requests
import stripe
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt
from pydantic import BaseModel
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
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")  # your API identifier in Auth0
AUTH0_ISSUER = f"https://{AUTH0_DOMAIN}/" if AUTH0_DOMAIN else None

# Stripe price IDs (set these in Render)
PRICE_STANDARD_MONTHLY = os.getenv("PRICE_STANDARD_MONTHLY")
PRICE_STANDARD_YEARLY = os.getenv("PRICE_STANDARD_YEARLY")
PRICE_PRO_MONTHLY = os.getenv("PRICE_PRO_MONTHLY")
PRICE_PRO_YEARLY = os.getenv("PRICE_PRO_YEARLY")
PRICE_ACADEMIC_YEARLY = os.getenv("PRICE_ACADEMIC_YEARLY")
PRICE_DEPT_YEARLY = os.getenv("PRICE_DEPT_YEARLY")

# Frontend URL (your Streamlit app)
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://cubesimprov2-lt6hcgkvpdvygnwbktyqdg.streamlit.app",
)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

if not STRIPE_API_KEY:
    raise RuntimeError("STRIPE_API_KEY is not set")

stripe.api_key = STRIPE_API_KEY

# Map plan_key -> Stripe price ID (used by /create-checkout-session)
PLAN_TO_PRICE = {
    "standard_monthly": PRICE_STANDARD_MONTHLY,
    "standard_yearly": PRICE_STANDARD_YEARLY,
    "pro_monthly": PRICE_PRO_MONTHLY,
    "pro_yearly": PRICE_PRO_YEARLY,
    "academic_yearly": PRICE_ACADEMIC_YEARLY,
    "dept_yearly": PRICE_DEPT_YEARLY,
}

# Map Stripe price/plan IDs -> logical plan ("standard" / "pro")
PRICE_TO_PLAN = {}
for pid in [
    PRICE_STANDARD_MONTHLY,
    PRICE_STANDARD_YEARLY,
]:
    if pid:
        PRICE_TO_PLAN[pid] = "standard"

for pid in [
    PRICE_PRO_MONTHLY,
    PRICE_PRO_YEARLY,
    PRICE_ACADEMIC_YEARLY,  # treat academic as Pro-level features
    PRICE_DEPT_YEARLY,      # department license also Pro-level
]:
    if pid:
        PRICE_TO_PLAN[pid] = "pro"


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
# Pydantic models for requests
# =========================
class SyncUserRequest(BaseModel):
    auth0_sub: str
    email: Optional[str] = None
    stripe_customer_id: Optional[str] = None


class CheckoutSessionRequest(BaseModel):
    plan_key: str
    email: str
    auth0_sub: Optional[str] = None  # optional, front-end can add later


class PortalSessionRequest(BaseModel):
    customer_id: str


# =========================
# /sync-user – called by Streamlit on login
# =========================
@app.post("/sync-user")
def sync_user(req: SyncUserRequest, db: Session = Depends(get_db)):
    """
    Best-effort sync of Auth0 identity into the backend DB.

    Body:
      {
        "auth0_sub": "...",
        "email": "...",
        "stripe_customer_id": "cus_xxx" (optional)
      }
    """
    user = db.query(User).filter(User.auth0_sub == req.auth0_sub).first()
    if not user:
        user = User(
            auth0_sub=req.auth0_sub,
            email=req.email,
            stripe_customer_id=req.stripe_customer_id,
        )
        db.add(user)
    else:
        # Update email / stripe_customer_id if provided
        if req.email and user.email != req.email:
            user.email = req.email
        if req.stripe_customer_id and user.stripe_customer_id != req.stripe_customer_id:
            user.stripe_customer_id = req.stripe_customer_id

    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return {"ok": True}


# =========================
# /subscription-state – used by Streamlit sidebar
# =========================
@app.get("/subscription-state")
def subscription_state(auth0_sub: str, db: Session = Depends(get_db)):
    """
    Returns the subscription state for a given Auth0 subject.

    Query param: ?auth0_sub=...
    Returns:
      {
        "plan": "standard" | "pro" | "none",
        "status": "active" | "trialing" | "canceled" | "none",
        "price_id": "price_xxx" | null,
        "current_period_end": "2025-01-01T00:00:00Z" | null,
        "customer_id": "cus_xxx" | null
      }
    """
    user = db.query(User).filter(User.auth0_sub == auth0_sub).first()
    if not user:
        return {
            "plan": "none",
            "status": "none",
            "price_id": None,
            "current_period_end": None,
            "customer_id": None,
        }

    price_id = None
    plan = "none"
    status = user.status or "none"
    current_period_end_iso = None
    customer_id = user.stripe_customer_id

    if user.stripe_subscription_id:
        try:
            sub = stripe.Subscription.retrieve(user.stripe_subscription_id)
            status = sub["status"]
            # Prefer price.id; fall back to plan.id if needed
            item = sub["items"]["data"][0]
            price_obj = item.get("price")
            plan_obj = item.get("plan")
            price_id = (price_obj or {}).get("id") or (plan_obj or {}).get("id")
            cpe = sub.get("current_period_end")
            if cpe:
                current_period_end_iso = datetime.utcfromtimestamp(cpe).isoformat() + "Z"
        except Exception:
            # fall back to whatever we stored
            price_id = user.plan
            current_period_end_iso = user.trial_end.isoformat() + "Z" if user.trial_end else None

    # Map price_id -> logical plan name
    if price_id and price_id in PRICE_TO_PLAN:
        plan = PRICE_TO_PLAN[price_id]
    else:
        # fall back if user.plan already stores "standard"/"pro"
        if user.plan in ("standard", "pro"):
            plan = user.plan

    return {
        "plan": plan,
        "status": status,
        "price_id": price_id,
        "current_period_end": current_period_end_iso,
        "customer_id": customer_id,
    }


# =========================
# /create-checkout-session – used by Streamlit to start Stripe Checkout
# =========================
@app.post("/create-checkout-session")
def create_checkout_session(
    req: CheckoutSessionRequest,
    db: Session = Depends(get_db),
):
    """
    Body:
      {
        "plan_key": "standard_monthly" | "standard_yearly" | "pro_monthly" | "pro_yearly" | "academic_yearly" | "dept_yearly",
        "email": "...",
        "auth0_sub": "..." (optional)
      }
    Returns:
      { "url": "https://checkout.stripe.com/..." }
    """
    price_id = PLAN_TO_PRICE.get(req.plan_key)
    if not price_id:
        raise HTTPException(status_code=400, detail=f"Unknown plan_key: {req.plan_key}")

    # Upsert user by auth0_sub or email (best effort)
    user = None
    if req.auth0_sub:
        user = db.query(User).filter(User.auth0_sub == req.auth0_sub).first()
    if not user and req.email:
        user = db.query(User).filter(User.email == req.email).first()

    if not user:
        if not req.auth0_sub:
            # fallback if no auth0_sub yet
            req.auth0_sub = f"email:{req.email}"
        user = User(auth0_sub=req.auth0_sub, email=req.email)
        db.add(user)
        db.commit()
        db.refresh(user)

    # Create or reuse Stripe customer
    customer_id = user.stripe_customer_id
    if customer_id:
        try:
            stripe.Customer.retrieve(customer_id)
        except Exception:
            customer_id = None

    if not customer_id:
        customer = stripe.Customer.create(email=req.email)
        customer_id = customer["id"]
        user.stripe_customer_id = customer_id
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)

    success_url = f"{FRONTEND_URL}?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = FRONTEND_URL

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "auth0_sub": req.auth0_sub or "",
                "plan_key": req.plan_key,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe error: {e}")

    return {"url": session.url}


# =========================
# /create-portal-session – Stripe Billing Portal
# =========================
@app.post("/create-portal-session")
def create_portal_session(req: PortalSessionRequest):
    """
    Body:
      { "customer_id": "cus_xxx" }
    Returns:
      { "url": "https://billing.stripe.com/..." }
    """
    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=req.customer_id,
            return_url=FRONTEND_URL,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe portal error: {e}")

    return {"url": portal_session.url}


# =========================
# /me/subscription – (Auth0-protected) optional
# =========================
@app.get("/me/subscription")
def get_my_subscription(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Alternative, Auth0-protected subscription endpoint.
    Not currently used by Streamlit front-end.
    """
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

        item = sub["items"]["data"][0]
        price_obj = item.get("price")
        plan_obj = item.get("plan")
        plan_id = (price_obj or {}).get("id") or (plan_obj or {}).get("id")

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
