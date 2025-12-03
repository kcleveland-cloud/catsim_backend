import os
from datetime import datetime
from typing import Optional

import stripe
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# =========================
# Environment variables
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://cubesimprov2-noruuoxdtsrjzdskhuobbr.streamlit.app",
)

# These match your Render screenshot: PRICE_STANDARD_MONTHLY, etc.
PRICE_MAP = {
    "standard_monthly": os.getenv("PRICE_STANDARD_MONTHLY"),
    "standard_yearly": os.getenv("PRICE_STANDARD_YEARLY"),
    "pro_monthly": os.getenv("PRICE_PRO_MONTHLY"),
    "pro_yearly": os.getenv("PRICE_PRO_YEARLY"),
    "academic_yearly": os.getenv("PRICE_ACADEMIC_YEARLY"),
    "dept_yearly": os.getenv("PRICE_DEPT_YEARLY"),
}

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
    auth0_sub = Column(String, unique=True, index=True, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)
    stripe_customer_id = Column(String, index=True)
    stripe_subscription_id = Column(String)
    plan = Column(String)     # Stripe price ID or 'standard'/'pro'
    status = Column(String)   # 'trialing', 'active', 'canceled', etc.
    trial_end = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# =========================
# FastAPI app
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later: restrict to FRONTEND_URL
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
# Schemas
# =========================
class SyncUserPayload(BaseModel):
    auth0_sub: Optional[str] = None
    email: str
    stripe_customer_id: Optional[str] = None


class CheckoutPayload(BaseModel):
    plan_key: str
    email: str


class PortalPayload(BaseModel):
    customer_id: str


# =========================
# Health & debug
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/prices")
def debug_prices():
    """Show which Stripe prices the backend sees (for debugging only)."""
    return PRICE_MAP


# =========================
# Simple user sync from Streamlit
# =========================
@app.post("/sync-user")
def sync_user(payload: SyncUserPayload, db: Session = Depends(get_db)):
    if not payload.email:
        raise HTTPException(status_code=400, detail="Email is required")

    email = payload.email.strip().lower()

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            auth0_sub=payload.auth0_sub,
            email=email,
            stripe_customer_id=payload.stripe_customer_id,
            created_at=datetime.utcnow(),
        )
        db.add(user)
    else:
        if payload.auth0_sub and not user.auth0_sub:
            user.auth0_sub = payload.auth0_sub
        if payload.stripe_customer_id:
            user.stripe_customer_id = payload.stripe_customer_id

    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)

    return {"ok": True}


# =========================
# Subscription state for Streamlit
# =========================
@app.get("/subscription-state")
def subscription_state(
    auth0_sub: Optional[str] = None,
    email: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Called from Streamlit: we pass auth0_sub; falls back to email if needed.
    """
    q = db.query(User)
    user = None

    if auth0_sub:
        user = q.filter(User.auth0_sub == auth0_sub).first()
    if not user and email:
        user = q.filter(User.email == email.strip().lower()).first()

    if not user:
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
        "price_id": user.plan,
        "current_period_end": user.trial_end.isoformat() if user.trial_end else None,
        "customer_id": user.stripe_customer_id,
    }


# =========================
# Stripe Checkout (subscriptions)
# =========================
@app.post("/create-checkout-session")
def create_checkout_session(payload: CheckoutPayload, db: Session = Depends(get_db)):
    plan_key = payload.plan_key
    email = payload.email.strip().lower()

    if plan_key not in PRICE_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown plan_key: {plan_key}")

    price_id = PRICE_MAP[plan_key]
    if not price_id:
        raise HTTPException(
            status_code=500,
            detail=f"Missing Stripe price id for {plan_key}. "
                   f"Check PRICE_* env vars in Render.",
        )

    # find or create customer by email
    customers = stripe.Customer.list(email=email, limit=1).data
    if customers:
        customer = customers[0]
    else:
        customer = stripe.Customer.create(email=email)

    success_url = f"{FRONTEND_URL}?checkout=success"
    cancel_url = f"{FRONTEND_URL}?checkout=cancel"

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer.id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            allow_promotion_codes=True,
        )
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Upsert user row with customer id
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            email=email,
            stripe_customer_id=customer.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(user)
    else:
        user.stripe_customer_id = customer.id
        user.updated_at = datetime.utcnow()

    db.commit()

    return {"url": session.url}


# =========================
# Stripe Billing Portal
# =========================
@app.post("/create-portal-session")
def create_portal_session(payload: PortalPayload):
    if not payload.customer_id:
        raise HTTPException(status_code=400, detail="customer_id is required")

    try:
        portal = stripe.billing_portal.Session.create(
            customer=payload.customer_id,
            return_url=FRONTEND_URL,
        )
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"url": portal.url}


# =========================
# Stripe webhook – update DB on subscription events
# =========================
@app.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=500,
            detail="STRIPE_WEBHOOK_SECRET not configured",
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
        email = (stripe_customer.get("email") or "").lower()

        user = None
        if email:
            user = db.query(User).filter(User.email == email).first()

        if not user:
            user = User(
                auth0_sub=email or f"stripe:{customer_id}",
                email=email,
                stripe_customer_id=customer_id,
                created_at=datetime.utcnow(),
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
