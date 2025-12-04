# main.py — CATSIM Backend (Auth0 + Stripe + Postgres)
#
# Uses env vars (from your Render screenshot):
#   AUTH0_AUDIENCE          (not used directly yet)
#   AUTH0_DOMAIN            (not used directly yet)
#   DATABASE_URL
#   FRONTEND_URL
#   PRICE_ACADEMIC_YEARLY
#   PRICE_DEPT_YEARLY
#   PRICE_PRO_MONTHLY
#   PRICE_PRO_YEARLY
#   PRICE_STANDARD_MONTHLY
#   PRICE_STANDARD_YEARLY
#   STRIPE_API_KEY or STRIPE_SECRET_KEY
#   STRIPE_WEBHOOK_SECRET
#
# Endpoints:
#   POST /sync-user
#   GET  /subscription-state
#   POST /subscription-state
#   POST /create-checkout-session
#   POST /create-portal-session
#   POST /stripe-webhook
#
# DB tables:
#   users           – Auth0 user + Stripe customer ID
#   subscriptions   – Stripe subscription ID + plan_key + status + raw JSON

import os
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import stripe
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from sqlalchemy.types import JSON as SAJSON

# ==========================================================
# Environment & Stripe setup
# ==========================================================

# Stripe secret key (either STRIPE_SECRET_KEY or STRIPE_API_KEY)
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
if not STRIPE_SECRET_KEY:
    raise RuntimeError("STRIPE_SECRET_KEY or STRIPE_API_KEY must be set.")
stripe.api_key = STRIPE_SECRET_KEY

# Frontend URL (your Streamlit app)
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://cubesimprov2.streamlit.app").rstrip(
    "/"
)

# Billing portal return URL
PORTAL_RETURN_URL = os.getenv("PORTAL_RETURN_URL", FRONTEND_URL)

# Prices (exactly as in your Render env)
PRICE_STANDARD_MONTHLY = os.getenv("PRICE_STANDARD_MONTHLY")
PRICE_STANDARD_YEARLY = os.getenv("PRICE_STANDARD_YEARLY")
PRICE_PRO_MONTHLY = os.getenv("PRICE_PRO_MONTHLY")
PRICE_PRO_YEARLY = os.getenv("PRICE_PRO_YEARLY")
PRICE_ACADEMIC_YEARLY = os.getenv("PRICE_ACADEMIC_YEARLY")
PRICE_DEPT_YEARLY = os.getenv("PRICE_DEPT_YEARLY")

# plan_key -> Stripe price_id
PRICE_MAP: Dict[str, Optional[str]] = {
    "standard_monthly": PRICE_STANDARD_MONTHLY,
    "standard_yearly": PRICE_STANDARD_YEARLY,
    "pro_monthly": PRICE_PRO_MONTHLY,
    "pro_yearly": PRICE_PRO_YEARLY,
    "academic_yearly": PRICE_ACADEMIC_YEARLY,
    "dept_yearly": PRICE_DEPT_YEARLY,
}

# Stripe price_id -> plan_key (for subscription-state mapping)
PRICE_ID_TO_PLAN_KEY: Dict[str, str] = {
    price_id: key for key, price_id in PRICE_MAP.items() if price_id
}

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# ==========================================================
# Database setup (Postgres via DATABASE_URL)
# ==========================================================

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be set.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    auth0_sub = Column(String(255), unique=True, index=True, nullable=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    stripe_customer_id = Column(String(255), index=True, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    subscriptions = relationship(
        "Subscription", back_populates="user", cascade="all, delete-orphan"
    )


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    stripe_subscription_id = Column(String(255), unique=True, index=True)
    plan_key = Column(String(64), nullable=True)  # e.g., 'pro_monthly'
    status = Column(String(64), nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    cancel_at_period_end = Column(Boolean, default=False)

    raw = Column(SAJSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    user = relationship("User", back_populates="subscriptions")


# Create tables if they don't exist
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================================
# FastAPI app & CORS
# ==========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Helper functions: Users, Customers, Subscriptions
# ==========================================================


def utc_from_timestamp(ts: Optional[int]) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def get_or_create_user(
    db: Session, auth0_sub: Optional[str], email: str, name: Optional[str]
) -> User:
    user = None
    if auth0_sub:
        user = (
            db.query(User)
            .filter(User.auth0_sub == auth0_sub)
            .one_or_none()
        )

    if not user:
        user = (
            db.query(User)
            .filter(User.email == email)
            .one_or_none()
        )

    if user:
        # Update basics if changed
        updated = False
        if auth0_sub and user.auth0_sub != auth0_sub:
            user.auth0_sub = auth0_sub
            updated = True
        if name and user.name != name:
            user.name = name
            updated = True
        if updated:
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    # Create new user
    user = User(
        auth0_sub=auth0_sub,
        email=email,
        name=name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def find_or_create_stripe_customer(db: Session, user: User) -> stripe.Customer:
    # If we already have a stored customer ID, try to fetch it
    if user.stripe_customer_id:
        try:
            customer = stripe.Customer.retrieve(user.stripe_customer_id)
            return customer
        except Exception:
            # If Stripe says it doesn't exist, we'll fall through and recreate
            pass

    # Try finding an existing customer by email
    customers = stripe.Customer.list(email=user.email, limit=1)
    if customers.data:
        customer = customers.data[0]
        # Ensure metadata has auth0_sub
        metadata = dict(customer.metadata or {})
        if user.auth0_sub and metadata.get("auth0_sub") != user.auth0_sub:
            metadata["auth0_sub"] = user.auth0_sub
            stripe.Customer.modify(customer.id, metadata=metadata)
    else:
        # Create a new Stripe customer
        customer = stripe.Customer.create(
            email=user.email,
            name=user.name,
            metadata={"auth0_sub": user.auth0_sub or ""},
        )

    # Store customer ID in DB
    user.stripe_customer_id = customer.id
    db.add(user)
    db.commit()
    db.refresh(user)
    return customer


def sync_subscriptions_from_stripe(
    db: Session, user: User, customer: stripe.Customer
) -> List[Subscription]:
    """
    Pull subscriptions from Stripe and upsert into DB for this user.
    """
    subs = stripe.Subscription.list(
        customer=customer.id,
        status="all",
        # Removed deep expand to avoid Stripe depth error
        # expand=["data.items.data.price.product"],
        limit=20,
    )

    existing = {
        s.stripe_subscription_id: s
        for s in user.subscriptions
    }

    result: List[Subscription] = []

    for s in subs.auto_paging_iter():
        # Normalize to a plain dict so we can safely access keys
        s_dict = s.to_dict() if hasattr(s, "to_dict") else dict(s)

        sub_id = s_dict["id"]
        plan_key = None

        # Determine plan_key from line items using PRICE_ID_TO_PLAN_KEY
        items = s_dict.get("items", {}).get("data", []) or []
        for item in items:
            price = item.get("price") or {}
            price_id = price.get("id")
            if price_id and price_id in PRICE_ID_TO_PLAN_KEY:
                plan_key = PRICE_ID_TO_PLAN_KEY[price_id]
                break

        sub_obj = existing.get(sub_id)
        if not sub_obj:
            sub_obj = Subscription(
                user_id=user.id,
                stripe_subscription_id=sub_id,
            )

        sub_obj.plan_key = plan_key
        sub_obj.status = s_dict.get("status")
        sub_obj.current_period_end = utc_from_timestamp(
            s_dict.get("current_period_end")
        )
        sub_obj.cancel_at_period_end = bool(
            s_dict.get("cancel_at_period_end") or False
        )
        sub_obj.raw = s_dict

        db.add(sub_obj)
        result.append(sub_obj)

    db.commit()
    return result



def compute_subscription_state(user: User) -> Dict[str, Any]:
    """
    Compute high-level plan/status from user's Subscription rows.
    Priority: dept > pro > academic > standard > free
    """

    plan_order = {
        "dept_yearly": 4,
        "pro_yearly": 3,
        "pro_monthly": 3,
        "academic_yearly": 2,
        "standard_yearly": 1,
        "standard_monthly": 1,
    }

    best_plan_key = None
    best_score = -1
    status = "none"
    current_period_end = None

    for s in user.subscriptions:
        if s.status in ("active", "trialing"):
            key = s.plan_key
            score = plan_order.get(key or "", 0)
            if score > best_score:
                best_score = score
                best_plan_key = key
                status = s.status
                current_period_end = s.current_period_end

    if not best_plan_key:
        return {
            "plan_key": None,
            "human_readable": "free",
            "status": "none",
            "current_period_end": None,
        }

    # Translate plan_key to something you can show in UI
    human = best_plan_key.replace("_", " ").title()
    return {
        "plan_key": best_plan_key,
        "human_readable": human,
        "status": status,
        "current_period_end": current_period_end,
    }


def sync_and_get_subscription_state(
    db: Session, user: User
) -> Dict[str, Any]:
    if not user.stripe_customer_id:
        # No customer → no subs yet
        return {
            "plan_key": None,
            "human_readable": "free",
            "status": "none",
            "current_period_end": None,
        }

    customer = stripe.Customer.retrieve(user.stripe_customer_id)
    sync_subscriptions_from_stripe(db, user, customer)
    return compute_subscription_state(user)


# ==========================================================
# API endpoints
# ==========================================================


@app.post("/sync-user")
async def sync_user(request: Request, db: Session = Depends(get_db)):
    """
    Called by frontend when user logs in.
    Body can be flexible, e.g.:

      {
        "user_id": "auth0|123",
        "sub": "auth0|123",
        "email": "user@example.com",
        "name": "User Name"
      }
    """
    try:
        try:
            data = await request.json()
        except Exception:
            data = {}

        print("sync_user incoming:", data)

        auth0_sub = (
            data.get("user_id")
            or data.get("auth0_sub")
            or data.get("sub")
        )
        email = data.get("email")
        name = data.get("name")

        if not email:
            raise HTTPException(
                status_code=400,
                detail="sync-user requires at least 'email' field.",
            )

        user = get_or_create_user(db, auth0_sub, email, name)
        customer = find_or_create_stripe_customer(db, user)
        sub_state = sync_and_get_subscription_state(db, user)

        return {
            "ok": True,
            "user": {
                "id": user.id,
                "auth0_sub": user.auth0_sub,
                "email": user.email,
                "name": user.name,
            },
            "customer_id": customer.id,
            "subscription": sub_state,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("sync_user error:", repr(e))
        raise HTTPException(status_code=500, detail=f"SYNC_USER_ERROR: {e}")


@app.get("/subscription-state")
def subscription_state_get(
    auth0_sub: Optional[str] = None,
    email: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    GET version: /subscription-state?auth0_sub=...&email=...
    """
    try:
        user = None
        if auth0_sub:
            user = (
                db.query(User)
                .filter(User.auth0_sub == auth0_sub)
                .one_or_none()
            )
        if not user and email:
            user = (
                db.query(User)
                .filter(User.email == email)
                .one_or_none()
            )

        if not user:
            # Not seen before -> effectively free
            return {
                "ok": True,
                "subscription": {
                    "plan_key": None,
                    "human_readable": "free",
                    "status": "none",
                    "current_period_end": None,
                },
            }

        sub_state = sync_and_get_subscription_state(db, user)
        return {"ok": True, "subscription": sub_state}
    except Exception as e:
        print("subscription_state GET error:", repr(e))
        raise HTTPException(status_code=500, detail=f"SUB_STATE_ERROR: {e}")


@app.post("/subscription-state")
async def subscription_state_post(
    request: Request, db: Session = Depends(get_db)
):
    """
    POST version: accepts JSON with user identifiers.
    """
    try:
        try:
            data = await request.json()
        except Exception:
            data = {}

        print("subscription_state POST incoming:", data)

        auth0_sub = (
            data.get("user_id")
            or data.get("auth0_sub")
            or data.get("sub")
        )
        email = data.get("email")

        user = None
        if auth0_sub:
            user = (
                db.query(User)
                .filter(User.auth0_sub == auth0_sub)
                .one_or_none()
            )
        if not user and email:
            user = (
                db.query(User)
                .filter(User.email == email)
                .one_or_none()
            )

        if not user:
            return {
                "ok": True,
                "subscription": {
                    "plan_key": None,
                    "human_readable": "free",
                    "status": "none",
                    "current_period_end": None,
                },
            }

        sub_state = sync_and_get_subscription_state(db, user)
        return {"ok": True, "subscription": sub_state}
    except Exception as e:
        print("subscription_state POST error:", repr(e))
        raise HTTPException(status_code=500, detail=f"SUB_STATE_ERROR: {e}")


@app.post("/create-checkout-session")
async def create_checkout_session(
    request: Request, db: Session = Depends(get_db)
):
    """
    Create a Stripe Checkout Session for a selected plan.

    Accepts flexible JSON, e.g.:

      {
        "plan_key": "pro_monthly",
        "email": "user@example.com",
        "user_id": "auth0|123",
        "name": "User Name",
        "success_url": "...",   (optional)
        "cancel_url": "..."     (optional)
      }
    """
    try:
        data = await request.json()
        print("create_checkout_session incoming:", data)

        plan_key = (
            data.get("plan_key")
            or data.get("plan")
            or data.get("tier")
            or "pro_monthly"
        )
        plan_key = str(plan_key).strip().lower()

        if plan_key not in PRICE_MAP:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown plan_key '{plan_key}'. "
                    "Expected one of: "
                    + ", ".join(sorted(PRICE_MAP.keys()))
                ),
            )

        price_id = PRICE_MAP[plan_key]
        if not price_id:
            raise HTTPException(
                status_code=400,
                detail=f"No Stripe price configured for plan_key '{plan_key}'.",
            )

        auth0_sub = (
            data.get("user_id")
            or data.get("auth0_sub")
            or data.get("sub")
        )
        email = data.get("email")
        name = data.get("name")

        if not email:
            raise HTTPException(
                status_code=400,
                detail="create-checkout-session requires 'email'.",
            )

        user = get_or_create_user(db, auth0_sub, email, name)
        customer = find_or_create_stripe_customer(db, user)

        success_url = data.get("success_url") or (
            FRONTEND_URL
            + "/?checkout=success&session_id={CHECKOUT_SESSION_ID}"
        )
        cancel_url = data.get("cancel_url") or (
            FRONTEND_URL + "/?checkout=cancelled"
        )

        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer.id,
            line_items=[
                {
                    "price": price_id,
                    "quantity": 1,
                }
            ],
            success_url=success_url,
            cancel_url=cancel_url,
            allow_promotion_codes=True,
        )

        print("Created checkout session:", session.id)
        return {"id": session.id, "url": session.url}

    except HTTPException:
        raise
    except Exception as e:
        print("create_checkout_session error:", repr(e))
        raise HTTPException(status_code=500, detail=f"CHECKOUT_ERROR: {e}")


@app.post("/create-portal-session")
async def create_portal_session(
    request: Request, db: Session = Depends(get_db)
):
    """
    Create a Stripe Billing Portal session for the user.

    Body can be flexible, e.g.:

      { "user_id": "auth0|123", "email": "user@example.com" }
    """
    try:
        try:
            data = await request.json()
        except Exception:
            data = {}

        print("create_portal_session incoming:", data)

        auth0_sub = (
            data.get("user_id")
            or data.get("auth0_sub")
            or data.get("sub")
        )
        email = data.get("email")
        name = data.get("name")

        if not email and not auth0_sub:
            raise HTTPException(
                status_code=400,
                detail="Need at least email or user_id/auth0_sub/sub to open billing portal.",
            )

        # If user exists, use it; otherwise create (so we can attach customer ID)
        if not email:
            # If no email but have auth0_sub, try to find user
            user = (
                db.query(User)
                .filter(User.auth0_sub == auth0_sub)
                .one_or_none()
            )
            if not user:
                raise HTTPException(
                    status_code=400,
                    detail="No email and no known user; cannot open portal.",
                )
        else:
            user = get_or_create_user(db, auth0_sub, email, name)

        customer = find_or_create_stripe_customer(db, user)

        session = stripe.billing_portal.Session.create(
            customer=customer.id,
            return_url=PORTAL_RETURN_URL,
        )

        print("Created billing portal session:", session.url)
        return {"url": session.url}

    except HTTPException:
        raise
    except Exception as e:
        print("create_portal_session error:", repr(e))
        raise HTTPException(status_code=500, detail=f"PORTAL_ERROR: {e}")


# ==========================================================
# Stripe webhook
# ==========================================================


@app.post("/stripe-webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
):
    # Just call the main handler
    return await stripe_webhook(request, stripe_signature)

@app.post("/stripe/webhook")
async def stripe_webhook_alias(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
):
    # Reuse the main handler so logic stays in one place
    return await stripe_webhook(request, stripe_signature)
    
    """
    Basic Stripe webhook endpoint.

    Configure in Stripe:
      https://<your-backend>/stripe-webhook

    Handles checkout.session.completed & customer.subscription.* to keep DB
    in sync. (For now we just sync from Stripe on demand — this can be
    expanded later.)
    """
    payload = await request.body()

    if not STRIPE_WEBHOOK_SECRET:
        print("Webhook received but STRIPE_WEBHOOK_SECRET is not set.")
        return {"received": True}

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=stripe_signature,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except Exception as e:
        print("stripe_webhook signature error:", repr(e))
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event_type = event["type"]
    obj = event["data"]["object"]
    print("Stripe event:", event_type)

    # For now we just log events; your /sync-user & /subscription-state
    # endpoints pull truth from Stripe and update DB.
    # If you want, this can later be extended to upsert users/subs directly.

    return {"received": True}


# ==========================================================
# Local dev entrypoint
# ==========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
