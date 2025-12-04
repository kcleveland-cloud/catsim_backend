# main.py — CATSIM backend for Auth0 + Stripe (Render)
#
# Endpoints:
#   POST /sync-user
#   POST /subscription-state
#   POST /create-checkout-session
#   POST /create-portal-session
#   POST /stripe-webhook
#
# This version uses Stripe as the source of truth:
# - We do NOT use a DB; we look up customers/subscriptions directly in Stripe.
# - Customers are keyed by email + metadata["user_id"] (Auth0 sub).

import os
import stripe
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Literal

# -------------------------
# Environment configuration
# -------------------------

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# Price IDs for your Standard/Pro subscription products (from Stripe dashboard)
STRIPE_PRICE_STANDARD = os.getenv("STRIPE_PRICE_STANDARD")  # e.g. price_123
STRIPE_PRICE_PRO = os.getenv("STRIPE_PRICE_PRO")            # e.g. price_456

# Frontend base URL (your Streamlit app)
FRONTEND_BASE_URL = os.getenv(
    "FRONTEND_BASE_URL",
    "https://cubesimprov2.streamlit.app"  # <-- adjust to your real URL
)

# Where Stripe Billing Portal sends the user back
PORTAL_RETURN_URL = os.getenv(
    "PORTAL_RETURN_URL",
    FRONTEND_BASE_URL.rstrip("/")        # default: just go back to the app
)

if not STRIPE_SECRET_KEY:
    raise RuntimeError("STRIPE_SECRET_KEY is not set.")

stripe.api_key = STRIPE_SECRET_KEY

# -------------------------
# FastAPI app & CORS
# -------------------------

app = FastAPI()

frontend_origin = FRONTEND_BASE_URL.rstrip("/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "*"],  # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Pydantic models
# -------------------------

class UserBase(BaseModel):
    user_id: str                # Auth0 sub
    email: str
    name: Optional[str] = None

class SyncUserRequest(UserBase):
    pass

class SubscriptionStateRequest(UserBase):
    pass

# class CheckoutSessionRequest(UserBase):
#    tier: Literal["standard", "pro"]

    # Optional override URLs; otherwise we derive from FRONTEND_BASE_URL
#    success_url: Optional[HttpUrl] = None
#    cancel_url: Optional[HttpUrl] = None

class PortalSessionRequest(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None

# -------------------------
# Stripe helper functions
# -------------------------

def find_or_create_customer(user_id: str, email: str, name: Optional[str]) -> stripe.Customer:
    """
    Find an existing Stripe customer by metadata.user_id or email.
    If not found, create a new one.
    """
    # 1) Try search by metadata.user_id
    try:
        search = stripe.Customer.search(
            query=f'metadata["user_id"]:"{user_id}"'
        )
        if search.data:
            return search.data[0]
    except Exception:
        # Search may not be enabled; fall through to email lookup
        pass

    # 2) Try list by email
    customers = stripe.Customer.list(email=email, limit=1)
    if customers.data:
        cust = customers.data[0]
        # Ensure metadata.user_id is set for future lookups
        metadata = dict(cust.metadata or {})
        if metadata.get("user_id") != user_id:
            metadata["user_id"] = user_id
            stripe.Customer.modify(cust.id, metadata=metadata)
        return cust

    # 3) No customer found → create new
    customer = stripe.Customer.create(
        email=email,
        name=name,
        metadata={"user_id": user_id},
    )
    return customer


def get_subscription_state_for_customer(customer: stripe.Customer) -> dict:
    """
    Return a simplified subscription state dict for the given Stripe customer.
    Looks at active subscriptions and picks the 'highest' plan it finds.
    """
    subs = stripe.Subscription.list(
        customer=customer.id,
        status="all",
        expand=["data.items.data.price.product"],
        limit=10,
    )

    plan = "free"
    status = "none"
    current_period_end = None
    raw_subscriptions = []

    for s in subs.auto_paging_iter():
        raw_subscriptions.append(
            {
                "id": s.id,
                "status": s.status,
                "cancel_at_period_end": s.cancel_at_period_end,
            }
        )
        if s.status in ("active", "trialing"):
            # We treat "pro" as higher than "standard".
            for item in s["items"]["data"]:
                price_id = item["price"]["id"]
                if price_id == STRIPE_PRICE_PRO:
                    plan = "pro"
                elif price_id == STRIPE_PRICE_STANDARD and plan != "pro":
                    plan = "standard"
            status = s.status
            current_period_end = s.current_period_end

    return {
        "plan": plan,
        "status": status,
        "current_period_end": current_period_end,
        "customer_id": customer.id,
        "subscriptions": raw_subscriptions,
    }

# -------------------------
# API endpoints
# -------------------------

@app.post("/sync-user")
def sync_user(body: SyncUserRequest):
    """
    Called by frontend when user logs in.
    Ensures a Stripe customer exists and returns subscription state.
    """
    try:
        customer = find_or_create_customer(body.user_id, body.email, body.name)
        sub_state = get_subscription_state_for_customer(customer)
        return {
            "ok": True,
            "customer_id": customer.id,
            "subscription": sub_state,
        }
    except Exception as e:
        print("sync_user error:", repr(e))
        raise HTTPException(status_code=500, detail=f"SYNC_USER_ERROR: {e}")


@app.post("/subscription-state")
def subscription_state(body: SubscriptionStateRequest):
    """
    Return the current subscription state for this user.
    """
    try:
        customer = find_or_create_customer(body.user_id, body.email, body.name)
        sub_state = get_subscription_state_for_customer(customer)
        return {
            "ok": True,
            "customer_id": customer.id,
            "subscription": sub_state,
        }
    except Exception as e:
        print("subscription_state error:", repr(e))
        raise HTTPException(status_code=500, detail=f"SUB_STATE_ERROR: {e}")


@app.post("/create-checkout-session")
async def create_checkout_session(request: Request):
    """
    Create a Stripe Checkout Session for Standard or Pro subscription.

    Accepts a very flexible JSON body from the frontend. Examples:
      {
        "user_id": "...",
        "email": "...",
        "name": "Keith",
        "tier": "pro"
      }

      or even:
      {
        "auth0_sub": "...",
        "email": "...",
        "plan": "standard"
      }
    """
    try:
        data = await request.json()

        # --- Extract user identity ---
        user_id = data.get("user_id") or data.get("auth0_sub")
        email = data.get("email")
        name = data.get("name")

        if not user_id or not email:
            raise HTTPException(
                status_code=400,
                detail="Missing user_id or email for checkout session."
            )

        # --- Determine tier/plan ---
        # Accepts "tier" or "plan", case-insensitive, defaults to "pro"
        raw_tier = (data.get("tier") or data.get("plan") or "pro").lower()
        if raw_tier not in ("standard", "pro"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier/plan '{raw_tier}', expected 'standard' or 'pro'."
            )
        tier = raw_tier

        # Map to Stripe price ID
        if tier == "standard":
            price_id = STRIPE_PRICE_STANDARD
        else:
            price_id = STRIPE_PRICE_PRO

        if not price_id:
            raise HTTPException(
                status_code=500,
                detail=f"Missing Stripe price id for tier '{tier}'. "
                       "Set STRIPE_PRICE_STANDARD / STRIPE_PRICE_PRO in env."
            )

        # --- Build URLs (allow overrides from frontend or use defaults) ---
        success_url = data.get("success_url") or (
            FRONTEND_BASE_URL.rstrip("/")
            + "/?checkout=success&session_id={CHECKOUT_SESSION_ID}"
        )
        cancel_url = data.get("cancel_url") or (
            FRONTEND_BASE_URL.rstrip("/")
            + "/?checkout=cancelled"
        )

        # --- Find or create Stripe customer ---
        customer = find_or_create_customer(user_id, email, name)

        # --- Create Stripe Checkout session ---
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

        return {"id": session.id, "url": session.url}

    except HTTPException:
        raise
    except Exception as e:
        print("create_checkout_session error:", repr(e))
        raise HTTPException(status_code=500, detail=f"CHECKOUT_ERROR: {e}")


@app.post("/create-portal-session")
def create_portal_session(body: PortalSessionRequest):
    """
    Create a Stripe Billing Portal session for the user.
    Frontend sends: { "user_id": "...", "email": "..." }
    """
    try:
        if not body.email and not body.user_id:
            raise HTTPException(
                status_code=400,
                detail="Need at least email or user_id to create billing portal session."
            )

        # If user_id is missing but we have email, we can still find the customer.
        user_id = body.user_id or "unknown"

        # Use a dummy name; not needed for portal.
        customer = find_or_create_customer(user_id, body.email or "", None)

        session = stripe.billing_portal.Session.create(
            customer=customer.id,
            return_url=PORTAL_RETURN_URL,
        )
        return {"url": session.url}
    except HTTPException:
        raise
    except Exception as e:
        print("create_portal_session error:", repr(e))
        raise HTTPException(status_code=500, detail=f"PORTAL_ERROR: {e}")


# -------------------------
# Stripe webhook
# -------------------------

@app.post("/stripe-webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
):
    """
    Stripe webhook endpoint.

    Configure this URL in your Stripe dashboard:
      https://<your-render-backend>/stripe-webhook

    And set STRIPE_WEBHOOK_SECRET to the signing secret.
    """
    payload = await request.body()
    sig_header = stripe_signature

    if not STRIPE_WEBHOOK_SECRET:
        # If you haven't configured it yet, just log and return 200
        print("Webhook received but STRIPE_WEBHOOK_SECRET is not set.")
        return {"received": True}

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except ValueError as e:
        print("Invalid payload:", e)
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        print("Invalid signature:", e)
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the events you care about. For now we just log them.
    event_type = event["type"]
    obj = event["data"]["object"]
    print(f"Stripe event received: {event_type}")

    # Example: you could inspect obj here and eventually sync to a DB
    # if event_type == "checkout.session.completed":
    #     ...
    # elif event_type == "customer.subscription.updated":
    #     ...
    # etc.

    return {"received": True}


# -------------------------
# Local dev entrypoint
# -------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
