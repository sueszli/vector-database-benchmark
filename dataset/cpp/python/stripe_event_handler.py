import logging
from contextlib import suppress
from typing import Any, Callable, Dict, Union

import stripe
from django.conf import settings

from corporate.lib.stripe import (
    BillingError,
    RealmBillingSession,
    UpgradeWithExistingPlanError,
    ensure_customer_does_not_have_active_plan,
)
from corporate.models import CustomerPlan, Event, PaymentIntent, Session
from zerver.models import get_active_user_profile_by_id_in_realm

billing_logger = logging.getLogger("corporate.stripe")


def error_handler(
    func: Callable[[Any, Any], None],
) -> Callable[[Union[stripe.checkout.Session, stripe.PaymentIntent], Event], None]:
    def wrapper(
        stripe_object: Union[stripe.checkout.Session, stripe.PaymentIntent], event: Event
    ) -> None:
        event.status = Event.EVENT_HANDLER_STARTED
        event.save(update_fields=["status"])

        try:
            func(stripe_object, event.content_object)
        except BillingError as e:
            billing_logger.warning(
                "BillingError in %s event handler: %s. stripe_object_id=%s, customer_id=%s metadata=%s",
                event.type,
                e.error_description,
                stripe_object.id,
                stripe_object.customer,
                stripe_object.metadata,
            )
            event.status = Event.EVENT_HANDLER_FAILED
            event.handler_error = {
                "message": e.msg,
                "description": e.error_description,
            }
            event.save(update_fields=["status", "handler_error"])
        except Exception:
            billing_logger.exception(
                "Uncaught exception in %s event handler:",
                event.type,
                stack_info=True,
            )
            event.status = Event.EVENT_HANDLER_FAILED
            event.handler_error = {
                "description": f"uncaught exception in {event.type} event handler",
                "message": BillingError.CONTACT_SUPPORT.format(email=settings.ZULIP_ADMINISTRATOR),
            }
            event.save(update_fields=["status", "handler_error"])
        else:
            event.status = Event.EVENT_HANDLER_SUCCEEDED
            event.save()

    return wrapper


@error_handler
def handle_checkout_session_completed_event(
    stripe_session: stripe.checkout.Session, session: Session
) -> None:
    session.status = Session.COMPLETED
    session.save()

    assert isinstance(stripe_session.setup_intent, str)
    stripe_setup_intent = stripe.SetupIntent.retrieve(stripe_session.setup_intent)
    assert session.customer.realm is not None
    assert stripe_session.metadata is not None
    user_id = stripe_session.metadata.get("user_id")
    assert user_id is not None
    user = get_active_user_profile_by_id_in_realm(int(user_id), session.customer.realm)
    billing_session = RealmBillingSession(user)
    payment_method = stripe_setup_intent.payment_method
    assert isinstance(payment_method, (str, type(None)))

    if session.type in [
        Session.UPGRADE_FROM_BILLING_PAGE,
        Session.RETRY_UPGRADE_WITH_ANOTHER_PAYMENT_METHOD,
    ]:
        ensure_customer_does_not_have_active_plan(session.customer)
        billing_session.update_or_create_stripe_customer(payment_method)
        assert session.payment_intent is not None
        session.payment_intent.status = PaymentIntent.PROCESSING
        session.payment_intent.last_payment_error = ()
        session.payment_intent.save(update_fields=["status", "last_payment_error"])
        with suppress(stripe.error.CardError):
            stripe.PaymentIntent.confirm(
                session.payment_intent.stripe_payment_intent_id,
                payment_method=payment_method,
                off_session=True,
            )
    elif session.type in [
        Session.FREE_TRIAL_UPGRADE_FROM_BILLING_PAGE,
        Session.FREE_TRIAL_UPGRADE_FROM_ONBOARDING_PAGE,
    ]:
        ensure_customer_does_not_have_active_plan(session.customer)
        billing_session.update_or_create_stripe_customer(payment_method)
        billing_session.process_initial_upgrade(
            CustomerPlan.STANDARD,
            int(stripe_session.metadata["licenses"]),
            stripe_session.metadata["license_management"] == "automatic",
            int(stripe_session.metadata["billing_schedule"]),
            charge_automatically=True,
            free_trial=True,
        )
    elif session.type in [Session.CARD_UPDATE_FROM_BILLING_PAGE]:
        billing_session.update_or_create_stripe_customer(payment_method)


@error_handler
def handle_payment_intent_succeeded_event(
    stripe_payment_intent: stripe.PaymentIntent, payment_intent: PaymentIntent
) -> None:
    payment_intent.status = PaymentIntent.SUCCEEDED
    payment_intent.save()
    metadata: Dict[str, Any] = stripe_payment_intent.metadata
    assert payment_intent.customer.realm is not None
    user_id = metadata.get("user_id")
    assert user_id is not None
    user = get_active_user_profile_by_id_in_realm(user_id, payment_intent.customer.realm)

    description = ""
    charge: stripe.Charge
    for charge in stripe_payment_intent.charges:  # type: ignore[attr-defined] # https://stripe.com/docs/upgrades#2022-11-15
        assert charge.payment_method_details is not None
        assert charge.payment_method_details.card is not None
        description = f"Payment (Card ending in {charge.payment_method_details.card.last4})"
        break

    stripe.InvoiceItem.create(
        amount=stripe_payment_intent.amount * -1,
        currency="usd",
        customer=stripe_payment_intent.customer,
        description=description,
        discountable=False,
    )
    try:
        ensure_customer_does_not_have_active_plan(payment_intent.customer)
    except UpgradeWithExistingPlanError as e:
        stripe_invoice = stripe.Invoice.create(
            auto_advance=True,
            collection_method="charge_automatically",
            customer=stripe_payment_intent.customer,
            days_until_due=None,
            statement_descriptor="Cloud Standard Credit",
        )
        stripe.Invoice.finalize_invoice(stripe_invoice)
        raise e

    billing_session = RealmBillingSession(user)
    billing_session.process_initial_upgrade(
        CustomerPlan.STANDARD,
        int(metadata["licenses"]),
        metadata["license_management"] == "automatic",
        int(metadata["billing_schedule"]),
        True,
        False,
    )


@error_handler
def handle_payment_intent_payment_failed_event(
    stripe_payment_intent: stripe.PaymentIntent, payment_intent: PaymentIntent
) -> None:
    assert stripe_payment_intent.last_payment_error is not None
    payment_intent.status = PaymentIntent.get_status_integer_from_status_text(
        stripe_payment_intent.status
    )
    assert payment_intent.customer.realm is not None
    billing_logger.info(
        "Stripe payment intent failed: %s %s %s %s",
        payment_intent.customer.realm.string_id,
        stripe_payment_intent.last_payment_error.get("type"),
        stripe_payment_intent.last_payment_error.get("code"),
        stripe_payment_intent.last_payment_error.get("param"),
    )
    payment_intent.last_payment_error = {
        "description": stripe_payment_intent.last_payment_error.get("type"),
    }
    payment_intent.last_payment_error["message"] = stripe_payment_intent.last_payment_error.get(
        "message"
    )
    payment_intent.save(update_fields=["status", "last_payment_error"])
