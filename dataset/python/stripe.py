import logging
import math
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generator, Optional, Tuple, TypedDict, TypeVar, Union

import stripe
from django.conf import settings
from django.core import signing
from django.core.signing import Signer
from django.db import transaction
from django.urls import reverse
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy
from django.utils.translation import override as override_language
from typing_extensions import ParamSpec, override

from corporate.models import (
    Customer,
    CustomerPlan,
    LicenseLedger,
    PaymentIntent,
    Session,
    get_current_plan_by_customer,
    get_current_plan_by_realm,
    get_customer_by_realm,
)
from zerver.lib.exceptions import JsonableError
from zerver.lib.logging_util import log_to_file
from zerver.lib.send_email import FromAddress, send_email_to_billing_admins_and_realm_owners
from zerver.lib.timestamp import datetime_to_timestamp, timestamp_to_datetime
from zerver.lib.utils import assert_is_not_none
from zerver.models import Realm, RealmAuditLog, UserProfile, get_system_bot
from zilencer.models import RemoteZulipServer, RemoteZulipServerAuditLog
from zproject.config import get_secret

stripe.api_key = get_secret("stripe_secret_key")

BILLING_LOG_PATH = os.path.join(
    "/var/log/zulip" if not settings.DEVELOPMENT else settings.DEVELOPMENT_LOG_DIRECTORY,
    "billing.log",
)
billing_logger = logging.getLogger("corporate.stripe")
log_to_file(billing_logger, BILLING_LOG_PATH)
log_to_file(logging.getLogger("stripe"), BILLING_LOG_PATH)

ParamT = ParamSpec("ParamT")
ReturnT = TypeVar("ReturnT")

MIN_INVOICED_LICENSES = 30
MAX_INVOICED_LICENSES = 1000
DEFAULT_INVOICE_DAYS_UNTIL_DUE = 30

VALID_BILLING_MODALITY_VALUES = ["send_invoice", "charge_automatically"]
VALID_BILLING_SCHEDULE_VALUES = ["annual", "monthly"]
VALID_LICENSE_MANAGEMENT_VALUES = ["automatic", "manual"]

# The version of Stripe API the billing system supports.
STRIPE_API_VERSION = "2020-08-27"

stripe.api_version = STRIPE_API_VERSION


# This function imitates the behavior of the format_money in billing/helpers.ts
def format_money(cents: float) -> str:
    # allow for small floating point errors
    cents = math.ceil(cents - 0.001)
    if cents % 100 == 0:
        precision = 0
    else:
        precision = 2

    dollars = cents / 100
    # Format the number as a string with the correct number of decimal places
    return f"{dollars:.{precision}f}"


def get_latest_seat_count(realm: Realm) -> int:
    return get_seat_count(realm, extra_non_guests_count=0, extra_guests_count=0)


def get_seat_count(
    realm: Realm, extra_non_guests_count: int = 0, extra_guests_count: int = 0
) -> int:
    non_guests = (
        UserProfile.objects.filter(realm=realm, is_active=True, is_bot=False)
        .exclude(role=UserProfile.ROLE_GUEST)
        .count()
    ) + extra_non_guests_count

    # This guest count calculation should match the similar query in render_stats().
    guests = (
        UserProfile.objects.filter(
            realm=realm, is_active=True, is_bot=False, role=UserProfile.ROLE_GUEST
        ).count()
        + extra_guests_count
    )

    # This formula achieves the pricing of the first 5*N guests
    # being free of charge (where N is the number of non-guests in the organization)
    # and each consecutive one being worth 1/5 the non-guest price.
    return max(non_guests, math.ceil(guests / 5))


def sign_string(string: str) -> Tuple[str, str]:
    salt = secrets.token_hex(32)
    signer = Signer(salt=salt)
    return signer.sign(string), salt


def unsign_string(signed_string: str, salt: str) -> str:
    signer = Signer(salt=salt)
    return signer.unsign(signed_string)


def unsign_seat_count(signed_seat_count: str, salt: str) -> int:
    try:
        return int(unsign_string(signed_seat_count, salt))
    except signing.BadSignature:
        raise BillingError("tampered seat count")


def validate_licenses(
    charge_automatically: bool,
    licenses: Optional[int],
    seat_count: int,
    exempt_from_license_number_check: bool,
) -> None:
    min_licenses = seat_count
    max_licenses = None
    if not charge_automatically:
        min_licenses = max(seat_count, MIN_INVOICED_LICENSES)
        max_licenses = MAX_INVOICED_LICENSES

    if licenses is None or (not exempt_from_license_number_check and licenses < min_licenses):
        raise BillingError(
            "not enough licenses",
            _(
                "You must purchase licenses for all active users in your organization (minimum {min_licenses})."
            ).format(min_licenses=min_licenses),
        )

    if max_licenses is not None and licenses > max_licenses:
        message = _(
            "Invoices with more than {max_licenses} licenses can't be processed from this page. To"
            " complete the upgrade, please contact {email}."
        ).format(max_licenses=max_licenses, email=settings.ZULIP_ADMINISTRATOR)
        raise BillingError("too many licenses", message)


def check_upgrade_parameters(
    billing_modality: str,
    schedule: str,
    license_management: Optional[str],
    licenses: Optional[int],
    seat_count: int,
    exempt_from_license_number_check: bool,
) -> None:
    if billing_modality not in VALID_BILLING_MODALITY_VALUES:  # nocoverage
        raise BillingError("unknown billing_modality", "")
    if schedule not in VALID_BILLING_SCHEDULE_VALUES:  # nocoverage
        raise BillingError("unknown schedule")
    if license_management not in VALID_LICENSE_MANAGEMENT_VALUES:  # nocoverage
        raise BillingError("unknown license_management")
    validate_licenses(
        billing_modality == "charge_automatically",
        licenses,
        seat_count,
        exempt_from_license_number_check,
    )


# Be extremely careful changing this function. Historical billing periods
# are not stored anywhere, and are just computed on the fly using this
# function. Any change you make here should return the same value (or be
# within a few seconds) for basically any value from when the billing system
# went online to within a year from now.
def add_months(dt: datetime, months: int) -> datetime:
    assert months >= 0
    # It's fine that the max day in Feb is 28 for leap years.
    MAX_DAY_FOR_MONTH = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
    year = dt.year
    month = dt.month + months
    while month > 12:
        year += 1
        month -= 12
    day = min(dt.day, MAX_DAY_FOR_MONTH[month])
    # datetimes don't support leap seconds, so don't need to worry about those
    return dt.replace(year=year, month=month, day=day)


def next_month(billing_cycle_anchor: datetime, dt: datetime) -> datetime:
    estimated_months = round((dt - billing_cycle_anchor).days * 12.0 / 365)
    for months in range(max(estimated_months - 1, 0), estimated_months + 2):
        proposed_next_month = add_months(billing_cycle_anchor, months)
        if 20 < (proposed_next_month - dt).days < 40:
            return proposed_next_month
    raise AssertionError(
        "Something wrong in next_month calculation with "
        f"billing_cycle_anchor: {billing_cycle_anchor}, dt: {dt}"
    )


def start_of_next_billing_cycle(plan: CustomerPlan, event_time: datetime) -> datetime:
    months_per_period = {
        CustomerPlan.ANNUAL: 12,
        CustomerPlan.MONTHLY: 1,
    }[plan.billing_schedule]
    periods = 1
    dt = plan.billing_cycle_anchor
    while dt <= event_time:
        dt = add_months(plan.billing_cycle_anchor, months_per_period * periods)
        periods += 1
    return dt


def next_invoice_date(plan: CustomerPlan) -> Optional[datetime]:
    if plan.status == CustomerPlan.ENDED:
        return None
    assert plan.next_invoice_date is not None  # for mypy
    months_per_period = {
        CustomerPlan.ANNUAL: 12,
        CustomerPlan.MONTHLY: 1,
    }[plan.billing_schedule]
    if plan.automanage_licenses:
        months_per_period = 1
    periods = 1
    dt = plan.billing_cycle_anchor
    while dt <= plan.next_invoice_date:
        dt = add_months(plan.billing_cycle_anchor, months_per_period * periods)
        periods += 1
    return dt


def renewal_amount(plan: CustomerPlan, event_time: datetime) -> int:  # nocoverage: TODO
    if plan.fixed_price is not None:
        return plan.fixed_price
    realm = plan.customer.realm
    billing_session = RealmBillingSession(user=None, realm=realm)
    new_plan, last_ledger_entry = billing_session.make_end_of_cycle_updates_if_needed(
        plan, event_time
    )
    if last_ledger_entry is None:
        return 0
    if last_ledger_entry.licenses_at_next_renewal is None:
        return 0
    if new_plan is not None:
        plan = new_plan
    assert plan.price_per_license is not None  # for mypy
    return plan.price_per_license * last_ledger_entry.licenses_at_next_renewal


def get_idempotency_key(ledger_entry: LicenseLedger) -> Optional[str]:
    if settings.TEST_SUITE:
        return None
    return f"ledger_entry:{ledger_entry.id}"  # nocoverage


def cents_to_dollar_string(cents: int) -> str:
    return f"{cents / 100.:,.2f}"


class BillingError(JsonableError):
    data_fields = ["error_description"]
    # error messages
    CONTACT_SUPPORT = gettext_lazy("Something went wrong. Please contact {email}.")
    TRY_RELOADING = gettext_lazy("Something went wrong. Please reload the page.")

    # description is used only for tests
    def __init__(self, description: str, message: Optional[str] = None) -> None:
        self.error_description = description
        if message is None:
            message = BillingError.CONTACT_SUPPORT.format(email=settings.ZULIP_ADMINISTRATOR)
        super().__init__(message)


class LicenseLimitError(Exception):
    pass


class StripeCardError(BillingError):
    pass


class StripeConnectionError(BillingError):
    pass


class UpgradeWithExistingPlanError(BillingError):
    def __init__(self) -> None:
        super().__init__(
            "subscribing with existing subscription",
            "The organization is already subscribed to a plan. Please reload the billing page.",
        )


class InvalidBillingScheduleError(Exception):
    def __init__(self, billing_schedule: int) -> None:
        self.message = f"Unknown billing_schedule: {billing_schedule}"
        super().__init__(self.message)


class InvalidTierError(Exception):
    def __init__(self, tier: int) -> None:
        self.message = f"Unknown tier: {tier}"
        super().__init__(self.message)


def catch_stripe_errors(func: Callable[ParamT, ReturnT]) -> Callable[ParamT, ReturnT]:
    @wraps(func)
    def wrapped(*args: ParamT.args, **kwargs: ParamT.kwargs) -> ReturnT:
        try:
            return func(*args, **kwargs)
        # See https://stripe.com/docs/api/python#error_handling, though
        # https://stripe.com/docs/api/ruby#error_handling suggests there are additional fields, and
        # https://stripe.com/docs/error-codes gives a more detailed set of error codes
        except stripe.error.StripeError as e:
            assert isinstance(e.json_body, dict)
            err = e.json_body.get("error", {})
            if isinstance(e, stripe.error.CardError):
                billing_logger.info(
                    "Stripe card error: %s %s %s %s",
                    e.http_status,
                    err.get("type"),
                    err.get("code"),
                    err.get("param"),
                )
                # TODO: Look into i18n for this
                raise StripeCardError("card error", err.get("message"))
            billing_logger.error(
                "Stripe error: %s %s %s %s",
                e.http_status,
                err.get("type"),
                err.get("code"),
                err.get("param"),
            )
            if isinstance(
                e, (stripe.error.RateLimitError, stripe.error.APIConnectionError)
            ):  # nocoverage TODO
                raise StripeConnectionError(
                    "stripe connection error",
                    _("Something went wrong. Please wait a few seconds and try again."),
                )
            raise BillingError("other stripe error")

    return wrapped


@catch_stripe_errors
def stripe_get_customer(stripe_customer_id: str) -> stripe.Customer:
    return stripe.Customer.retrieve(
        stripe_customer_id, expand=["invoice_settings", "invoice_settings.default_payment_method"]
    )


@dataclass
class StripeCustomerData:
    description: str
    email: str
    metadata: Dict[str, Any]


@dataclass
class StripePaymentIntentData:
    amount: int
    description: str
    plan_name: str
    email: str


@dataclass
class UpgradeRequest:
    billing_modality: str
    schedule: str
    signed_seat_count: str
    salt: str
    onboarding: bool
    license_management: Optional[str]
    licenses: Optional[int]


class AuditLogEventType(Enum):
    STRIPE_CUSTOMER_CREATED = 1
    STRIPE_CARD_CHANGED = 2
    CUSTOMER_PLAN_CREATED = 3
    DISCOUNT_CHANGED = 4
    SPONSORSHIP_APPROVED = 5
    SPONSORSHIP_PENDING_STATUS_CHANGED = 6
    BILLING_METHOD_CHANGED = 7
    CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN = 8


class BillingSessionAuditLogEventError(Exception):
    def __init__(self, event_type: AuditLogEventType) -> None:
        self.message = f"Unknown audit log event type: {event_type}"
        super().__init__(self.message)


class BillingSession(ABC):
    @property
    @abstractmethod
    def billing_session_url(self) -> str:
        pass

    @abstractmethod
    def get_customer(self) -> Optional[Customer]:
        pass

    @abstractmethod
    def current_count_for_billed_licenses(self) -> int:
        pass

    @abstractmethod
    def get_audit_log_event(self, event_type: AuditLogEventType) -> int:
        pass

    @abstractmethod
    def write_to_audit_log(
        self,
        event_type: AuditLogEventType,
        event_time: datetime,
        *,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @abstractmethod
    def get_data_for_stripe_customer(self) -> StripeCustomerData:
        pass

    @abstractmethod
    def update_data_for_checkout_session_and_payment_intent(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_data_for_stripe_payment_intent(
        self, price_per_license: int, licenses: int
    ) -> StripePaymentIntentData:
        pass

    @abstractmethod
    def update_or_create_customer(
        self, stripe_customer_id: Optional[str] = None, *, defaults: Optional[Dict[str, Any]] = None
    ) -> Customer:
        pass

    @abstractmethod
    def do_change_plan_type(self, *, tier: Optional[int], is_sponsored: bool = False) -> None:
        pass

    @abstractmethod
    def process_downgrade(self, plan: CustomerPlan) -> None:
        pass

    @abstractmethod
    def approve_sponsorship(self) -> None:
        pass

    @catch_stripe_errors
    def create_stripe_customer(self) -> Customer:
        stripe_customer_data = self.get_data_for_stripe_customer()
        stripe_customer = stripe.Customer.create(
            description=stripe_customer_data.description,
            email=stripe_customer_data.email,
            metadata=stripe_customer_data.metadata,
        )
        event_time = timestamp_to_datetime(stripe_customer.created)
        with transaction.atomic():
            self.write_to_audit_log(AuditLogEventType.STRIPE_CUSTOMER_CREATED, event_time)
            customer = self.update_or_create_customer(stripe_customer.id)
        return customer

    @catch_stripe_errors
    def replace_payment_method(
        self, stripe_customer_id: str, payment_method: str, pay_invoices: bool = False
    ) -> None:
        stripe.Customer.modify(
            stripe_customer_id, invoice_settings={"default_payment_method": payment_method}
        )
        self.write_to_audit_log(AuditLogEventType.STRIPE_CARD_CHANGED, timezone_now())
        if pay_invoices:
            for stripe_invoice in stripe.Invoice.list(
                collection_method="charge_automatically",
                customer=stripe_customer_id,
                status="open",
            ):
                # The stripe customer with the associated ID will get either a receipt
                # or a "failed payment" email, but the in-app messaging could be clearer
                # here (e.g. it could explicitly tell the user that there were payment(s)
                # and that they succeeded or failed). Worth fixing if we notice that a
                # lot of cards end up failing at this step.
                stripe.Invoice.pay(stripe_invoice)

    # Returns Customer instead of stripe_customer so that we don't make a Stripe
    # API call if there's nothing to update
    @catch_stripe_errors
    def update_or_create_stripe_customer(self, payment_method: Optional[str] = None) -> Customer:
        customer = self.get_customer()
        if customer is None or customer.stripe_customer_id is None:
            # A stripe.PaymentMethod should be attached to a stripe.Customer via
            # a stripe.SetupIntent or stripe.PaymentIntent. Here we just want to
            # create a new stripe.Customer.
            assert payment_method is None
            # We could do a better job of handling race conditions here, but if two
            # people try to upgrade at exactly the same time, the main bad thing that
            # will happen is that we will create an extra stripe.Customer that we can
            # delete or ignore.
            return self.create_stripe_customer()
        if payment_method is not None:
            self.replace_payment_method(customer.stripe_customer_id, payment_method, True)
        return customer

    def create_stripe_payment_intent(
        self, price_per_license: int, licenses: int, metadata: Dict[str, Any]
    ) -> PaymentIntent:
        customer = self.get_customer()
        assert customer is not None and customer.stripe_customer_id is not None
        payment_intent_data = self.get_data_for_stripe_payment_intent(price_per_license, licenses)
        stripe_payment_intent = stripe.PaymentIntent.create(
            amount=payment_intent_data.amount,
            currency="usd",
            customer=customer.stripe_customer_id,
            description=payment_intent_data.description,
            receipt_email=payment_intent_data.email,
            confirm=False,
            statement_descriptor=payment_intent_data.plan_name,
            metadata=metadata,
        )
        payment_intent = PaymentIntent.objects.create(
            customer=customer,
            stripe_payment_intent_id=stripe_payment_intent.id,
            status=PaymentIntent.get_status_integer_from_status_text(stripe_payment_intent.status),
        )
        return payment_intent

    def create_stripe_checkout_session(
        self,
        metadata: Dict[str, Any],
        session_type: int,
        payment_intent: Optional[PaymentIntent] = None,
    ) -> stripe.checkout.Session:
        customer = self.get_customer()
        assert customer is not None and customer.stripe_customer_id is not None
        stripe_session = stripe.checkout.Session.create(
            cancel_url=f"{self.billing_session_url}/billing/",
            customer=customer.stripe_customer_id,
            metadata=metadata,
            mode="setup",
            payment_method_types=["card"],
            success_url=f"{self.billing_session_url}/billing/event_status?stripe_session_id={{CHECKOUT_SESSION_ID}}",
        )
        session = Session.objects.create(
            stripe_session_id=stripe_session.id,
            customer=customer,
            type=session_type,
        )
        if payment_intent is not None:
            session.payment_intent = payment_intent
            session.save(update_fields=["payment_intent"])
            session.save()
        return stripe_session

    def attach_discount_to_customer(self, discount: Decimal) -> None:
        customer = self.get_customer()
        old_discount: Optional[Decimal] = None
        if customer is not None:
            old_discount = customer.default_discount
            customer.default_discount = discount
            customer.save(update_fields=["default_discount"])
        else:
            customer = self.update_or_create_customer(defaults={"default_discount": discount})
        plan = get_current_plan_by_customer(customer)
        if plan is not None:
            plan.price_per_license = get_price_per_license(
                plan.tier, plan.billing_schedule, discount
            )
            plan.discount = discount
            plan.save(update_fields=["price_per_license", "discount"])
        self.write_to_audit_log(
            event_type=AuditLogEventType.DISCOUNT_CHANGED,
            event_time=timezone_now(),
            extra_data={"old_discount": old_discount, "new_discount": discount},
        )

    def update_customer_sponsorship_status(self, sponsorship_pending: bool) -> None:
        customer = self.get_customer()
        if customer is None:
            customer = self.update_or_create_customer()
        customer.sponsorship_pending = sponsorship_pending
        customer.save(update_fields=["sponsorship_pending"])
        self.write_to_audit_log(
            event_type=AuditLogEventType.SPONSORSHIP_PENDING_STATUS_CHANGED,
            event_time=timezone_now(),
            extra_data={"sponsorship_pending": sponsorship_pending},
        )

    def update_billing_method_of_current_plan(self, charge_automatically: bool) -> None:
        customer = self.get_customer()
        if customer is not None:
            plan = get_current_plan_by_customer(customer)
            if plan is not None:
                plan.charge_automatically = charge_automatically
                plan.save(update_fields=["charge_automatically"])
                self.write_to_audit_log(
                    event_type=AuditLogEventType.BILLING_METHOD_CHANGED,
                    event_time=timezone_now(),
                    extra_data={"charge_automatically": charge_automatically},
                )

    def setup_upgrade_checkout_session_and_payment_intent(
        self,
        plan_tier: int,
        seat_count: int,
        licenses: int,
        license_management: str,
        billing_schedule: int,
        billing_modality: str,
        onboarding: bool,
    ) -> stripe.checkout.Session:
        customer = self.update_or_create_stripe_customer()
        assert customer is not None  # for mypy
        free_trial = is_free_trial_offer_enabled()
        price_per_license = get_price_per_license(
            plan_tier, billing_schedule, customer.default_discount
        )
        general_metadata = {
            "billing_modality": billing_modality,
            "billing_schedule": billing_schedule,
            "licenses": licenses,
            "license_management": license_management,
            "price_per_license": price_per_license,
            "seat_count": seat_count,
            "type": "upgrade",
        }
        updated_metadata = self.update_data_for_checkout_session_and_payment_intent(
            general_metadata
        )
        if free_trial:
            if onboarding:
                session_type = Session.FREE_TRIAL_UPGRADE_FROM_ONBOARDING_PAGE
            else:
                session_type = Session.FREE_TRIAL_UPGRADE_FROM_BILLING_PAGE
            payment_intent = None
        else:
            session_type = Session.UPGRADE_FROM_BILLING_PAGE
            payment_intent = self.create_stripe_payment_intent(
                price_per_license, licenses, updated_metadata
            )
        stripe_session = self.create_stripe_checkout_session(
            updated_metadata, session_type, payment_intent
        )
        return stripe_session

    # Only used for cloud signups
    @catch_stripe_errors
    def process_initial_upgrade(
        self,
        plan_tier: int,
        licenses: int,
        automanage_licenses: bool,
        billing_schedule: int,
        charge_automatically: bool,
        free_trial: bool,
    ) -> None:
        customer = self.update_or_create_stripe_customer()
        assert customer.stripe_customer_id is not None  # for mypy
        ensure_customer_does_not_have_active_plan(customer)
        (
            billing_cycle_anchor,
            next_invoice_date,
            period_end,
            price_per_license,
        ) = compute_plan_parameters(
            plan_tier,
            automanage_licenses,
            billing_schedule,
            customer.default_discount,
            free_trial,
        )

        # TODO: The correctness of this relies on user creation, deactivation, etc being
        # in a transaction.atomic() with the relevant RealmAuditLog entries
        with transaction.atomic():
            if customer.exempt_from_license_number_check:
                billed_licenses = licenses
            else:
                # billed_licenses can be greater than licenses if users are added between the start of
                # this function (process_initial_upgrade) and now
                current_licenses_count = self.current_count_for_billed_licenses()
                billed_licenses = max(current_licenses_count, licenses)
            plan_params = {
                "automanage_licenses": automanage_licenses,
                "charge_automatically": charge_automatically,
                "price_per_license": price_per_license,
                "discount": customer.default_discount,
                "billing_cycle_anchor": billing_cycle_anchor,
                "billing_schedule": billing_schedule,
                "tier": plan_tier,
            }
            if free_trial:
                plan_params["status"] = CustomerPlan.FREE_TRIAL
            plan = CustomerPlan.objects.create(
                customer=customer, next_invoice_date=next_invoice_date, **plan_params
            )
            ledger_entry = LicenseLedger.objects.create(
                plan=plan,
                is_renewal=True,
                event_time=billing_cycle_anchor,
                licenses=billed_licenses,
                licenses_at_next_renewal=billed_licenses,
            )
            plan.invoiced_through = ledger_entry
            plan.save(update_fields=["invoiced_through"])
            self.write_to_audit_log(
                event_type=AuditLogEventType.CUSTOMER_PLAN_CREATED,
                event_time=billing_cycle_anchor,
                extra_data=plan_params,
            )

        if not free_trial:
            stripe.InvoiceItem.create(
                currency="usd",
                customer=customer.stripe_customer_id,
                description=plan.name,
                discountable=False,
                period={
                    "start": datetime_to_timestamp(billing_cycle_anchor),
                    "end": datetime_to_timestamp(period_end),
                },
                quantity=billed_licenses,
                unit_amount=price_per_license,
            )

            if charge_automatically:
                collection_method = "charge_automatically"
                days_until_due = None
            else:
                collection_method = "send_invoice"
                days_until_due = DEFAULT_INVOICE_DAYS_UNTIL_DUE

            stripe_invoice = stripe.Invoice.create(
                auto_advance=True,
                collection_method=collection_method,
                customer=customer.stripe_customer_id,
                days_until_due=days_until_due,
                statement_descriptor=plan.name,
            )
            stripe.Invoice.finalize_invoice(stripe_invoice)

        self.do_change_plan_type(tier=plan_tier)

    def do_upgrade(self, upgrade_request: UpgradeRequest) -> Dict[str, Any]:
        customer = self.get_customer()
        if customer is not None:
            ensure_customer_does_not_have_active_plan(customer)
        billing_modality = upgrade_request.billing_modality
        schedule = upgrade_request.schedule
        license_management = upgrade_request.license_management
        licenses = upgrade_request.licenses

        seat_count = unsign_seat_count(upgrade_request.signed_seat_count, upgrade_request.salt)
        if billing_modality == "charge_automatically" and license_management == "automatic":
            licenses = seat_count
        if billing_modality == "send_invoice":
            schedule = "annual"
            license_management = "manual"

        exempt_from_license_number_check = (
            customer is not None and customer.exempt_from_license_number_check
        )
        check_upgrade_parameters(
            billing_modality,
            schedule,
            license_management,
            licenses,
            seat_count,
            exempt_from_license_number_check,
        )
        assert licenses is not None and license_management is not None
        automanage_licenses = license_management == "automatic"
        charge_automatically = billing_modality == "charge_automatically"

        billing_schedule = {"annual": CustomerPlan.ANNUAL, "monthly": CustomerPlan.MONTHLY}[
            schedule
        ]
        data: Dict[str, Any] = {}
        if charge_automatically:
            stripe_checkout_session = self.setup_upgrade_checkout_session_and_payment_intent(
                CustomerPlan.STANDARD,
                seat_count,
                licenses,
                license_management,
                billing_schedule,
                billing_modality,
                upgrade_request.onboarding,
            )
            data = {
                "stripe_session_url": stripe_checkout_session.url,
                "stripe_session_id": stripe_checkout_session.id,
            }
        else:
            self.process_initial_upgrade(
                CustomerPlan.STANDARD,
                licenses,
                automanage_licenses,
                billing_schedule,
                False,
                is_free_trial_offer_enabled(),
            )
        return data

    # event_time should roughly be timezone_now(). Not designed to handle
    # event_times in the past or future
    @transaction.atomic
    def make_end_of_cycle_updates_if_needed(
        self, plan: CustomerPlan, event_time: datetime
    ) -> Tuple[Optional[CustomerPlan], Optional[LicenseLedger]]:
        last_ledger_entry = LicenseLedger.objects.filter(plan=plan).order_by("-id").first()
        last_ledger_renewal = (
            LicenseLedger.objects.filter(plan=plan, is_renewal=True).order_by("-id").first()
        )
        assert last_ledger_renewal is not None
        last_renewal = last_ledger_renewal.event_time

        if plan.is_free_trial() or plan.status == CustomerPlan.SWITCH_NOW_FROM_STANDARD_TO_PLUS:
            assert plan.next_invoice_date is not None
            next_billing_cycle = plan.next_invoice_date
        else:
            next_billing_cycle = start_of_next_billing_cycle(plan, last_renewal)
        if next_billing_cycle <= event_time and last_ledger_entry is not None:
            licenses_at_next_renewal = last_ledger_entry.licenses_at_next_renewal
            assert licenses_at_next_renewal is not None
            if plan.status == CustomerPlan.ACTIVE:
                return None, LicenseLedger.objects.create(
                    plan=plan,
                    is_renewal=True,
                    event_time=next_billing_cycle,
                    licenses=licenses_at_next_renewal,
                    licenses_at_next_renewal=licenses_at_next_renewal,
                )
            if plan.is_free_trial():
                plan.invoiced_through = last_ledger_entry
                plan.billing_cycle_anchor = next_billing_cycle.replace(microsecond=0)
                plan.status = CustomerPlan.ACTIVE
                plan.save(update_fields=["invoiced_through", "billing_cycle_anchor", "status"])
                return None, LicenseLedger.objects.create(
                    plan=plan,
                    is_renewal=True,
                    event_time=next_billing_cycle,
                    licenses=licenses_at_next_renewal,
                    licenses_at_next_renewal=licenses_at_next_renewal,
                )

            if plan.status == CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE:
                if plan.fixed_price is not None:  # nocoverage
                    raise NotImplementedError("Can't switch fixed priced monthly plan to annual.")

                plan.status = CustomerPlan.ENDED
                plan.save(update_fields=["status"])

                discount = plan.customer.default_discount or plan.discount
                _, _, _, price_per_license = compute_plan_parameters(
                    tier=plan.tier,
                    automanage_licenses=plan.automanage_licenses,
                    billing_schedule=CustomerPlan.ANNUAL,
                    discount=plan.discount,
                )

                new_plan = CustomerPlan.objects.create(
                    customer=plan.customer,
                    billing_schedule=CustomerPlan.ANNUAL,
                    automanage_licenses=plan.automanage_licenses,
                    charge_automatically=plan.charge_automatically,
                    price_per_license=price_per_license,
                    discount=discount,
                    billing_cycle_anchor=next_billing_cycle,
                    tier=plan.tier,
                    status=CustomerPlan.ACTIVE,
                    next_invoice_date=next_billing_cycle,
                    invoiced_through=None,
                    invoicing_status=CustomerPlan.INITIAL_INVOICE_TO_BE_SENT,
                )

                new_plan_ledger_entry = LicenseLedger.objects.create(
                    plan=new_plan,
                    is_renewal=True,
                    event_time=next_billing_cycle,
                    licenses=licenses_at_next_renewal,
                    licenses_at_next_renewal=licenses_at_next_renewal,
                )

                self.write_to_audit_log(
                    event_type=AuditLogEventType.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN,
                    event_time=event_time,
                    extra_data={
                        "monthly_plan_id": plan.id,
                        "annual_plan_id": new_plan.id,
                    },
                )
                return new_plan, new_plan_ledger_entry

            if plan.status == CustomerPlan.SWITCH_NOW_FROM_STANDARD_TO_PLUS:
                standard_plan = plan
                standard_plan.end_date = next_billing_cycle
                standard_plan.status = CustomerPlan.ENDED
                standard_plan.save(update_fields=["status", "end_date"])

                (_, _, _, plus_plan_price_per_license) = compute_plan_parameters(
                    CustomerPlan.PLUS,
                    standard_plan.automanage_licenses,
                    standard_plan.billing_schedule,
                    standard_plan.customer.default_discount,
                )
                plus_plan_billing_cycle_anchor = standard_plan.end_date.replace(microsecond=0)

                plus_plan = CustomerPlan.objects.create(
                    customer=standard_plan.customer,
                    status=CustomerPlan.ACTIVE,
                    automanage_licenses=standard_plan.automanage_licenses,
                    charge_automatically=standard_plan.charge_automatically,
                    price_per_license=plus_plan_price_per_license,
                    discount=standard_plan.customer.default_discount,
                    billing_schedule=standard_plan.billing_schedule,
                    tier=CustomerPlan.PLUS,
                    billing_cycle_anchor=plus_plan_billing_cycle_anchor,
                    invoicing_status=CustomerPlan.INITIAL_INVOICE_TO_BE_SENT,
                    next_invoice_date=plus_plan_billing_cycle_anchor,
                )

                standard_plan_last_ledger = (
                    LicenseLedger.objects.filter(plan=standard_plan).order_by("id").last()
                )
                assert standard_plan_last_ledger is not None
                licenses_for_plus_plan = standard_plan_last_ledger.licenses_at_next_renewal
                assert licenses_for_plus_plan is not None
                plus_plan_ledger_entry = LicenseLedger.objects.create(
                    plan=plus_plan,
                    is_renewal=True,
                    event_time=plus_plan_billing_cycle_anchor,
                    licenses=licenses_for_plus_plan,
                    licenses_at_next_renewal=licenses_for_plus_plan,
                )
                return plus_plan, plus_plan_ledger_entry

            if plan.status == CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE:
                self.process_downgrade(plan)
            return None, None
        return None, last_ledger_entry


class RealmBillingSession(BillingSession):
    def __init__(self, user: Optional[UserProfile] = None, realm: Optional[Realm] = None) -> None:
        self.user = user
        assert user is not None or realm is not None
        if user is not None and realm is not None:
            assert user.is_staff
            self.realm = realm
            self.support_session = True
        elif user is not None:
            self.realm = user.realm
            self.support_session = False
        else:
            assert realm is not None  # for mypy
            self.realm = realm
            self.support_session = False

    @override
    @property
    def billing_session_url(self) -> str:
        return self.realm.uri

    @override
    def get_customer(self) -> Optional[Customer]:
        return get_customer_by_realm(self.realm)

    @override
    def current_count_for_billed_licenses(self) -> int:
        return get_latest_seat_count(self.realm)

    @override
    def get_audit_log_event(self, event_type: AuditLogEventType) -> int:
        if event_type is AuditLogEventType.STRIPE_CUSTOMER_CREATED:
            return RealmAuditLog.STRIPE_CUSTOMER_CREATED
        elif event_type is AuditLogEventType.STRIPE_CARD_CHANGED:
            return RealmAuditLog.STRIPE_CARD_CHANGED
        elif event_type is AuditLogEventType.CUSTOMER_PLAN_CREATED:
            return RealmAuditLog.CUSTOMER_PLAN_CREATED
        elif event_type is AuditLogEventType.DISCOUNT_CHANGED:
            return RealmAuditLog.REALM_DISCOUNT_CHANGED
        elif event_type is AuditLogEventType.SPONSORSHIP_APPROVED:
            return RealmAuditLog.REALM_SPONSORSHIP_APPROVED
        elif event_type is AuditLogEventType.SPONSORSHIP_PENDING_STATUS_CHANGED:
            return RealmAuditLog.REALM_SPONSORSHIP_PENDING_STATUS_CHANGED
        elif event_type is AuditLogEventType.BILLING_METHOD_CHANGED:
            return RealmAuditLog.REALM_BILLING_METHOD_CHANGED
        elif event_type is AuditLogEventType.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN:
            return RealmAuditLog.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN
        else:
            raise BillingSessionAuditLogEventError(event_type)

    @override
    def write_to_audit_log(
        self,
        event_type: AuditLogEventType,
        event_time: datetime,
        *,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        audit_log_event = self.get_audit_log_event(event_type)
        audit_log_data = {
            "realm": self.realm,
            "event_type": audit_log_event,
            "event_time": event_time,
        }

        if extra_data:
            audit_log_data["extra_data"] = extra_data

        if self.user is not None:
            audit_log_data["acting_user"] = self.user

        RealmAuditLog.objects.create(**audit_log_data)

    @override
    def get_data_for_stripe_customer(self) -> StripeCustomerData:
        # Support requests do not set any stripe billing information.
        assert self.support_session is False
        assert self.user is not None
        metadata: Dict[str, Any] = {}
        metadata["realm_id"] = self.realm.id
        metadata["realm_str"] = self.realm.string_id
        realm_stripe_customer_data = StripeCustomerData(
            description=f"{self.realm.string_id} ({self.realm.name})",
            email=self.user.delivery_email,
            metadata=metadata,
        )
        return realm_stripe_customer_data

    @override
    def update_data_for_checkout_session_and_payment_intent(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        assert self.user is not None
        updated_metadata = dict(
            user_email=self.user.delivery_email,
            realm_id=self.realm.id,
            realm_str=self.realm.string_id,
            user_id=self.user.id,
            **metadata,
        )
        return updated_metadata

    @override
    def get_data_for_stripe_payment_intent(
        self, price_per_license: int, licenses: int
    ) -> StripePaymentIntentData:
        # Support requests do not set any stripe billing information.
        assert self.support_session is False
        assert self.user is not None
        amount = price_per_license * licenses
        description = f"Upgrade to Zulip Cloud Standard, ${price_per_license/100} x {licenses}"
        plan_name = "Zulip Cloud Standard"
        return StripePaymentIntentData(
            amount=amount,
            description=description,
            plan_name=plan_name,
            email=self.user.delivery_email,
        )

    @override
    def update_or_create_customer(
        self, stripe_customer_id: Optional[str] = None, *, defaults: Optional[Dict[str, Any]] = None
    ) -> Customer:
        if stripe_customer_id is not None:
            # Support requests do not set any stripe billing information.
            assert self.support_session is False
            customer, created = Customer.objects.update_or_create(
                realm=self.realm, defaults={"stripe_customer_id": stripe_customer_id}
            )
            from zerver.actions.users import do_change_is_billing_admin

            assert self.user is not None
            do_change_is_billing_admin(self.user, True)
            return customer
        else:
            customer, created = Customer.objects.update_or_create(
                realm=self.realm, defaults=defaults
            )
            return customer

    @override
    def do_change_plan_type(self, *, tier: Optional[int], is_sponsored: bool = False) -> None:
        from zerver.actions.realm_settings import do_change_realm_plan_type

        # This function needs to translate between the different
        # formats of CustomerPlan.tier and Realm.plan_type.
        if is_sponsored:
            plan_type = Realm.PLAN_TYPE_STANDARD_FREE
        elif tier == CustomerPlan.STANDARD:
            plan_type = Realm.PLAN_TYPE_STANDARD
        elif tier == CustomerPlan.PLUS:  # nocoverage # Plus plan doesn't use this code path yet.
            plan_type = Realm.PLAN_TYPE_PLUS
        else:
            raise AssertionError("Unexpected tier")
        do_change_realm_plan_type(self.realm, plan_type, acting_user=self.user)

    @override
    def process_downgrade(self, plan: CustomerPlan) -> None:
        from zerver.actions.realm_settings import do_change_realm_plan_type

        assert plan.customer.realm is not None
        do_change_realm_plan_type(plan.customer.realm, Realm.PLAN_TYPE_LIMITED, acting_user=None)
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=["status"])

    @override
    def approve_sponsorship(self) -> None:
        # Sponsorship approval is only a support admin action.
        assert self.support_session

        from zerver.actions.message_send import internal_send_private_message

        self.do_change_plan_type(tier=None, is_sponsored=True)
        customer = self.get_customer()
        if customer is not None and customer.sponsorship_pending:
            customer.sponsorship_pending = False
            customer.save(update_fields=["sponsorship_pending"])
            self.write_to_audit_log(
                event_type=AuditLogEventType.SPONSORSHIP_APPROVED, event_time=timezone_now()
            )
        notification_bot = get_system_bot(settings.NOTIFICATION_BOT, self.realm.id)
        for user in self.realm.get_human_billing_admin_and_realm_owner_users():
            with override_language(user.default_language):
                # Using variable to make life easier for translators if these details change.
                message = _(
                    "Your organization's request for sponsored hosting has been approved! "
                    "You have been upgraded to {plan_name}, free of charge. {emoji}\n\n"
                    "If you could {begin_link}list Zulip as a sponsor on your website{end_link}, "
                    "we would really appreciate it!"
                ).format(
                    plan_name="Zulip Cloud Standard",
                    emoji=":tada:",
                    begin_link="[",
                    end_link="](/help/linking-to-zulip-website)",
                )
                internal_send_private_message(notification_bot, user, message)


def stripe_customer_has_credit_card_as_default_payment_method(
    stripe_customer: stripe.Customer,
) -> bool:
    assert stripe_customer.invoice_settings is not None
    if not stripe_customer.invoice_settings.default_payment_method:
        return False
    assert isinstance(stripe_customer.invoice_settings.default_payment_method, stripe.PaymentMethod)
    return stripe_customer.invoice_settings.default_payment_method.type == "card"


def customer_has_credit_card_as_default_payment_method(customer: Customer) -> bool:
    if not customer.stripe_customer_id:
        return False
    stripe_customer = stripe_get_customer(customer.stripe_customer_id)
    return stripe_customer_has_credit_card_as_default_payment_method(stripe_customer)


def calculate_discounted_price_per_license(
    original_price_per_license: int, discount: Decimal
) -> int:
    # There are no fractional cents in Stripe, so round down to nearest integer.
    return int(float(original_price_per_license * (1 - discount / 100)) + 0.00001)


def get_price_per_license(
    tier: int, billing_schedule: int, discount: Optional[Decimal] = None
) -> int:
    price_per_license: Optional[int] = None

    if tier == CustomerPlan.STANDARD:
        if billing_schedule == CustomerPlan.ANNUAL:
            price_per_license = 8000
        elif billing_schedule == CustomerPlan.MONTHLY:
            price_per_license = 800
        else:  # nocoverage
            raise InvalidBillingScheduleError(billing_schedule)
    elif tier == CustomerPlan.PLUS:
        if billing_schedule == CustomerPlan.ANNUAL:
            price_per_license = 16000
        elif billing_schedule == CustomerPlan.MONTHLY:
            price_per_license = 1600
        else:  # nocoverage
            raise InvalidBillingScheduleError(billing_schedule)
    else:
        raise InvalidTierError(tier)

    if discount is not None:
        price_per_license = calculate_discounted_price_per_license(price_per_license, discount)
    return price_per_license


def compute_plan_parameters(
    tier: int,
    automanage_licenses: bool,
    billing_schedule: int,
    discount: Optional[Decimal],
    free_trial: bool = False,
) -> Tuple[datetime, datetime, datetime, int]:
    # Everything in Stripe is stored as timestamps with 1 second resolution,
    # so standardize on 1 second resolution.
    # TODO talk about leap seconds?
    billing_cycle_anchor = timezone_now().replace(microsecond=0)
    if billing_schedule == CustomerPlan.ANNUAL:
        period_end = add_months(billing_cycle_anchor, 12)
    elif billing_schedule == CustomerPlan.MONTHLY:
        period_end = add_months(billing_cycle_anchor, 1)
    else:  # nocoverage
        raise InvalidBillingScheduleError(billing_schedule)

    price_per_license = get_price_per_license(tier, billing_schedule, discount)

    next_invoice_date = period_end
    if automanage_licenses:
        next_invoice_date = add_months(billing_cycle_anchor, 1)
    if free_trial:
        period_end = billing_cycle_anchor + timedelta(
            days=assert_is_not_none(settings.FREE_TRIAL_DAYS)
        )
        next_invoice_date = period_end
    return billing_cycle_anchor, next_invoice_date, period_end, price_per_license


def is_free_trial_offer_enabled() -> bool:
    return settings.FREE_TRIAL_DAYS not in (None, 0)


def ensure_customer_does_not_have_active_plan(customer: Customer) -> None:
    if get_current_plan_by_customer(customer) is not None:
        # Unlikely race condition from two people upgrading (clicking "Make payment")
        # at exactly the same time. Doesn't fully resolve the race condition, but having
        # a check here reduces the likelihood.
        billing_logger.warning(
            "Upgrade of %s failed because of existing active plan.",
            str(customer),
        )
        raise UpgradeWithExistingPlanError


@transaction.atomic
def do_change_remote_server_plan_type(remote_server: RemoteZulipServer, plan_type: int) -> None:
    old_value = remote_server.plan_type
    remote_server.plan_type = plan_type
    remote_server.save(update_fields=["plan_type"])
    RemoteZulipServerAuditLog.objects.create(
        event_type=RealmAuditLog.REMOTE_SERVER_PLAN_TYPE_CHANGED,
        server=remote_server,
        event_time=timezone_now(),
        extra_data={"old_value": old_value, "new_value": plan_type},
    )


@transaction.atomic
def do_deactivate_remote_server(remote_server: RemoteZulipServer) -> None:
    if remote_server.deactivated:
        billing_logger.warning(
            "Cannot deactivate remote server with ID %d, server has already been deactivated.",
            remote_server.id,
        )
        return

    remote_server.deactivated = True
    remote_server.save(update_fields=["deactivated"])
    RemoteZulipServerAuditLog.objects.create(
        event_type=RealmAuditLog.REMOTE_SERVER_DEACTIVATED,
        server=remote_server,
        event_time=timezone_now(),
    )


def update_license_ledger_for_manual_plan(
    plan: CustomerPlan,
    event_time: datetime,
    licenses: Optional[int] = None,
    licenses_at_next_renewal: Optional[int] = None,
) -> None:
    if licenses is not None:
        assert plan.customer.realm is not None
        if not plan.customer.exempt_from_license_number_check:
            assert get_latest_seat_count(plan.customer.realm) <= licenses
        assert licenses > plan.licenses()
        LicenseLedger.objects.create(
            plan=plan, event_time=event_time, licenses=licenses, licenses_at_next_renewal=licenses
        )
    elif licenses_at_next_renewal is not None:
        assert plan.customer.realm is not None
        if not plan.customer.exempt_from_license_number_check:
            assert get_latest_seat_count(plan.customer.realm) <= licenses_at_next_renewal
        LicenseLedger.objects.create(
            plan=plan,
            event_time=event_time,
            licenses=plan.licenses(),
            licenses_at_next_renewal=licenses_at_next_renewal,
        )
    else:
        raise AssertionError("Pass licenses or licenses_at_next_renewal")


def update_license_ledger_for_automanaged_plan(
    realm: Realm, plan: CustomerPlan, event_time: datetime
) -> None:
    billing_session = RealmBillingSession(user=None, realm=realm)
    new_plan, last_ledger_entry = billing_session.make_end_of_cycle_updates_if_needed(
        plan, event_time
    )
    if last_ledger_entry is None:
        return
    if new_plan is not None:
        plan = new_plan
    licenses_at_next_renewal = get_latest_seat_count(realm)
    licenses = max(licenses_at_next_renewal, last_ledger_entry.licenses)

    LicenseLedger.objects.create(
        plan=plan,
        event_time=event_time,
        licenses=licenses,
        licenses_at_next_renewal=licenses_at_next_renewal,
    )


def update_license_ledger_if_needed(realm: Realm, event_time: datetime) -> None:
    plan = get_current_plan_by_realm(realm)
    if plan is None:
        return
    if not plan.automanage_licenses:
        return
    update_license_ledger_for_automanaged_plan(realm, plan, event_time)


def get_plan_renewal_or_end_date(plan: CustomerPlan, event_time: datetime) -> datetime:
    billing_period_end = start_of_next_billing_cycle(plan, event_time)

    if plan.end_date is not None and plan.end_date < billing_period_end:
        return plan.end_date
    return billing_period_end


class PriceArgs(TypedDict, total=False):
    amount: int
    unit_amount: int
    quantity: int


def invoice_plan(plan: CustomerPlan, event_time: datetime) -> None:
    if plan.invoicing_status == CustomerPlan.STARTED:
        raise NotImplementedError("Plan with invoicing_status==STARTED needs manual resolution.")
    if not plan.customer.stripe_customer_id:
        assert plan.customer.realm is not None
        raise BillingError(
            f"Realm {plan.customer.realm.string_id} has a paid plan without a Stripe customer."
        )

    realm = plan.customer.realm
    billing_session = RealmBillingSession(user=None, realm=realm)
    billing_session.make_end_of_cycle_updates_if_needed(plan, event_time)

    if plan.invoicing_status == CustomerPlan.INITIAL_INVOICE_TO_BE_SENT:
        invoiced_through_id = -1
        licenses_base = None
    else:
        assert plan.invoiced_through is not None
        licenses_base = plan.invoiced_through.licenses
        invoiced_through_id = plan.invoiced_through.id

    invoice_item_created = False
    for ledger_entry in LicenseLedger.objects.filter(
        plan=plan, id__gt=invoiced_through_id, event_time__lte=event_time
    ).order_by("id"):
        price_args: PriceArgs = {}
        if ledger_entry.is_renewal:
            if plan.fixed_price is not None:
                price_args = {"amount": plan.fixed_price}
            else:
                assert plan.price_per_license is not None  # needed for mypy
                price_args = {
                    "unit_amount": plan.price_per_license,
                    "quantity": ledger_entry.licenses,
                }
            description = f"{plan.name} - renewal"
        elif licenses_base is not None and ledger_entry.licenses != licenses_base:
            assert plan.price_per_license
            last_ledger_entry_renewal = (
                LicenseLedger.objects.filter(
                    plan=plan, is_renewal=True, event_time__lte=ledger_entry.event_time
                )
                .order_by("-id")
                .first()
            )
            assert last_ledger_entry_renewal is not None
            last_renewal = last_ledger_entry_renewal.event_time
            billing_period_end = start_of_next_billing_cycle(plan, ledger_entry.event_time)
            plan_renewal_or_end_date = get_plan_renewal_or_end_date(plan, ledger_entry.event_time)
            proration_fraction = (plan_renewal_or_end_date - ledger_entry.event_time) / (
                billing_period_end - last_renewal
            )
            price_args = {
                "unit_amount": int(plan.price_per_license * proration_fraction + 0.5),
                "quantity": ledger_entry.licenses - licenses_base,
            }
            description = "Additional license ({} - {})".format(
                ledger_entry.event_time.strftime("%b %-d, %Y"),
                plan_renewal_or_end_date.strftime("%b %-d, %Y"),
            )

        if price_args:
            plan.invoiced_through = ledger_entry
            plan.invoicing_status = CustomerPlan.STARTED
            plan.save(update_fields=["invoicing_status", "invoiced_through"])
            stripe.InvoiceItem.create(
                currency="usd",
                customer=plan.customer.stripe_customer_id,
                description=description,
                discountable=False,
                period={
                    "start": datetime_to_timestamp(ledger_entry.event_time),
                    "end": datetime_to_timestamp(
                        get_plan_renewal_or_end_date(plan, ledger_entry.event_time)
                    ),
                },
                idempotency_key=get_idempotency_key(ledger_entry),
                **price_args,
            )
            invoice_item_created = True
        plan.invoiced_through = ledger_entry
        plan.invoicing_status = CustomerPlan.DONE
        plan.save(update_fields=["invoicing_status", "invoiced_through"])
        licenses_base = ledger_entry.licenses

    if invoice_item_created:
        if plan.charge_automatically:
            collection_method = "charge_automatically"
            days_until_due = None
        else:
            collection_method = "send_invoice"
            days_until_due = DEFAULT_INVOICE_DAYS_UNTIL_DUE
        stripe_invoice = stripe.Invoice.create(
            auto_advance=True,
            collection_method=collection_method,
            customer=plan.customer.stripe_customer_id,
            days_until_due=days_until_due,
            statement_descriptor=plan.name,
        )
        stripe.Invoice.finalize_invoice(stripe_invoice)

    plan.next_invoice_date = next_invoice_date(plan)
    plan.save(update_fields=["next_invoice_date"])


def invoice_plans_as_needed(event_time: Optional[datetime] = None) -> None:
    if event_time is None:  # nocoverage
        event_time = timezone_now()
    for plan in CustomerPlan.objects.filter(next_invoice_date__lte=event_time):
        invoice_plan(plan, event_time)


def is_realm_on_free_trial(realm: Realm) -> bool:
    plan = get_current_plan_by_realm(realm)
    return plan is not None and plan.is_free_trial()


def is_sponsored_realm(realm: Realm) -> bool:
    return realm.plan_type == Realm.PLAN_TYPE_STANDARD_FREE


def do_change_plan_status(plan: CustomerPlan, status: int) -> None:
    plan.status = status
    plan.save(update_fields=["status"])
    billing_logger.info(
        "Change plan status: Customer.id: %s, CustomerPlan.id: %s, status: %s",
        plan.customer.id,
        plan.id,
        status,
    )


# During realm deactivation we instantly downgrade the plan to Limited.
# Extra users added in the final month are not charged. Also used
# for the cancellation of Free Trial.
def downgrade_now_without_creating_additional_invoices(realm: Realm) -> None:
    plan = get_current_plan_by_realm(realm)
    if plan is None:
        return

    billing_session = RealmBillingSession(user=None, realm=realm)
    billing_session.process_downgrade(plan)
    plan.invoiced_through = LicenseLedger.objects.filter(plan=plan).order_by("id").last()
    plan.next_invoice_date = next_invoice_date(plan)
    plan.save(update_fields=["invoiced_through", "next_invoice_date"])


def downgrade_at_the_end_of_billing_cycle(realm: Realm) -> None:
    plan = get_current_plan_by_realm(realm)
    assert plan is not None
    do_change_plan_status(plan, CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE)


def get_all_invoices_for_customer(customer: Customer) -> Generator[stripe.Invoice, None, None]:
    if customer.stripe_customer_id is None:
        return

    invoices = stripe.Invoice.list(customer=customer.stripe_customer_id, limit=100)
    while len(invoices):
        for invoice in invoices:
            yield invoice
            last_invoice = invoice
        invoices = stripe.Invoice.list(
            customer=customer.stripe_customer_id, starting_after=last_invoice, limit=100
        )


def void_all_open_invoices(realm: Realm) -> int:
    customer = get_customer_by_realm(realm)
    if customer is None:
        return 0
    invoices = get_all_invoices_for_customer(customer)
    voided_invoices_count = 0
    for invoice in invoices:
        if invoice.status == "open":
            assert invoice.id is not None
            stripe.Invoice.void_invoice(invoice.id)
            voided_invoices_count += 1
    return voided_invoices_count


def customer_has_last_n_invoices_open(customer: Customer, n: int) -> bool:
    if customer.stripe_customer_id is None:  # nocoverage
        return False

    open_invoice_count = 0
    for invoice in stripe.Invoice.list(customer=customer.stripe_customer_id, limit=n):
        if invoice.status == "open":
            open_invoice_count += 1
    return open_invoice_count == n


def downgrade_small_realms_behind_on_payments_as_needed() -> None:
    customers = Customer.objects.all().exclude(stripe_customer_id=None).exclude(realm=None)
    for customer in customers:
        realm = customer.realm
        assert realm is not None

        # For larger realms, we generally want to talk to the customer
        # before downgrading or cancelling invoices; so this logic only applies with 5.
        if get_latest_seat_count(realm) >= 5:
            continue

        if get_current_plan_by_customer(customer) is not None:
            # Only customers with last 2 invoices open should be downgraded.
            if not customer_has_last_n_invoices_open(customer, 2):
                continue

            # We've now decided to downgrade this customer and void all invoices, and the below will execute this.

            downgrade_now_without_creating_additional_invoices(realm)
            void_all_open_invoices(realm)
            context: Dict[str, Union[str, Realm]] = {
                "upgrade_url": f"{realm.uri}{reverse('initial_upgrade')}",
                "realm": realm,
            }
            send_email_to_billing_admins_and_realm_owners(
                "zerver/emails/realm_auto_downgraded",
                realm,
                from_name=FromAddress.security_email_from_name(language=realm.default_language),
                from_address=FromAddress.tokenized_no_reply_address(),
                language=realm.default_language,
                context=context,
            )
        else:
            if customer_has_last_n_invoices_open(customer, 1):
                void_all_open_invoices(realm)


def switch_realm_from_standard_to_plus_plan(realm: Realm) -> None:
    standard_plan = get_current_plan_by_realm(realm)

    if (
        not standard_plan
        or standard_plan.status != CustomerPlan.ACTIVE
        or standard_plan.tier != CustomerPlan.STANDARD
    ):
        raise BillingError("Organization does not have an active Standard plan")

    if not standard_plan.customer.stripe_customer_id:
        raise BillingError("Organization missing Stripe customer.")

    plan_switch_time = timezone_now()

    standard_plan.status = CustomerPlan.SWITCH_NOW_FROM_STANDARD_TO_PLUS
    standard_plan.next_invoice_date = plan_switch_time
    standard_plan.save(update_fields=["status", "next_invoice_date"])

    from zerver.actions.realm_settings import do_change_realm_plan_type

    do_change_realm_plan_type(realm, Realm.PLAN_TYPE_PLUS, acting_user=None)

    standard_plan_next_renewal_date = start_of_next_billing_cycle(standard_plan, plan_switch_time)

    standard_plan_last_renewal_ledger = (
        LicenseLedger.objects.filter(is_renewal=True, plan=standard_plan).order_by("id").last()
    )
    assert standard_plan_last_renewal_ledger is not None
    assert standard_plan.price_per_license is not None
    standard_plan_last_renewal_amount = (
        standard_plan_last_renewal_ledger.licenses * standard_plan.price_per_license
    )
    standard_plan_last_renewal_date = standard_plan_last_renewal_ledger.event_time
    unused_proration_fraction = 1 - (plan_switch_time - standard_plan_last_renewal_date) / (
        standard_plan_next_renewal_date - standard_plan_last_renewal_date
    )
    amount_to_credit_back_to_realm = math.ceil(
        standard_plan_last_renewal_amount * unused_proration_fraction
    )
    stripe.Customer.create_balance_transaction(
        standard_plan.customer.stripe_customer_id,
        amount=-1 * amount_to_credit_back_to_realm,
        currency="usd",
        description="Credit from early termination of Standard plan",
    )
    invoice_plan(standard_plan, plan_switch_time)
    plus_plan = get_current_plan_by_realm(realm)
    assert plus_plan is not None  # for mypy
    invoice_plan(plus_plan, plan_switch_time)
