from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar
import sentry_sdk
from sqlalchemy import ForeignKey, String, orm
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from warehouse import db
from warehouse.oidc.errors import InvalidPublisherError
from warehouse.oidc.interfaces import SignedClaims
if TYPE_CHECKING:
    from warehouse.accounts.models import User
    from warehouse.macaroons.models import Macaroon
    from warehouse.packaging.models import Project
C = TypeVar('C')
CheckClaimCallable = Callable[[C, C, SignedClaims], bool]

def check_claim_binary(binary_func: Callable[[C, C], bool]) -> CheckClaimCallable[C]:
    if False:
        print('Hello World!')
    '\n    Wraps a binary comparison function so that it takes three arguments instead,\n    ignoring the third.\n\n    This is used solely to make claim verification compatible with "trivial"\n    comparison checks like `str.__eq__`.\n    '

    def wrapper(ground_truth: C, signed_claim: C, all_signed_claims: SignedClaims):
        if False:
            for i in range(10):
                print('nop')
        return binary_func(ground_truth, signed_claim)
    return wrapper

def check_claim_invariant(value: C) -> CheckClaimCallable[C]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Wraps a fixed value comparison into a three-argument function.\n\n    This is used solely to make claim verification compatible with "invariant"\n    comparison checks, like "claim x is always the literal `true` value".\n    '

    def wrapper(ground_truth: C, signed_claim: C, all_signed_claims: SignedClaims):
        if False:
            print('Hello World!')
        return ground_truth == signed_claim == value
    return wrapper

class OIDCPublisherProjectAssociation(db.Model):
    __tablename__ = 'oidc_publisher_project_association'
    oidc_publisher_id = mapped_column(UUID(as_uuid=True), ForeignKey('oidc_publishers.id'), nullable=False, primary_key=True)
    project_id = mapped_column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False, primary_key=True)

class OIDCPublisherMixin:
    """
    A mixin for common functionality between all OIDC publishers, including
    "pending" publishers that don't correspond to an extant project yet.
    """
    discriminator = mapped_column(String)
    __required_verifiable_claims__: dict[str, CheckClaimCallable[Any]] = dict()
    __required_unverifiable_claims__: set[str] = set()
    __optional_verifiable_claims__: dict[str, CheckClaimCallable[Any]] = dict()
    __preverified_claims__ = {'iss', 'iat', 'nbf', 'exp', 'aud'}
    __unchecked_claims__: set[str] = set()
    __lookup_strategies__: list = []

    @classmethod
    def lookup_by_claims(cls, session, signed_claims: SignedClaims):
        if False:
            print('Hello World!')
        for lookup in cls.__lookup_strategies__:
            query = lookup(cls, signed_claims)
            if not query:
                continue
            if (publisher := query.with_session(session).one_or_none()):
                return publisher
        raise InvalidPublisherError('All lookup strategies exhausted')

    @classmethod
    def all_known_claims(cls) -> set[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns all claims "known" to this publisher.\n        '
        return cls.__required_verifiable_claims__.keys() | cls.__required_unverifiable_claims__ | cls.__optional_verifiable_claims__.keys() | cls.__preverified_claims__ | cls.__unchecked_claims__

    def verify_claims(self, signed_claims: SignedClaims):
        if False:
            print('Hello World!')
        '\n        Given a JWT that has been successfully decoded (checked for a valid\n        signature and basic claims), verify it against the more specific\n        claims of this publisher.\n        '
        if not self.__required_verifiable_claims__:
            raise InvalidPublisherError('No required verifiable claims')
        unaccounted_claims = sorted(list(signed_claims.keys() - self.all_known_claims()))
        if unaccounted_claims:
            with sentry_sdk.push_scope() as scope:
                scope.fingerprint = unaccounted_claims
                sentry_sdk.capture_message(f'JWT for {self.__class__.__name__} has unaccounted claims: {unaccounted_claims}')
        for claim_name in self.__required_verifiable_claims__.keys() | self.__required_unverifiable_claims__:
            signed_claim = signed_claims.get(claim_name)
            if signed_claim is None:
                with sentry_sdk.push_scope() as scope:
                    scope.fingerprint = [claim_name]
                    sentry_sdk.capture_message(f'JWT for {self.__class__.__name__} is missing claim: {claim_name}')
                raise InvalidPublisherError(f'Missing claim {claim_name!r}')
        for (claim_name, check) in self.__required_verifiable_claims__.items():
            signed_claim = signed_claims.get(claim_name)
            if not check(getattr(self, claim_name), signed_claim, signed_claims):
                raise InvalidPublisherError(f'Check failed for required claim {claim_name!r}')
        for (claim_name, check) in self.__optional_verifiable_claims__.items():
            signed_claim = signed_claims.get(claim_name)
            if not check(getattr(self, claim_name), signed_claim, signed_claims):
                raise InvalidPublisherError(f'Check failed for optional claim {claim_name!r}')
        return True

    @property
    def publisher_name(self) -> str:
        if False:
            return 10
        raise NotImplementedError

    def publisher_url(self, claims=None) -> str | None:
        if False:
            print('Hello World!')
        '\n        NOTE: This is **NOT** a `@property` because we pass `claims` to it.\n        When calling, make sure to use `publisher_url()`\n        '
        raise NotImplementedError

class OIDCPublisher(OIDCPublisherMixin, db.Model):
    __tablename__ = 'oidc_publishers'
    projects: Mapped[list[Project]] = orm.relationship(secondary=OIDCPublisherProjectAssociation.__table__, back_populates='oidc_publishers')
    macaroons: Mapped[list[Macaroon]] = orm.relationship(cascade='all, delete-orphan', lazy=True)
    __mapper_args__ = {'polymorphic_identity': 'oidc_publishers', 'polymorphic_on': OIDCPublisherMixin.discriminator}

class PendingOIDCPublisher(OIDCPublisherMixin, db.Model):
    """
    A "pending" OIDC publisher, i.e. one that's been registered by a user
    but doesn't correspond to an existing PyPI project yet.
    """
    __tablename__ = 'pending_oidc_publishers'
    project_name = mapped_column(String, nullable=False)
    added_by_id = mapped_column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    added_by: Mapped[User] = orm.relationship(back_populates='pending_oidc_publishers')
    __mapper_args__ = {'polymorphic_identity': 'pending_oidc_publishers', 'polymorphic_on': OIDCPublisherMixin.discriminator}

    def reify(self, session):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an equivalent "normal" OIDC publisher model for this pending publisher,\n        deleting the pending publisher in the process.\n        '
        raise NotImplementedError