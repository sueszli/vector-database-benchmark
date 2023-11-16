"""Algorithm implementation for verifying Azure Confidential Ledger write
transaction receipts."""
from base64 import b64decode
from hashlib import sha256
from typing import Dict, List, Any, cast, Optional
from cryptography.x509 import load_pem_x509_certificate, Certificate
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from azure.confidentialledger.receipt._receipt_models import LeafComponents, ProofElement, Receipt
from azure.confidentialledger.receipt._utils import _convert_dict_to_camel_case
from azure.confidentialledger.receipt._claims_digest_computation import compute_claims_digest

def verify_receipt(receipt: Dict[str, Any], service_cert: str, *, application_claims: Optional[List[Dict[str, Any]]]=None) -> None:
    if False:
        while True:
            i = 10
    'Verify that a given Azure Confidential Ledger write transaction receipt\n    is valid from its content and the Confidential Ledger service identity\n    certificate.\n\n    :param receipt: Receipt dictionary containing the content of an Azure\n     Confidential Ledger write transaction receipt.\n    :type receipt: Dict[str, Any]\n\n    :param service_cert: String containing the PEM-encoded\n     certificate of the Confidential Ledger service identity.\n    :type service_cert: str\n\n    :keyword application_claims: List of application claims to be verified against the receipt.\n    :paramtype application_claims: Optional[List[Dict[str, Any]]]\n\n    :raises ValueError: If the receipt verification has failed.\n    '
    receipt_obj = _preprocess_input_receipt(receipt)
    if application_claims:
        computed_claims_digest = compute_claims_digest(application_claims)
        if computed_claims_digest != receipt_obj.leafComponents.claimsDigest:
            raise ValueError('The computed claims digest from application claims does not match the receipt claims digest.')
    node_cert = _load_and_verify_pem_certificate(receipt_obj.cert)
    _verify_node_cert_endorsed_by_service_cert(node_cert, service_cert, cast(List[str], receipt_obj.serviceEndorsements))
    leaf_node_hash = _compute_leaf_node_hash(receipt_obj.leafComponents)
    root_node_hash = _compute_root_node_hash(leaf_node_hash, receipt_obj.proof)
    _verify_signature_over_root_node_hash(receipt_obj.signature, node_cert, cast(str, receipt_obj.nodeId), root_node_hash)

def _preprocess_input_receipt(receipt_dict: Dict[str, Any]) -> Receipt:
    if False:
        i = 10
        return i + 15
    'Preprocess input receipt dictionary, validate its content, and returns a\n    valid Receipt object based on the vetted input data.\n\n    :param dict[str, any] receipt_dict: Receipt dictionary\n    :return: Receipt object\n    :rtype: Receipt\n    '
    receipt_dict = _convert_dict_to_camel_case(receipt_dict)
    _validate_receipt_content(receipt_dict)
    return Receipt.from_dict(receipt_dict)

def _validate_receipt_content(receipt: Dict[str, Any]):
    if False:
        for i in range(10):
            print('nop')
    'Validate the content of a write transaction receipt.\n\n    :param dict[str, any] receipt: Receipt dictionary\n    '
    try:
        assert 'cert' in receipt
        assert isinstance(receipt['cert'], str)
        assert 'leafComponents' in receipt
        assert isinstance(receipt['leafComponents'], dict)
        assert 'claimsDigest' in receipt['leafComponents']
        assert isinstance(receipt['leafComponents']['claimsDigest'], str)
        assert 'commitEvidence' in receipt['leafComponents']
        assert isinstance(receipt['leafComponents']['commitEvidence'], str)
        assert 'writeSetDigest' in receipt['leafComponents']
        assert isinstance(receipt['leafComponents']['writeSetDigest'], str)
        assert 'proof' in receipt
        assert isinstance(receipt['proof'], list)
        for elem in receipt['proof']:
            assert 'left' in elem or 'right' in elem
            if 'left' in elem:
                assert isinstance(elem['left'], str)
            if 'right' in elem:
                assert isinstance(elem['right'], str)
        assert 'signature' in receipt
        assert isinstance(receipt['signature'], str)
        if 'nodeId' in receipt:
            assert isinstance(receipt['nodeId'], str)
        if 'serviceEndorsements' in receipt:
            assert isinstance(receipt['serviceEndorsements'], list)
            for elem in receipt['serviceEndorsements']:
                assert isinstance(elem, str)
    except Exception as exception:
        raise ValueError('The receipt content is invalid.') from exception

def _verify_signature_over_root_node_hash(signature: str, node_cert: Certificate, node_id: str, root_node_hash: bytes) -> None:
    if False:
        print('Hello World!')
    'Verify signature over root node hash of the Merkle Tree using node\n    certificate public key.\n\n    :param str signature: Signature\n    :param Certificate node_cert: Node certificate\n    :param str node_id: Node ID\n    :param bytes root_node_hash: Root node hash\n    '
    try:
        public_key_bytes = node_cert.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
        if node_id is not None:
            assert sha256(public_key_bytes).digest() == bytes.fromhex(node_id)
        _verify_ec_signature(node_cert, b64decode(signature, validate=True), root_node_hash, hashes.SHA256())
    except Exception as exception:
        raise ValueError(f'Encountered exception when verifying signature {signature} over root node hash.') from exception

def _compute_leaf_node_hash(leaf_components: LeafComponents) -> bytes:
    if False:
        while True:
            i = 10
    'Compute the hash of the leaf node associated to a transaction given the\n    leaf components from a write transaction receipt.\n\n    :param LeafComponents leaf_components: Leaf components\n    :return: Leaf node hash\n    :rtype: bytes\n    '
    try:
        commit_evidence_digest = sha256(leaf_components.commitEvidence.encode()).digest()
        write_set_digest = bytes.fromhex(leaf_components.writeSetDigest)
        claims_digest = bytes.fromhex(leaf_components.claimsDigest)
        return sha256(write_set_digest + commit_evidence_digest + claims_digest).digest()
    except Exception as exception:
        raise ValueError(f'Encountered exception when computing leaf node hash from leaf components {leaf_components}.') from exception

def _compute_root_node_hash(leaf_hash: bytes, proof: List[ProofElement]) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    'Re-compute the hash of the root of the Merkle tree from a leaf node hash\n    and a receipt proof list containing the required nodes hashes for the\n    computation.\n\n    :param bytes leaf_hash: Leaf node hash\n    :param list[ProofElement] proof: Receipt proof list\n    :return: Root node hash\n    :rtype: bytes\n    '
    try:
        current_node_hash = leaf_hash
        for element in proof:
            if element is None or (element.left is None and element.right is None) or (element.left is not None and element.right is not None):
                raise ValueError('Invalid proof element in receipt: element must contain either one left or right node hash.')
            parent_node_hash = bytes()
            if element.left is not None:
                parent_node_hash = bytes.fromhex(element.left) + current_node_hash
            if element.right is not None:
                parent_node_hash = current_node_hash + bytes.fromhex(element.right)
            current_node_hash = sha256(parent_node_hash).digest()
        return current_node_hash
    except Exception as exception:
        raise ValueError(f'Encountered exception when computing root node hash from proof list {proof}.') from exception

def _verify_certificate_endorsement(endorsee: Certificate, endorser: Certificate) -> None:
    if False:
        while True:
            i = 10
    'Verify that the endorser certificate has endorsed endorsee\n    certificate using ECDSA.\n\n    :param Certificate endorsee: Endorsee certificate\n    :param Certificate endorser: Endorser certificate\n    '
    try:
        hash_algorithm = cast(hashes.HashAlgorithm, endorsee.signature_hash_algorithm)
        digester = hashes.Hash(hash_algorithm)
        digester.update(endorsee.tbs_certificate_bytes)
        cert_digest = digester.finalize()
        _verify_ec_signature(endorser, endorsee.signature, cert_digest, hash_algorithm)
    except Exception as exception:
        raise ValueError(f'Encountered exception when verifying endorsement of certificate {endorsee} by certificate {endorser}.') from exception

def _verify_ec_signature(certificate: Certificate, signature: bytes, data: bytes, hash_algorithm: hashes.HashAlgorithm) -> None:
    if False:
        return 10
    'Verify a signature over data using the certificate public key.\n\n    :param Certificate certificate: Certificate\n    :param bytes signature: Signature\n    :param bytes data: Data\n    :param hashes.HashAlgorithm hash_algorithm: Hash algorithm\n    '
    public_key = cast(ec.EllipticCurvePublicKey, certificate.public_key())
    public_key.verify(signature, data, ec.ECDSA(utils.Prehashed(hash_algorithm)))

def _verify_node_cert_endorsed_by_service_cert(node_cert: Certificate, service_cert_str: str, endorsements_certs: List[str]) -> None:
    if False:
        print('Hello World!')
    'Check a node certificate is endorsed by a service certificate.\n\n    If a list of endorsements certificates is not empty, check that the\n    node certificate is transitively endorsed by the service certificate\n    through the endorsement certificates in the list.\n\n    :param Certificate node_cert: Node certificate\n    :param str service_cert_str: Service certificate string\n    :param list[str] endorsements_certs: Endorsements certificates list\n    '
    current_cert = node_cert
    if endorsements_certs is None:
        endorsements_certs = []
    endorsements_certs.append(service_cert_str)
    for endorsement in endorsements_certs:
        endorsement_cert = _load_and_verify_pem_certificate(endorsement)
        _verify_certificate_endorsement(current_cert, endorsement_cert)
        current_cert = endorsement_cert

def _load_and_verify_pem_certificate(cert_str: str) -> Certificate:
    if False:
        print('Hello World!')
    'Load PEM certificate from a string representation and verify it is a\n    valid certificate with expected Elliptic Curve public key.\n\n    :param str cert_str: PEM certificate string\n    :return: Certificate\n    :rtype: Certificate\n    '
    try:
        cert = load_pem_x509_certificate(cert_str.encode())
        assert isinstance(cert.public_key(), ec.EllipticCurvePublicKey)
        return cert
    except Exception as exception:
        raise ValueError(f'PEM certificate {cert_str} is not valid.') from exception