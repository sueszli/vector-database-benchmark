from prowler.providers.aws.lib.audit_info.audit_info import current_audit_info
from prowler.providers.aws.services.macie.macie_service import Macie

macie_client = Macie(current_audit_info)
