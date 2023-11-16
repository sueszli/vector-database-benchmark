def is_account_only_allowed_in_condition(condition_statement: dict, source_account: str):
    if False:
        i = 10
        return i + 15
    '\n    is_account_only_allowed_in_condition parses the IAM Condition policy block and returns True if the source_account passed as argument is within, False if not.\n\n    @param condition_statement: dict with an IAM Condition block, e.g.:\n        {\n            "StringLike": {\n                "AWS:SourceAccount": 111122223333\n            }\n        }\n\n    @param source_account: str with a 12-digit AWS Account number, e.g.: 111122223333\n    '
    is_condition_valid = False
    valid_condition_options = {'StringEquals': ['aws:sourceaccount', 'aws:sourceowner', 's3:resourceaccount', 'aws:principalaccount', 'aws:resourceaccount', 'aws:sourcearn'], 'StringLike': ['aws:sourceaccount', 'aws:sourceowner', 'aws:sourcearn', 'aws:principalarn', 'aws:resourceaccount', 'aws:principalaccount'], 'ArnLike': ['aws:sourcearn', 'aws:principalarn'], 'ArnEquals': ['aws:sourcearn', 'aws:principalarn']}
    for (condition_operator, condition_operator_key) in valid_condition_options.items():
        if condition_operator in condition_statement:
            for value in condition_operator_key:
                condition_statement[condition_operator] = {k.lower(): v for (k, v) in condition_statement[condition_operator].items()}
                if value in condition_statement[condition_operator]:
                    if isinstance(condition_statement[condition_operator][value], list):
                        is_condition_key_restrictive = True
                        for item in condition_statement[condition_operator][value]:
                            if source_account not in item:
                                is_condition_key_restrictive = False
                                break
                        if is_condition_key_restrictive:
                            is_condition_valid = True
                    elif isinstance(condition_statement[condition_operator][value], str):
                        if source_account in condition_statement[condition_operator][value]:
                            is_condition_valid = True
    return is_condition_valid