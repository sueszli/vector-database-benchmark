def handler(event, ctx):
    if False:
        return 10
    verification_token = event['verification_token']
    print(f'verification_token={verification_token!r}')
    return {'verification_token': verification_token}