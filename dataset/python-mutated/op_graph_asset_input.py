from dagster import asset, job, op

@asset
def emails_to_send():
    if False:
        while True:
            i = 10
    ...

@op
def send_emails(emails) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

@job
def send_emails_job():
    if False:
        while True:
            i = 10
    send_emails(emails_to_send.to_source_asset())