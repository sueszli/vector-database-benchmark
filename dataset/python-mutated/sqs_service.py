from .sqs_receive import receive

class SqsService(object):
    name = 'sqs-service'

    @receive('https://sqs.eu-west-1.amazonaws.com/123456789012/nameko-sqs')
    def handle_sqs_message(self, body):
        if False:
            for i in range(10):
                print('nop')
        ' This method is called by the `receive` entrypoint whenever\n        a message sent to the given SQS queue.\n        '
        print(body)
        return body