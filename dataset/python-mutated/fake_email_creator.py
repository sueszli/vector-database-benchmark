from faker import Faker
from ray import serve

@serve.deployment
def create_fake_email():
    if False:
        return 10
    return Faker().email()
app = create_fake_email.bind()
handle = serve.run(app)
assert handle.remote().result() == 'fake@fake.com'