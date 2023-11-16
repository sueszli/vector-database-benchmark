import stripe

def init_app(app):
    if False:
        for i in range(10):
            print('nop')
    if app.config.get('STRIPE_API_KEY'):
        stripe.api_key = app.config.get('STRIPE_API_KEY')