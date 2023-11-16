from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.conf import settings
from celery import shared_task
from sentry_sdk import capture_exception

@shared_task
def forgot_password(first_name, email, uidb64, token, current_site):
    if False:
        return 10
    try:
        realtivelink = f'/accounts/reset-password/?uidb64={uidb64}&token={token}'
        abs_url = current_site + realtivelink
        from_email_string = settings.EMAIL_FROM
        subject = 'Reset Your Password - Plane'
        context = {'first_name': first_name, 'forgot_password_url': abs_url}
        html_content = render_to_string('emails/auth/forgot_password.html', context)
        text_content = strip_tags(html_content)
        msg = EmailMultiAlternatives(subject, text_content, from_email_string, [email])
        msg.attach_alternative(html_content, 'text/html')
        msg.send()
        return
    except Exception as e:
        if settings.DEBUG:
            print(e)
        capture_exception(e)
        return