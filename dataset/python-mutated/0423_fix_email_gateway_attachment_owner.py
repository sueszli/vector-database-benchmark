from datetime import timedelta
from django.conf import settings
from django.db import connection, migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from psycopg2.sql import SQL, Identifier, Literal

def fix_email_gateway_attachment_owner(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    Realm = apps.get_model('zerver', 'Realm')
    UserProfile = apps.get_model('zerver', 'UserProfile')
    Client = apps.get_model('zerver', 'Client')
    Message = apps.get_model('zerver', 'Message')
    ArchivedMessage = apps.get_model('zerver', 'ArchivedMessage')
    Stream = apps.get_model('zerver', 'Stream')
    Attachment = apps.get_model('zerver', 'Attachment')
    ArchivedAttachment = apps.get_model('zerver', 'ArchivedAttachment')
    if not Realm.objects.exists():
        return
    mail_gateway_bot = UserProfile.objects.get(email__iexact=settings.EMAIL_GATEWAY_BOT)
    (internal_client, _) = Client.objects.get_or_create(name='Internal')
    orphan_attachments = Attachment.objects.filter(messages=None, owner_id=mail_gateway_bot.id)
    if len(orphan_attachments) == 0:
        return
    print('')
    print(f'Found {len(orphan_attachments)} email gateway attachments to reattach')
    for attachment in orphan_attachments:
        print(f'Looking for a message to attach {attachment.path_id}, created {attachment.create_time}')
        possible_matches = []
        for model_class in (Message, ArchivedMessage):
            possible_matches.extend(model_class.objects.filter(has_attachment=False, realm_id=attachment.realm_id, sending_client_id=internal_client.id, date_sent__gte=attachment.create_time, date_sent__lte=attachment.create_time + timedelta(minutes=5), content__contains='/user_uploads/' + attachment.path_id).order_by('date_sent'))
        if len(possible_matches) == 0:
            print('  No matches!')
            continue
        message = possible_matches[0]
        print(f'  Found {message.id} @ {message.date_sent} by {message.sender.delivery_email})')
        if isinstance(message, ArchivedMessage):
            fields = list(Attachment._meta.fields)
            src_fields = [Identifier('zerver_attachment', field.column) for field in fields]
            dst_fields = [Identifier(field.column) for field in fields]
            with connection.cursor() as cursor:
                raw_query = SQL('\n                    INSERT INTO zerver_archivedattachment ({dst_fields})\n                        SELECT {src_fields}\n                        FROM zerver_attachment\n                        WHERE id = {id}\n                    ON CONFLICT (id) DO NOTHING\n                    RETURNING id\n                    ')
                cursor.execute(raw_query.format(src_fields=SQL(',').join(src_fields), dst_fields=SQL(',').join(dst_fields), id=Literal(attachment.id)))
                archived_ids = [id for (id,) in cursor.fetchall()]
                if len(archived_ids) != 1:
                    print('!!! Did not create one archived attachment row!')
            attachment.delete()
            attachment = ArchivedAttachment.objects.get(id=archived_ids[0])
        is_message_realm_public = False
        is_message_web_public = False
        if message.recipient.type == 2:
            stream = Stream.objects.get(id=message.recipient.type_id)
            is_message_realm_public = not stream.invite_only and (not stream.is_in_zephyr_realm)
            is_message_web_public = stream.is_web_public
        attachment.owner_id = message.sender_id
        attachment.is_web_public = is_message_web_public
        attachment.is_realm_public = is_message_realm_public
        attachment.save(update_fields=['owner_id', 'is_web_public', 'is_realm_public'])
        if isinstance(attachment, ArchivedAttachment):
            assert isinstance(message, ArchivedMessage)
            with connection.cursor() as cursor:
                raw_query = SQL("\n                    INSERT INTO zerver_archivedattachment_messages\n                           (id, archivedattachment_id, archivedmessage_id)\n                    VALUES (nextval(pg_get_serial_sequence('zerver_attachment_messages', 'id')),\n                            {attachment_id}, {message_id})\n                    ")
                cursor.execute(raw_query.format(attachment_id=Literal(attachment.id), message_id=Literal(message.id)))
        else:
            assert isinstance(message, Message)
            attachment.messages.add(message)
        message.has_attachment = True
        message.save(update_fields=['has_attachment'])

class Migration(migrations.Migration):
    """
    Messages sent "as" a user via the email gateway had their
    attachments left orphan, accidentally owned by the email gateway
    bot.  Find each such orphaned attachment, and re-own it and attach
    it to the appropriate message.

    """
    dependencies = [('zerver', '0422_multiuseinvite_status')]
    operations = [migrations.RunPython(fix_email_gateway_attachment_owner, reverse_code=migrations.RunPython.noop, elidable=True)]