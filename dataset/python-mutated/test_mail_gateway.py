import socket
from odoo.addons.mail.tests.common import TestMail
from odoo.tools import mute_logger
MAIL_TEMPLATE = 'Return-Path: <whatever-2a840@postmaster.twitter.com>\nTo: {to}\ncc: {cc}\nReceived: by mail1.openerp.com (Postfix, from userid 10002)\n    id 5DF9ABFB2A; Fri, 10 Aug 2012 16:16:39 +0200 (CEST)\nFrom: {email_from}\nSubject: {subject}\nMIME-Version: 1.0\nContent-Type: multipart/alternative;\n    boundary="----=_Part_4200734_24778174.1344608186754"\nDate: Fri, 10 Aug 2012 14:16:26 +0000\nMessage-ID: {msg_id}\n{extra}\n------=_Part_4200734_24778174.1344608186754\nContent-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: quoted-printable\n\nPlease call me as soon as possible this afternoon!\n\n--\nSylvie\n------=_Part_4200734_24778174.1344608186754\nContent-Type: text/html; charset=utf-8\nContent-Transfer-Encoding: quoted-printable\n\n<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n<html>\n <head>=20\n  <meta http-equiv=3D"Content-Type" content=3D"text/html; charset=3Dutf-8" />\n </head>=20\n <body style=3D"margin: 0; padding: 0; background: #ffffff;-webkit-text-size-adjust: 100%;">=20\n\n  <p>Please call me as soon as possible this afternoon!</p>\n\n  <p>--<br/>\n     Sylvie\n  <p>\n </body>\n</html>\n------=_Part_4200734_24778174.1344608186754--\n'
MAIL_TEMPLATE_PLAINTEXT = 'Return-Path: <whatever-2a840@postmaster.twitter.com>\nTo: {to}\nReceived: by mail1.openerp.com (Postfix, from userid 10002)\n    id 5DF9ABFB2A; Fri, 10 Aug 2012 16:16:39 +0200 (CEST)\nFrom: Sylvie Lelitre <test.sylvie.lelitre@agrolait.com>\nSubject: {subject}\nMIME-Version: 1.0\nContent-Type: text/plain\nDate: Fri, 10 Aug 2012 14:16:26 +0000\nMessage-ID: {msg_id}\n{extra}\n\nPlease call me as soon as possible this afternoon!\n\n--\nSylvie\n'
MAIL_MULTIPART_MIXED = 'Return-Path: <ignasse.carambar@gmail.com>\nX-Original-To: raoul@grosbedon.fr\nDelivered-To: raoul@grosbedon.fr\nReceived: by mail1.grosbedon.com (Postfix, from userid 10002)\n    id E8166BFACA; Fri, 23 Aug 2013 13:18:01 +0200 (CEST)\nX-Spam-Checker-Version: SpamAssassin 3.3.1 (2010-03-16) on mail1.grosbedon.com\nX-Spam-Level:\nX-Spam-Status: No, score=-2.6 required=5.0 tests=BAYES_00,FREEMAIL_FROM,\n    HTML_MESSAGE,RCVD_IN_DNSWL_LOW autolearn=unavailable version=3.3.1\nReceived: from mail-ie0-f173.google.com (mail-ie0-f173.google.com [209.85.223.173])\n    by mail1.grosbedon.com (Postfix) with ESMTPS id 9BBD7BFAAA\n    for <raoul@openerp.fr>; Fri, 23 Aug 2013 13:17:55 +0200 (CEST)\nReceived: by mail-ie0-f173.google.com with SMTP id qd12so575130ieb.4\n        for <raoul@grosbedon.fr>; Fri, 23 Aug 2013 04:17:54 -0700 (PDT)\nDKIM-Signature: v=1; a=rsa-sha256; c=relaxed/relaxed;\n        d=gmail.com; s=20120113;\n        h=mime-version:date:message-id:subject:from:to:content-type;\n        bh=dMNHV52EC7GAa7+9a9tqwT9joy9z+1950J/3A6/M/hU=;\n        b=DGuv0VjegdSrEe36ADC8XZ9Inrb3Iu+3/52Bm+caltddXFH9yewTr0JkCRQaJgMwG9\n         qXTQgP8qu/VFEbCh6scu5ZgU1hknzlNCYr3LT+Ih7dAZVUEHUJdwjzUU1LFV95G2RaCd\n         /Lwff6CibuUvrA+0CBO7IRKW0Sn5j0mukYu8dbaKsm6ou6HqS8Nuj85fcXJfHSHp6Y9u\n         dmE8jBh3fHCHF/nAvU+8aBNSIzl1FGfiBYb2jCoapIuVFitKR4q5cuoodpkH9XqqtOdH\n         DG+YjEyi8L7uvdOfN16eMr7hfUkQei1yQgvGu9/5kXoHg9+Gx6VsZIycn4zoaXTV3Nhn\n         nu4g==\nMIME-Version: 1.0\nX-Received: by 10.50.124.65 with SMTP id mg1mr1144467igb.43.1377256674216;\n Fri, 23 Aug 2013 04:17:54 -0700 (PDT)\nReceived: by 10.43.99.71 with HTTP; Fri, 23 Aug 2013 04:17:54 -0700 (PDT)\nDate: Fri, 23 Aug 2013 13:17:54 +0200\nMessage-ID: <CAP76m_V4BY2F7DWHzwfjteyhW8L2LJswVshtmtVym+LUJ=rASQ@mail.gmail.com>\nSubject: Test mail multipart/mixed\nFrom: =?ISO-8859-1?Q?Raoul Grosbedon=E9e?= <ignasse.carambar@gmail.com>\nTo: Followers of ASUSTeK-Joseph-Walters <raoul@grosbedon.fr>\nContent-Type: multipart/mixed; boundary=089e01536c4ed4d17204e49b8e96\n\n--089e01536c4ed4d17204e49b8e96\nContent-Type: multipart/alternative; boundary=089e01536c4ed4d16d04e49b8e94\n\n--089e01536c4ed4d16d04e49b8e94\nContent-Type: text/plain; charset=ISO-8859-1\n\nShould create a multipart/mixed: from gmail, *bold*, with attachment.\n\n--\nMarcel Boitempoils.\n\n--089e01536c4ed4d16d04e49b8e94\nContent-Type: text/html; charset=ISO-8859-1\n\n<div dir="ltr">Should create a multipart/mixed: from gmail, <b>bold</b>, with attachment.<br clear="all"><div><br></div>-- <br>Marcel Boitempoils.</div>\n\n--089e01536c4ed4d16d04e49b8e94--\n--089e01536c4ed4d17204e49b8e96\nContent-Type: text/plain; charset=US-ASCII; name="test.txt"\nContent-Disposition: attachment; filename="test.txt"\nContent-Transfer-Encoding: base64\nX-Attachment-Id: f_hkpb27k00\n\ndGVzdAo=\n--089e01536c4ed4d17204e49b8e96--'
MAIL_MULTIPART_MIXED_TWO = 'X-Original-To: raoul@grosbedon.fr\nDelivered-To: raoul@grosbedon.fr\nReceived: by mail1.grosbedon.com (Postfix, from userid 10002)\n    id E8166BFACA; Fri, 23 Aug 2013 13:18:01 +0200 (CEST)\nFrom: "Bruce Wayne" <bruce@wayneenterprises.com>\nContent-Type: multipart/alternative;\n boundary="Apple-Mail=_9331E12B-8BD2-4EC7-B53E-01F3FBEC9227"\nMessage-Id: <6BB1FAB2-2104-438E-9447-07AE2C8C4A92@sexample.com>\nMime-Version: 1.0 (Mac OS X Mail 7.3 \\(1878.6\\))\n\n--Apple-Mail=_9331E12B-8BD2-4EC7-B53E-01F3FBEC9227\nContent-Transfer-Encoding: 7bit\nContent-Type: text/plain;\n    charset=us-ascii\n\nFirst and second part\n\n--Apple-Mail=_9331E12B-8BD2-4EC7-B53E-01F3FBEC9227\nContent-Type: multipart/mixed;\n boundary="Apple-Mail=_CA6C687E-6AA0-411E-B0FE-F0ABB4CFED1F"\n\n--Apple-Mail=_CA6C687E-6AA0-411E-B0FE-F0ABB4CFED1F\nContent-Transfer-Encoding: 7bit\nContent-Type: text/html;\n    charset=us-ascii\n\n<html><head></head><body>First part</body></html>\n\n--Apple-Mail=_CA6C687E-6AA0-411E-B0FE-F0ABB4CFED1F\nContent-Disposition: inline;\n    filename=thetruth.pdf\nContent-Type: application/pdf;\n    name="thetruth.pdf"\nContent-Transfer-Encoding: base64\n\nSSBhbSB0aGUgQmF0TWFuCg==\n\n--Apple-Mail=_CA6C687E-6AA0-411E-B0FE-F0ABB4CFED1F\nContent-Transfer-Encoding: 7bit\nContent-Type: text/html;\n    charset=us-ascii\n\n<html><head></head><body>Second part</body></html>\n--Apple-Mail=_CA6C687E-6AA0-411E-B0FE-F0ABB4CFED1F--\n\n--Apple-Mail=_9331E12B-8BD2-4EC7-B53E-01F3FBEC9227--\n'
MAIL_MULTIPART_IMAGE = 'X-Original-To: raoul@example.com\nDelivered-To: micheline@example.com\nReceived: by mail1.example.com (Postfix, from userid 99999)\n    id 9DFB7BF509; Thu, 17 Dec 2015 15:22:56 +0100 (CET)\nX-Spam-Checker-Version: SpamAssassin 3.4.0 (2014-02-07) on mail1.example.com\nX-Spam-Level: *\nX-Spam-Status: No, score=1.1 required=5.0 tests=FREEMAIL_FROM,\n    HTML_IMAGE_ONLY_08,HTML_MESSAGE,RCVD_IN_DNSWL_LOW,RCVD_IN_MSPIKE_H3,\n    RCVD_IN_MSPIKE_WL,T_DKIM_INVALID autolearn=no autolearn_force=no version=3.4.0\nReceived: from mail-lf0-f44.example.com (mail-lf0-f44.example.com [209.85.215.44])\n    by mail1.example.com (Postfix) with ESMTPS id 1D80DBF509\n    for <micheline@example.com>; Thu, 17 Dec 2015 15:22:56 +0100 (CET)\nAuthentication-Results: mail1.example.com; dkim=pass\n    reason="2048-bit key; unprotected key"\n    header.d=example.com header.i=@example.com header.b=kUkTIIlt;\n    dkim-adsp=pass; dkim-atps=neutral\nReceived: by mail-lf0-f44.example.com with SMTP id z124so47959461lfa.3\n        for <micheline@example.com>; Thu, 17 Dec 2015 06:22:56 -0800 (PST)\nDKIM-Signature: v=1; a=rsa-sha256; c=relaxed/relaxed;\n        d=example.com; s=20120113;\n        h=mime-version:date:message-id:subject:from:to:content-type;\n        bh=GdrEuMrz6vxo/Z/F+mJVho/1wSe6hbxLx2SsP8tihzw=;\n        b=kUkTIIlt6fe4dftKHPNBkdHU2rO052o684R0e2bqH7roGUQFb78scYE+kqX0wo1zlk\n         zhKPVBR1TqTsYlqcHu+D3aUzai7L/Q5m40sSGn7uYGkZJ6m1TwrWNqVIgTZibarqvy94\n         NWhrjjK9gqd8segQdSjCgTipNSZME4bJCzPyBg/D5mqe07FPBJBGoF9SmIzEBhYeqLj1\n         GrXjb/D8J11aOyzmVvyt+bT+oeLUJI8E7qO5g2eQkMncyu+TyIXaRofOOBA14NhQ+0nS\n         w5O9rzzqkKuJEG4U2TJ2Vi2nl2tHJW2QPfTtFgcCzGxQ0+5n88OVlbGTLnhEIJ/SYpem\n         O5EA==\nMIME-Version: 1.0\nX-Received: by 10.25.167.197 with SMTP id q188mr22222517lfe.129.1450362175493;\n Thu, 17 Dec 2015 06:22:55 -0800 (PST)\nReceived: by 10.25.209.145 with HTTP; Thu, 17 Dec 2015 06:22:55 -0800 (PST)\nDate: Thu, 17 Dec 2015 15:22:55 +0100\nMessage-ID: <CAP76m_UB=aLqWEFccnq86AhkpwRB3aZoGL9vMffX7co3YEro_A@mail.gmail.com>\nSubject: {subject}\nFrom: =?UTF-8?Q?Thibault_Delavall=C3=A9e?= <raoul@example.com>\nTo: {to}\nContent-Type: multipart/related; boundary=001a11416b9e9b229a05272b7052\n\n--001a11416b9e9b229a05272b7052\nContent-Type: multipart/alternative; boundary=001a11416b9e9b229805272b7051\n\n--001a11416b9e9b229805272b7051\nContent-Type: text/plain; charset=UTF-8\nContent-Transfer-Encoding: quoted-printable\n\nPremi=C3=A8re image, orang=C3=A9e.\n\n[image: Inline image 1]\n\nSeconde image, rosa=C3=A7=C3=A9e.\n\n[image: Inline image 2]\n\nTroisi=C3=A8me image, verte!=C2=B5\n\n[image: Inline image 3]\n\nJ\'esp=C3=A8re que tout se passera bien.\n--=20\nThibault Delavall=C3=A9e\n\n--001a11416b9e9b229805272b7051\nContent-Type: text/html; charset=UTF-8\nContent-Transfer-Encoding: quoted-printable\n\n<div dir=3D"ltr"><div>Premi=C3=A8re image, orang=C3=A9e.</div><div><br></di=\nv><div><img src=3D"cid:ii_151b519fc025fdd3" alt=3D"Inline image 1" width=3D=\n"2" height=3D"2"><br></div><div><br></div><div>Seconde image, rosa=C3=A7=C3=\n=A9e.</div><div><br></div><div><img src=3D"cid:ii_151b51a290ed6a91" alt=3D"=\nInline image 2" width=3D"2" height=3D"2"></div><div><br></div><div>Troisi=\n=C3=A8me image, verte!=C2=B5</div><div><br></div><div><img src=3D"cid:ii_15=\n1b51a37e5eb7a6" alt=3D"Inline image 3" width=3D"10" height=3D"10"><br></div=\n><div><br></div><div>J&#39;esp=C3=A8re que tout se passera bien.</div>-- <b=\nr><div class=3D"gmail_signature">Thibault Delavall=C3=A9e</div>\n</div>\n\n--001a11416b9e9b229805272b7051--\n--001a11416b9e9b229a05272b7052\nContent-Type: image/gif; name="=?UTF-8?B?b3JhbmfDqWUuZ2lm?="\nContent-Disposition: inline; filename="=?UTF-8?B?b3JhbmfDqWUuZ2lm?="\nContent-Transfer-Encoding: base64\nContent-ID: <ii_151b519fc025fdd3>\nX-Attachment-Id: ii_151b519fc025fdd3\n\nR0lGODdhAgACALMAAAAAAP///wAAAP//AP8AAP+AAAD/AAAAAAAA//8A/wAAAAAAAAAAAAAAAAAA\nAAAAACwAAAAAAgACAAAEA7DIEgA7\n--001a11416b9e9b229a05272b7052\nContent-Type: image/gif; name="=?UTF-8?B?dmVydGUhwrUuZ2lm?="\nContent-Disposition: inline; filename="=?UTF-8?B?dmVydGUhwrUuZ2lm?="\nContent-Transfer-Encoding: base64\nContent-ID: <ii_151b51a37e5eb7a6>\nX-Attachment-Id: ii_151b51a37e5eb7a6\n\nR0lGODlhCgAKALMAAAAAAIAAAACAAICAAAAAgIAAgACAgMDAwICAgP8AAAD/AP//AAAA//8A/wD/\n/////ywAAAAACgAKAAAEClDJSau9OOvNe44AOw==\n--001a11416b9e9b229a05272b7052\nContent-Type: image/gif; name="=?UTF-8?B?cm9zYcOnw6llLmdpZg==?="\nContent-Disposition: inline; filename="=?UTF-8?B?cm9zYcOnw6llLmdpZg==?="\nContent-Transfer-Encoding: base64\nContent-ID: <ii_151b51a290ed6a91>\nX-Attachment-Id: ii_151b51a290ed6a91\n\nR0lGODdhAgACALMAAAAAAP///wAAAP//AP8AAP+AAAD/AAAAAAAA//8A/wAAAP+AgAAAAAAAAAAA\nAAAAACwAAAAAAgACAAAEA3DJFQA7\n--001a11416b9e9b229a05272b7052--\n'

class TestMailgateway(TestMail):

    def setUp(self):
        if False:
            return 10
        super(TestMailgateway, self).setUp()
        self.mail_channel_model = self.env['ir.model'].search([('model', '=', 'mail.channel')], limit=1)
        self.alias = self.env['mail.alias'].create({'alias_name': 'groups', 'alias_user_id': False, 'alias_model_id': self.mail_channel_model.id, 'alias_contact': 'everyone'})
        self.mail_test_model = self.env['ir.model'].search([('model', '=', 'mail.test')], limit=1)
        self.alias_2 = self.env['mail.alias'].create({'alias_name': 'test', 'alias_user_id': False, 'alias_model_id': self.mail_test_model.id, 'alias_contact': 'everyone'})
        self.fake_email = self.env['mail.message'].create({'model': 'mail.channel', 'res_id': self.group_public.id, 'subject': 'Public Discussion', 'message_type': 'email', 'author_id': self.partner_1.id, 'message_id': '<123456-openerp-%s-mail.channel@%s>' % (self.group_public.id, socket.gethostname())})

    @mute_logger('odoo.addons.mail.models.mail_thread')
    def test_message_parse(self):
        if False:
            print('Hello World!')
        ' Test parsing of various scenarios of incoming emails '
        res = self.env['mail.thread'].message_parse(MAIL_TEMPLATE_PLAINTEXT)
        self.assertIn('Please call me as soon as possible this afternoon!', res.get('body', ''), 'message_parse: missing text in text/plain body after parsing')
        res = self.env['mail.thread'].message_parse(MAIL_TEMPLATE)
        self.assertIn('<p>Please call me as soon as possible this afternoon!</p>', res.get('body', ''), 'message_parse: missing html in multipart/alternative body after parsing')
        res = self.env['mail.thread'].message_parse(MAIL_MULTIPART_MIXED)
        self.assertNotIn('Should create a multipart/mixed: from gmail, *bold*, with attachment', res.get('body', ''), 'message_parse: text version should not be in body after parsing multipart/mixed')
        self.assertIn('<div dir="ltr">Should create a multipart/mixed: from gmail, <b>bold</b>, with attachment.<br clear="all"><div><br></div>', res.get('body', ''), 'message_parse: html version should be in body after parsing multipart/mixed')
        res = self.env['mail.thread'].message_parse(MAIL_MULTIPART_MIXED_TWO)
        self.assertNotIn('First and second part', res.get('body', ''), 'message_parse: text version should not be in body after parsing multipart/mixed')
        self.assertIn('First part', res.get('body', ''), 'message_parse: first part of the html version should be in body after parsing multipart/mixed')
        self.assertIn('Second part', res.get('body', ''), 'message_parse: second part of the html version should be in body after parsing multipart/mixed')

    @mute_logger('odoo.addons.mail.models.mail_thread')
    def test_message_process_cid(self):
        if False:
            while True:
                i = 10
        new_groups = self.format_and_process(MAIL_MULTIPART_IMAGE, subject='My Frogs', to='groups@example.com')
        message = new_groups.message_ids[0]
        for attachment in message.attachment_ids:
            self.assertIn('/web/image/%s' % attachment.id, message.body)

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_alias_basic(self):
        if False:
            print('Hello World!')
        ' Incoming email on an alias creating a new record + message_new + message details '
        new_groups = self.format_and_process(MAIL_TEMPLATE, subject='My Frogs', to='groups@example.com, other@gmail.com')
        self.assertEqual(len(new_groups), 1, 'message_process: a new mail.channel should have been created')
        res = new_groups.get_metadata()[0].get('create_uid') or [None]
        self.assertEqual(res[0], self.env.uid, 'message_process: group should have been created by uid as alias_user_id is False on the alias')
        self.assertEqual(len(new_groups.message_ids), 1, 'message_process: newly created group should have the incoming email in message_ids')
        msg = new_groups.message_ids[0]
        self.assertEqual(msg.subject, 'My Frogs', 'message_process: newly created group should have the incoming email as first message')
        self.assertIn('Please call me as soon as possible this afternoon!', msg.body, 'message_process: newly created group should have the incoming email as first message')
        self.assertEqual(msg.message_type, 'email', 'message_process: newly created group should have an email as first message')
        self.assertEqual(msg.subtype_id, self.env.ref('mail.mt_comment'), 'message_process: newly created group should not have a log first message but an email')
        self.assertEqual(len(self._mails), 0, 'message_process: should create emails without any follower added')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_alias_user_id(self):
        if False:
            while True:
                i = 10
        ' Test alias ownership '
        self.alias.write({'alias_user_id': self.user_employee.id})
        new_groups = self.format_and_process(MAIL_TEMPLATE, to='groups@example.com, other@gmail.com')
        self.assertEqual(len(new_groups), 1, 'message_process: a new mail.channel should have been created')
        res = new_groups.get_metadata()[0].get('create_uid') or [None]
        self.assertEqual(res[0], self.user_employee.id, 'message_process: group should have been created by alias_user_id')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_email_email_from(self):
        if False:
            while True:
                i = 10
        ' Incoming email: not recognized author: email_from, no author_id, no followers '
        new_groups = self.format_and_process(MAIL_TEMPLATE, to='groups@example.com, other@gmail.com')
        self.assertFalse(new_groups.message_ids[0].author_id, 'message_process: unrecognized email -> no author_id')
        self.assertIn('test.sylvie.lelitre@agrolait.com', new_groups.message_ids[0].email_from, 'message_process: unrecognized email -> email_from')
        self.assertEqual(len(new_groups.message_partner_ids), 0, 'message_process: newly create group should not have any follower')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_email_author(self):
        if False:
            print('Hello World!')
        ' Incoming email: recognized author: email_from, author_id, added as follower '
        new_groups = self.format_and_process(MAIL_TEMPLATE, email_from='Valid Lelitre <valid.lelitre@agrolait.com>', to='groups@example.com, valid.other@gmail.com')
        self.assertEqual(new_groups.message_ids[0].author_id, self.partner_1, 'message_process: recognized email -> author_id')
        self.assertIn('Valid Lelitre <valid.lelitre@agrolait.com>', new_groups.message_ids[0].email_from, 'message_process: recognized email -> email_from')
        self.assertEqual(len(self._mails), 0, 'message_process: no bounce or notificatoin email should be sent with follower = author')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models', 'odoo.addons.mail.models.mail_mail')
    def test_message_process_alias_partners_bounce(self):
        if False:
            for i in range(10):
                print('nop')
        ' Incoming email from an unknown partner on a Partners only alias -> bounce '
        self.alias.write({'alias_contact': 'partners'})
        new_groups = self.format_and_process(MAIL_TEMPLATE, subject='New Frogs', to='groups@example.com, other@gmail.com')
        self.assertTrue(len(new_groups) == 0)
        self.assertEqual(len(self._mails), 1, 'message_process: incoming email on Partners alias should send a bounce email')
        self.assertIn('New Frogs', self._mails[0].get('subject'), 'message_process: bounce email on Partners alias should contain the original subject')
        self.assertIn('whatever-2a840@postmaster.twitter.com', self._mails[0].get('email_to'), 'message_process: bounce email on Partners alias should go to Return-Path address')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models', 'odoo.addons.mail.models.mail_mail')
    def test_message_process_alias_followers_bounce(self):
        if False:
            print('Hello World!')
        ' Incoming email from unknown partner / not follower partner on a Followers only alias -> bounce '
        self.alias.write({'alias_contact': 'followers', 'alias_parent_model_id': self.mail_channel_model.id, 'alias_parent_thread_id': self.group_pigs.id})
        new_groups = self.format_and_process(MAIL_TEMPLATE, to='groups@example.com, other@gmail.com')
        self.assertEqual(len(new_groups), 0, 'message_process: should have bounced')
        self.assertEqual(len(self._mails), 1, 'message_process: incoming email on Followers alias should send a bounce email')
        self._init_mock_build_email()
        new_groups = self.format_and_process(MAIL_TEMPLATE, email_from='Valid Lelitre <valid.lelitre@agrolait.com>', to='groups@example.com, other@gmail.com')
        self.assertTrue(len(new_groups) == 0, 'message_process: should have bounced')
        self.assertEqual(len(self._mails), 1, 'message_process: incoming email on Followers alias should send a bounce email')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_alias_partner(self):
        if False:
            return 10
        ' Incoming email from a known partner on a Partners alias -> ok (+ test on alias.user_id) '
        self.alias.write({'alias_contact': 'partners'})
        new_groups = self.format_and_process(MAIL_TEMPLATE, email_from='Valid Lelitre <valid.lelitre@agrolait.com>', to='groups@example.com, valid.other@gmail.com')
        self.assertEqual(len(new_groups), 1, 'message_process: a new mail.channel should have been created')
        self.assertEqual(len(new_groups.message_ids), 1, 'message_process: newly created group should have the incoming email in message_ids')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_alias_followers(self):
        if False:
            for i in range(10):
                print('nop')
        ' Incoming email from a parent document follower on a Followers only alias -> ok '
        self.alias.write({'alias_contact': 'followers', 'alias_parent_model_id': self.mail_channel_model.id, 'alias_parent_thread_id': self.group_pigs.id})
        self.group_pigs.message_subscribe(partner_ids=[self.partner_1.id])
        new_groups = self.format_and_process(MAIL_TEMPLATE, email_from='Valid Lelitre <valid.lelitre@agrolait.com>', to='groups@example.com, other6@gmail.com')
        self.assertEqual(len(new_groups), 1, 'message_process: a new mail.channel should have been created')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models', 'odoo.addons.mail.models.mail_mail')
    def test_message_process_alias_update(self):
        if False:
            return 10
        ' Incoming email update discussion + notification email '
        self.alias.write({'alias_force_thread_id': self.group_public.id})
        self.group_public.message_subscribe(partner_ids=[self.partner_1.id])
        new_groups = self.format_and_process(MAIL_TEMPLATE, email_from='valid.other@gmail.com', msg_id='<1198923581.41972151344608186799.JavaMail.diff1@agrolait.com>', to='groups@example.com>', subject='Re: cats')
        self.assertEqual(len(new_groups), 0, 'message_process: reply on Frogs should not have created a new group with new subject')
        self.assertEqual(len(self.group_public.message_ids), 2, 'message_process: group should contain one new message')
        self.assertEqual(len(self._mails), 1, 'message_process: one email should have been generated')
        self.assertIn('valid.lelitre@agrolait.com', self._mails[0].get('email_to')[0], 'message_process: email should be sent to Sylvie')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_in_reply_to(self):
        if False:
            i = 10
            return i + 15
        ' Incoming email using in-rely-to should go into the right destination even with a wrong destination '
        self.format_and_process(MAIL_TEMPLATE, email_from='valid.other@gmail.com', msg_id='<1198923581.41972151344608186800.JavaMail.diff1@agrolait.com>', to='erroneous@example.com>', subject='Re: news', extra='In-Reply-To:\r\n\t%s\n' % self.fake_email.message_id)
        self.assertEqual(len(self.group_public.message_ids), 2, 'message_process: group should contain one new message')
        self.assertEqual(len(self.fake_email.child_ids), 1, 'message_process: new message should be children of the existing one')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_references(self):
        if False:
            print('Hello World!')
        ' Incoming email using references should go into the right destination even with a wrong destination '
        self.format_and_process(MAIL_TEMPLATE, to='erroneous@example.com', extra='References: <2233@a.com>\r\n\t<3edss_dsa@b.com> %s' % self.fake_email.message_id, msg_id='<1198923581.41972151344608186800.JavaMail.4@agrolait.com>')
        self.assertEqual(len(self.group_public.message_ids), 2, 'message_process: group should contain one new message')
        self.assertEqual(len(self.fake_email.child_ids), 1, 'message_process: new message should be children of the existing one')

    def test_message_process_references_external(self):
        if False:
            return 10
        ' Incoming email being a reply to an external email processed by odoo should update thread accordingly '
        new_message_id = '<ThisIsTooMuchFake.MonsterEmail.789@agrolait.com>'
        self.fake_email.write({'message_id': new_message_id})
        self.format_and_process(MAIL_TEMPLATE, to='erroneous@example.com', extra='References: <2233@a.com>\r\n\t<3edss_dsa@b.com> %s' % self.fake_email.message_id, msg_id='<1198923581.41972151344608186800.JavaMail.4@agrolait.com>')
        self.assertEqual(len(self.group_public.message_ids), 2, 'message_process: group should contain one new message')
        self.assertEqual(len(self.fake_email.child_ids), 1, 'message_process: new message should be children of the existing one')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_references_forward(self):
        if False:
            for i in range(10):
                print('nop')
        ' Incoming email using references but with alias forward should not go into references destination '
        res_test = self.format_and_process(MAIL_TEMPLATE, to='test@example.com', subject='My Dear Forward', extra='References: <2233@a.com>\r\n\t<3edss_dsa@b.com> %s' % self.fake_email.message_id, msg_id='<1198923581.41972151344608186800.JavaMail.4@agrolait.com>', target_model='mail.test')
        self.assertEqual(len(self.group_public.message_ids), 1, 'message_process: group should not contain new message')
        self.assertEqual(len(self.fake_email.child_ids), 0, 'message_process: original email should not contain childs')
        self.assertEqual(res_test.name, 'My Dear Forward')
        self.assertEqual(len(res_test.message_ids), 1)

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_references_forward_cc(self):
        if False:
            return 10
        ' Incoming email using references but with alias forward should not go into references destination '
        self.format_and_process(MAIL_TEMPLATE, to='erroneous@example.com', cc='test@example.com', subject='My Dear Forward', extra='References: <2233@a.com>\r\n\t<3edss_dsa@b.com> %s' % self.fake_email.message_id, msg_id='<1198923581.41972151344608186800.JavaMail.4@agrolait.com>', target_model='mail.test')
        self.assertEqual(len(self.group_public.message_ids), 2, 'message_process: group should contain one new message')
        self.assertEqual(len(self.fake_email.child_ids), 1, 'message_process: new message should be children of the existing one')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_model_res_id(self):
        if False:
            while True:
                i = 10
        ' Incoming email with ref holding model / res_id but that does not match any message in the thread: must raise since OpenERP saas-3 '
        self.assertRaises(ValueError, self.format_and_process, MAIL_TEMPLATE, email_from='valid.lelitre@agrolait.com', to='noone@example.com', subject='spam', extra='In-Reply-To: <12321321-openerp-%d-mail.channel@%s>' % (self.group_public.id, socket.gethostname()), msg_id='<1198923581.41972151344608186802.JavaMail.diff1@agrolait.com>')
        self.fake_email.write({'message_id': False})
        self.assertRaises(ValueError, self.format_and_process, MAIL_TEMPLATE, email_from='other5@gmail.com', msg_id='<1.2.JavaMail.new@agrolait.com>', to='noone@example.com>', subject='spam', extra='In-Reply-To: <12321321-openerp-%d-mail.channel@%s>' % (self.group_public.id, socket.gethostname()))
        self.assertRaises(ValueError, self.format_and_process, MAIL_TEMPLATE, email_from='other5@gmail.com', msg_id='<1.3.JavaMail.new@agrolait.com>', to='noone@example.com>', subject='spam', extra='In-Reply-To: <12321321-openerp-%d-mail.channel@neighbor.com>' % self.group_public.id)
        self.assertEqual(len(self.group_public.message_ids), 1)
        self.assertEqual(len(self.group_public.message_ids[0].child_ids), 0)

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_duplicate(self):
        if False:
            while True:
                i = 10
        ' Duplicate emails (same message_id) are not processed '
        self.alias.write({'alias_force_thread_id': self.group_public.id})
        frog_groups = self.format_and_process(MAIL_TEMPLATE, email_from='valid.other@gmail.com', subject='Re: super cats', msg_id='<1198923581.41972151344608186799.JavaMail.diff1@agrolait.com>')
        frog_groups = self.format_and_process(MAIL_TEMPLATE, email_from='other4@gmail.com', subject='Re: news', msg_id='<1198923581.41972151344608186799.JavaMail.diff1@agrolait.com>', extra='In-Reply-To: <1198923581.41972151344608186799.JavaMail.diff1@agrolait.com>\n')
        self.assertEqual(len(frog_groups), 0, 'message_process: reply on Frogs should not have created a new group with new subject')
        self.assertEqual(len(self.group_public.message_ids), 2, 'message_process: message with already existing message_id should not have been duplicated')
        no_of_msg = self.env['mail.message'].search_count([('message_id', 'ilike', '<1198923581.41972151344608186799.JavaMail.diff1@agrolait.com>')])
        self.assertEqual(no_of_msg, 1, 'message_process: message with already existing message_id should not have been duplicated')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_partner_find(self):
        if False:
            while True:
                i = 10
        ' Finding the partner based on email, based on partner / user / follower '
        from_1 = self.env['res.partner'].create({'name': 'A', 'email': 'from.test@example.com'})
        self.format_and_process(MAIL_TEMPLATE, to='public@example.com', msg_id='<1>', email_from='Brice Denisse <from.test@example.com>')
        self.assertEqual(self.group_public.message_ids[0].author_id, from_1, 'message_process: email_from -> author_id wrong')
        self.group_public.message_unsubscribe([from_1.id])
        from_2 = self.env['res.users'].with_context({'no_reset_password': True}).create({'name': 'B', 'login': 'B', 'email': 'from.test@example.com'})
        self.format_and_process(MAIL_TEMPLATE, to='public@example.com', msg_id='<2>', email_from='Brice Denisse <from.test@example.com>')
        self.assertEqual(self.group_public.message_ids[0].author_id, from_2.partner_id, 'message_process: email_from -> author_id wrong')
        self.group_public.message_unsubscribe([from_2.partner_id.id])
        from_3 = self.env['res.partner'].create({'name': 'C', 'email': 'from.test@example.com'})
        self.group_public.message_subscribe([from_3.id])
        self.format_and_process(MAIL_TEMPLATE, to='public@example.com', msg_id='<3>', email_from='Brice Denisse <from.test@example.com>')
        self.assertEqual(self.group_public.message_ids[0].author_id, from_3, 'message_process: email_from -> author_id wrong')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_crash_wrong_model(self):
        if False:
            while True:
                i = 10
        ' Incoming email with model that does not accepts incoming emails must raise '
        self.assertRaises(ValueError, self.format_and_process, MAIL_TEMPLATE, to='noone@example.com', subject='spam', extra='', model='res.country', msg_id='<1198923581.41972151344608186760.JavaMail.new4@agrolait.com>')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_crash_no_data(self):
        if False:
            i = 10
            return i + 15
        ' Incoming email without model and without alias must raise '
        self.assertRaises(ValueError, self.format_and_process, MAIL_TEMPLATE, to='noone@example.com', subject='spam', extra='', msg_id='<1198923581.41972151344608186760.JavaMail.new5@agrolait.com>')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_fallback(self):
        if False:
            while True:
                i = 10
        ' Incoming email with model that accepting incoming emails as fallback '
        frog_groups = self.format_and_process(MAIL_TEMPLATE, to='noone@example.com', subject='Spammy', extra='', model='mail.channel', msg_id='<1198923581.41972151344608186760.JavaMail.new6@agrolait.com>')
        self.assertEqual(len(frog_groups), 1, 'message_process: erroneous email but with a fallback model should have created a new mail.channel')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models')
    def test_message_process_plain_text(self):
        if False:
            i = 10
            return i + 15
        ' Incoming email in plaintext should be stored as html '
        frog_groups = self.format_and_process(MAIL_TEMPLATE_PLAINTEXT, to='groups@example.com', subject='Frogs Return', extra='', msg_id='<deadcafe.1337@smtp.agrolait.com>')
        self.assertEqual(len(frog_groups), 1, 'message_process: a new mail.channel should have been created')
        msg = frog_groups.message_ids[0]
        self.assertIn('<pre>\nPlease call me as soon as possible this afternoon!\n<span data-o-mail-quote="1">\n--\nSylvie\n</span></pre>', msg.body, 'message_process: plaintext incoming email incorrectly parsed')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models', 'odoo.addons.mail.models.mail_mail')
    def test_private_discussion(self):
        if False:
            return 10
        ' Testing private discussion between partners. '
        msg1_pids = [self.env.user.partner_id.id, self.partner_1.id]
        msg1 = self.env['mail.thread'].with_context({'thread_model': 'mail.channel'}).sudo(self.user_employee).message_post(partner_ids=msg1_pids, subtype='mail.mt_comment')
        msg = self.env['mail.message'].browse(msg1.id)
        self.assertEqual(msg.partner_ids, self.env.user.partner_id | self.partner_1, 'message_post: private discussion: incorrect recipients')
        self.assertEqual(msg.model, False, 'message_post: private discussion: context key "thread_model" not correctly ignored when having no res_id')
        self.assertIn('openerp-private', msg.message_id, 'message_post: private discussion: message-id should contain the private keyword')
        self.format_and_process(MAIL_TEMPLATE, to='not_important@mydomain.com', email_from='valid.lelitre@agrolait.com', extra='In-Reply-To: %s' % msg.message_id, msg_id='<test30.JavaMail.0@agrolait.com>')
        msg2 = self.env['mail.message'].search([], limit=1)
        self.assertEqual(msg2.author_id, self.partner_1, 'message_post: private discussion: wrong author through mailgatewya based on email')
        self.assertEqual(msg2.partner_ids, self.user_employee.partner_id | self.env.user.partner_id, 'message_post: private discussion: incorrect recipients when replying')
        msg3 = self.env['mail.thread'].message_post(author_id=self.partner_1.id, parent_id=msg1.id, subtype='mail.mt_comment')
        msg = self.env['mail.message'].browse(msg3.id)
        self.assertEqual(msg.partner_ids, self.user_employee.partner_id | self.env.user.partner_id, 'message_post: private discussion: incorrect recipients when replying')
        self.assertEqual(msg.needaction_partner_ids, self.user_employee.partner_id | self.env.user.partner_id, 'message_post: private discussion: incorrect notified recipients when replying')

    @mute_logger('odoo.addons.mail.models.mail_thread', 'odoo.models', 'odoo.addons.mail.models.mail_mail')
    def test_forward_parent_id(self):
        if False:
            print('Hello World!')
        msg = self.group_pigs.sudo(self.user_employee).message_post(no_auto_thread=True, subtype='mail.mt_comment')
        self.assertNotIn(msg.model, msg.message_id)
        self.assertNotIn('-%d-' % msg.res_id, msg.message_id)
        self.assertIn('reply_to', msg.message_id)
        fw_msg_id = '<THIS.IS.A.FW.MESSAGE.1@bert.fr>'
        fw_message = MAIL_TEMPLATE.format(to='groups@example.com', cc='', subject='FW: Re: 1', email_from='b.t@example.com', extra='In-Reply-To: %s' % msg.message_id, msg_id=fw_msg_id)
        self.env['mail.thread'].message_process(None, fw_message)
        msg_fw = self.env['mail.message'].search([('message_id', '=', fw_msg_id)])
        self.assertEqual(len(msg_fw), 1)
        channel = self.env['mail.channel'].search([('name', '=', msg_fw.subject)])
        self.assertEqual(len(channel), 1)
        self.assertEqual(msg_fw.model, 'mail.channel')
        self.assertFalse(msg_fw.parent_id)
        self.assertTrue(msg_fw.res_id == channel.id)
        fw_msg_id = '<THIS.IS.A.FW.MESSAGE.2@bert.fr>'
        fw_message = MAIL_TEMPLATE.format(to='public@example.com', cc='', subject='FW: Re: 2', email_from='b.t@example.com', extra='In-Reply-To: %s' % msg.message_id, msg_id=fw_msg_id)
        self.env['mail.thread'].message_process(None, fw_message)
        msg_fw = self.env['mail.message'].search([('message_id', '=', fw_msg_id)])
        self.assertEqual(len(msg_fw), 1)
        channel = self.env['mail.channel'].search([('name', '=', msg_fw.subject)])
        self.assertEqual(len(channel), 0)
        self.assertEqual(msg_fw.model, 'mail.channel')
        self.assertFalse(msg_fw.parent_id)
        self.assertTrue(msg_fw.res_id == self.group_public.id)