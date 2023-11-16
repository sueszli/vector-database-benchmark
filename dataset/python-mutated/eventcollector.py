from cStringIO import StringIO
import datetime
import gzip
import hashlib
import hmac
import itertools
import json
import pytz
import random
import requests
import time
import httpagentparser
import time
from pylons import app_globals as g
from uuid import uuid4
from wsgiref.handlers import format_date_time
from r2.lib import amqp, hooks
from r2.lib.language import charset_summary
from r2.lib.geoip import get_request_location, location_by_ips
from r2.lib.cache_poisoning import cache_headers_valid
from r2.lib.utils import domain, to_epoch_milliseconds, sampled, squelch_exceptions, to36

def _make_http_date(when=None):
    if False:
        return 10
    if when is None:
        when = datetime.datetime.now(pytz.UTC)
    return format_date_time(time.mktime(when.timetuple()))
_datetime_to_millis = to_epoch_milliseconds

def parse_agent(ua):
    if False:
        print('Hello World!')
    agent_summary = {}
    parsed = httpagentparser.detect(ua)
    for attr in ('browser', 'os', 'platform'):
        d = parsed.get(attr)
        if d:
            for subattr in ('name', 'version'):
                if subattr in d:
                    key = '%s_%s' % (attr, subattr)
                    agent_summary[key] = d[subattr]
    agent_summary['bot'] = parsed.get('bot')
    return agent_summary

class EventQueue(object):

    def __init__(self, queue=amqp):
        if False:
            return 10
        self.queue = queue

    def save_event(self, event):
        if False:
            i = 10
            return i + 15
        if event.testing:
            queue_name = 'event_collector_test'
        else:
            queue_name = 'event_collector'
        headers = None
        if event.truncatable_field:
            headers = {'truncatable_field': event.truncatable_field}
        self.queue.add_item(queue_name, event.dump(), headers=headers)

    @squelch_exceptions
    @sampled('events_collector_vote_sample_rate')
    def vote_event(self, vote):
        if False:
            print('Hello World!')
        "Create a 'vote' event for event-collector\n\n        vote: An r2.models.vote Vote object\n        "

        def get_vote_direction_name(vote):
            if False:
                print('Hello World!')
            if vote.is_upvote:
                return 'up'
            elif vote.is_downvote:
                return 'down'
            else:
                return 'clear'
        event = Event(topic='vote_server', event_type='server_vote', time=vote.date, data=vote.event_data['context'], obfuscated_data=vote.event_data['sensitive'])
        event.add('vote_direction', get_vote_direction_name(vote))
        if vote.previous_vote:
            event.add('prev_vote_direction', get_vote_direction_name(vote.previous_vote))
            event.add('prev_vote_ts', to_epoch_milliseconds(vote.previous_vote.date))
        if vote.is_automatic_initial_vote:
            event.add('auto_self_vote', True)
        for (name, value) in vote.effects.serializable_data.iteritems():
            if name == 'notes':
                name = 'details_text'
            event.add(name, value)
        event.add('process_notes', ', '.join(vote.effects.note_codes))
        event.add_subreddit_fields(vote.thing.subreddit_slow)
        event.add_target_fields(vote.thing)
        rank = vote.data.get('rank')
        if rank:
            event.add('target_rank', rank)
        self.save_event(event)

    @squelch_exceptions
    @sampled('events_collector_submit_sample_rate')
    def submit_event(self, new_post, request=None, context=None):
        if False:
            for i in range(10):
                print('nop')
        "Create a 'submit' event for event-collector\n\n        new_post: An r2.models.Link object\n        request, context: Should be pylons.request & pylons.c respectively\n\n        "
        event = Event(topic='submit_events', event_type='ss.submit', time=new_post._date, request=request, context=context, truncatable_field='post_body')
        event.add('post_id', new_post._id)
        event.add('post_fullname', new_post._fullname)
        event.add_text('post_title', new_post.title)
        event.add('user_neutered', new_post.author_slow._spam)
        if new_post.is_self:
            event.add('post_type', 'self')
            event.add_text('post_body', new_post.selftext)
        else:
            event.add('post_type', 'link')
            event.add('post_target_url', new_post.url)
            event.add('post_target_domain', new_post.link_domain())
        event.add_subreddit_fields(new_post.subreddit_slow)
        self.save_event(event)

    @squelch_exceptions
    @sampled('events_collector_comment_sample_rate')
    def comment_event(self, new_comment, request=None, context=None):
        if False:
            print('Hello World!')
        "Create a 'comment' event for event-collector.\n\n        new_comment: An r2.models.Comment object\n        request, context: Should be pylons.request & pylons.c respectively\n        "
        from r2.models import Comment, Link
        event = Event(topic='comment_events', event_type='ss.comment', time=new_comment._date, request=request, context=context, truncatable_field='comment_body')
        event.add('comment_id', new_comment._id)
        event.add('comment_fullname', new_comment._fullname)
        event.add_text('comment_body', new_comment.body)
        post = Link._byID(new_comment.link_id)
        event.add('post_id', post._id)
        event.add('post_fullname', post._fullname)
        event.add('post_created_ts', to_epoch_milliseconds(post._date))
        if post.promoted:
            event.add('post_is_promoted', bool(post.promoted))
        if new_comment.parent_id:
            parent = Comment._byID(new_comment.parent_id)
        else:
            parent = post
        event.add('parent_id', parent._id)
        event.add('parent_fullname', parent._fullname)
        event.add('parent_created_ts', to_epoch_milliseconds(parent._date))
        event.add('user_neutered', new_comment.author_slow._spam)
        event.add_subreddit_fields(new_comment.subreddit_slow)
        self.save_event(event)

    @squelch_exceptions
    @sampled('events_collector_poison_sample_rate')
    def cache_poisoning_event(self, poison_info, request=None, context=None):
        if False:
            i = 10
            return i + 15
        "Create a 'cache_poisoning_server' event for event-collector\n\n        poison_info: Details from the client about the poisoning event\n        request, context: Should be pylons.request & pylons.c respectively\n\n        "
        poisoner_name = poison_info.pop('poisoner_name')
        event = Event(topic='cache_poisoning_events', event_type='ss.cache_poisoning', request=request, context=context, data=poison_info, truncatable_field='resp_headers')
        event.add('poison_blame_guess', 'proxy')
        resp_headers = poison_info['resp_headers']
        if resp_headers:
            cache_policy = poison_info['cache_policy']
            headers_valid = cache_headers_valid(cache_policy, resp_headers)
            event.add('cache_headers_valid', headers_valid)
        if poison_info['source'] == 'web':
            valid_login_hook = hooks.get_hook('poisoning.guess_valid_login')
            if valid_login_hook.call_until_return(poisoner_name=poisoner_name):
                event.add('poison_blame_guess', 'local_proxy')
                event.add('poison_credentialed_guess', False)
            elif context.user_is_loggedin and context.user.name == poisoner_name:
                event.add('poison_credentialed_guess', True)
            else:
                event.add('poison_credentialed_guess', False)
        elif poison_info['source'] == 'mweb':
            event.add('poison_credentialed_guess', True)
        else:
            raise Exception('Unsupported source in cache_poisoning_event')
        if resp_headers and 'cf-cache-status' in resp_headers:
            event.add('poison_blame_guess', 'cdn')
        self.save_event(event)

    @squelch_exceptions
    def muted_forbidden_event(self, details_text, subreddit=None, parent_message=None, target=None, request=None, context=None):
        if False:
            i = 10
            return i + 15
        'Create a mute-related \'forbidden_event\' for event-collector.\n\n        details_text: "muted" if a muted user is trying to message the\n            subreddit or "muted mod" if the subreddit mod is attempting\n            to message the muted user\n        subreddit: The Subreddit of the mod messaging the muted user\n        parent_message: Message that is being responded to\n        target: The intended recipient (Subreddit or Account)\n        request, context: Should be pylons.request & pylons.c respectively;\n\n        '
        event = Event(topic='forbidden_actions', event_type='ss.forbidden_message_attempt', request=request, context=context)
        event.add('details_text', details_text)
        if parent_message:
            event.add('parent_message_id', parent_message._id)
            event.add('parent_message_fullname', parent_message._fullname)
        event.add_subreddit_fields(subreddit)
        event.add_target_fields(target)
        self.save_event(event)

    @squelch_exceptions
    def timeout_forbidden_event(self, action_name, details_text, target=None, target_fullname=None, subreddit=None, request=None, context=None):
        if False:
            for i in range(10):
                print('nop')
        "Create a timeout-related 'forbidden_actions' for event-collector.\n\n        action_name: the action taken by a user in timeout\n        details_text: this provides more details about the action\n        target: The intended item the action was to be taken on\n        target_fullname: The fullname used to convert to a target\n        subreddit: The Subreddit the action was taken in. If target is of the\n            type Subreddit, then this won't be passed in\n        request, context: Should be pylons.request & pylons.c respectively;\n\n        "
        if not action_name:
            request_vars = request.environ['pylons.routes_dict']
            action_name = request_vars.get('action_name')
            if action_name == 'vote':
                direction = int(request.POST.get('dir', 0))
                if direction == 1:
                    action_name = 'upvote'
                elif direction == -1:
                    action_name = 'downvote'
                else:
                    action_name = 'clearvote'
            elif action_name in ('set_contest_mode', 'set_subreddit_sticky'):
                action_name = action_name.replace('_', '')
                if request.POST.get('state') == 'False':
                    action_name = 'un' + action_name
            elif action_name == 'set_suggested_sort':
                action_name = action_name.replace('_', '')
                if request.POST.get('sort') in ('', 'clear'):
                    action_name = 'un' + action_name
            elif action_name == 'spamlisting':
                action_name = 'pageview'
                details_text = request_vars.get('location')
            elif action_name == 'clearflairtemplates':
                action_name = 'editflair'
                details_text = 'flair_clear_template'
            elif action_name in ('flairconfig', 'flaircsv', 'flairlisting'):
                details_text = action_name.replace('flair', 'flair_')
                action_name = 'editflair'
        if not target:
            if not target_fullname:
                if action_name in ('wiki_settings', 'wiki_edit'):
                    target = context.site
                elif action_name in 'wiki_allow_editor':
                    target = Account._by_name(request.POST.get('username'))
                elif action_name in ('delete_sr_header', 'delete_sr_icon', 'delete_sr_banner'):
                    details_text = '%s' % action_name.replace('ete_sr', '')
                    action_name = 'editsettings'
                    target = context.site
                elif action_name in ('bannedlisting', 'mutedlisting', 'wikibannedlisting', 'wikicontributorslisting'):
                    target = context.site
            if target_fullname:
                from r2.models import Thing
                target = Thing._by_fullname(target_fullname, return_dict=False, data=True)
        event = Event(topic='forbidden_actions', event_type='ss.forbidden_%s' % action_name, request=request, context=context)
        event.add('details_text', details_text)
        event.add('process_notes', 'IN_TIMEOUT')
        from r2.models import Comment, Link, Subreddit
        if not subreddit:
            if isinstance(context.site, Subreddit):
                subreddit = context.site
            elif isinstance(target, (Comment, Link)):
                subreddit = target.subreddit_slow
            elif isinstance(target, Subreddit):
                subreddit = target
        event.add_subreddit_fields(subreddit)
        event.add_target_fields(target)
        self.save_event(event)

    @squelch_exceptions
    @sampled('events_collector_mod_sample_rate')
    def mod_event(self, modaction, subreddit, mod, target=None, request=None, context=None):
        if False:
            for i in range(10):
                print('nop')
        "Create a 'mod' event for event-collector.\n\n        modaction: An r2.models.ModAction object\n        subreddit: The Subreddit the mod action is being performed in\n        mod: The Account that is performing the mod action\n        target: The Thing the mod action was applied to\n        request, context: Should be pylons.request & pylons.c respectively\n\n        "
        event = Event(topic='mod_events', event_type=modaction.action, time=modaction.date, uuid=modaction._id, request=request, context=context)
        event.add('details_text', modaction.details_text)
        if not event.get('user_id'):
            event['user_id'] = mod._id
            event['user_name'] = mod.name
        event.add_subreddit_fields(subreddit)
        event.add_target_fields(target)
        self.save_event(event)

    @squelch_exceptions
    @sampled('events_collector_report_sample_rate')
    def report_event(self, reason=None, details_text=None, subreddit=None, target=None, request=None, context=None, event_type='ss.report'):
        if False:
            while True:
                i = 10
        "Create a 'report' event for event-collector.\n\n        process_notes: Type of rule (pre-defined report reasons or custom)\n        details_text: The report reason\n        subreddit: The Subreddit the action is being performed in\n        target: The Thing the action was applied to\n        request, context: Should be pylons.request & pylons.c respectively\n\n        "
        from r2.models.rules import OLD_SITEWIDE_RULES, SITEWIDE_RULES, SubredditRules
        event = Event(topic='report_events', event_type=event_type, request=request, context=context)
        if reason in OLD_SITEWIDE_RULES or reason in SITEWIDE_RULES:
            process_notes = 'SITE_RULES'
        elif subreddit and SubredditRules.get_rule(subreddit, reason):
            process_notes = 'SUBREDDIT_RULES'
        else:
            process_notes = 'CUSTOM'
        event.add('process_notes', process_notes)
        event.add('details_text', details_text)
        event.add_subreddit_fields(subreddit)
        event.add_target_fields(target)
        self.save_event(event)

    @squelch_exceptions
    @sampled('events_collector_quarantine_sample_rate')
    def quarantine_event(self, event_type, subreddit, request=None, context=None):
        if False:
            i = 10
            return i + 15
        "Create a 'quarantine' event for event-collector.\n\n        event_type: quarantine_interstitial_view, quarantine_opt_in,\n            quarantine_opt_out, quarantine_interstitial_dismiss\n        subreddit: The quarantined subreddit\n        request, context: Should be pylons.request & pylons.c respectively\n\n        "
        event = Event(topic='quarantine', event_type=event_type, request=request, context=context)
        if context:
            if context.user_is_loggedin:
                event.add('verified_email', context.user.email_verified)
            else:
                event.add('verified_email', False)
        if request and (event_type == 'quarantine_interstitial_view' or event_type == 'quarantine_opt_out'):
            request_vars = request.environ['pylons.routes_dict']
            event.add('sr_action', request_vars.get('action', None))
            if request.environ['pylons.routes_dict'].get('comment', None):
                thing_id36 = request_vars.get('comment', None)
            else:
                thing_id36 = request_vars.get('article', None)
            if thing_id36:
                event.add('thing_id', int(thing_id36, 36))
        event.add_subreddit_fields(subreddit)
        self.save_event(event)

    @squelch_exceptions
    @sampled('events_collector_modmail_sample_rate')
    def modmail_event(self, message, request=None, context=None):
        if False:
            i = 10
            return i + 15
        "Create a 'modmail' event for event-collector.\n\n        message: An r2.models.Message object\n        request: pylons.request of the request that created the message\n        context: pylons.tmpl_context of the request that created the message\n\n        "
        from r2.models import Account, Message
        sender = message.author_slow
        sr = message.subreddit_slow
        sender_is_moderator = sr.is_moderator_with_perms(sender, 'mail')
        if message.first_message:
            first_message = Message._byID(message.first_message, data=True)
        else:
            first_message = message
        event = Event(topic='message_events', event_type='ss.send_message', time=message._date, request=request, context=context, data={'user_id': sender._id, 'user_name': sender.name})
        if sender == Account.system_user():
            sender_type = 'automated'
        elif sender_is_moderator:
            sender_type = 'moderator'
        else:
            sender_type = 'user'
        event.add('sender_type', sender_type)
        event.add('sr_id', sr._id)
        event.add('sr_name', sr.name)
        event.add('message_id', message._id)
        event.add('message_kind', 'modmail')
        event.add('message_fullname', message._fullname)
        event.add_text('message_body', message.body)
        event.add_text('message_subject', message.subject)
        event.add('first_message_id', first_message._id)
        event.add('first_message_fullname', first_message._fullname)
        if request and request.POST.get('source', None):
            source = request.POST['source']
            if source in {'compose', 'permalink', 'modmail', 'usermail'}:
                event.add('page', source)
        if message.sent_via_email:
            event.add('is_third_party', True)
            event.add('third_party_metadata', 'mailgun')
        if not message.to_id:
            target = sr
        else:
            target = Account._byID(message.to_id, data=True)
        event.add_target_fields(target)
        self.save_event(event)

    @squelch_exceptions
    @sampled('events_collector_message_sample_rate')
    def message_event(self, message, event_type='ss.send_message', request=None, context=None):
        if False:
            for i in range(10):
                print('nop')
        "Create a 'message' event for event-collector.\n\n        message: An r2.models.Message object\n        request: pylons.request of the request that created the message\n        context: pylons.tmpl_context of the request that created the message\n\n        "
        from r2.models import Account, Message
        sender = message.author_slow
        if message.first_message:
            first_message = Message._byID(message.first_message, data=True)
        else:
            first_message = message
        event = Event(topic='message_events', event_type=event_type, time=message._date, request=request, context=context, data={'user_id': sender._id, 'user_name': sender.name})
        if sender == Account.system_user():
            sender_type = 'automated'
        else:
            sender_type = 'user'
        event.add('sender_type', sender_type)
        event.add('message_kind', 'message')
        event.add('message_id', message._id)
        event.add('message_fullname', message._fullname)
        event.add_text('message_body', message.body)
        event.add_text('message_subject', message.subject)
        event.add('first_message_id', first_message._id)
        event.add('first_message_fullname', first_message._fullname)
        if request and request.POST.get('source', None):
            source = request.POST['source']
            if source in {'compose', 'permalink', 'usermail'}:
                event.add('page', source)
        if message.sent_via_email:
            event.add('is_third_party', True)
            event.add('third_party_metadata', 'mailgun')
        target = Account._byID(message.to_id, data=True)
        event.add_target_fields(target)
        self.save_event(event)

    def loid_event(self, loid, action_name, request=None, context=None):
        if False:
            print('Hello World!')
        "Create a 'loid' event for event-collector.\n\n        loid: the created/modified loid\n        action_name: create_loid (only allowed value currently)\n        "
        event = Event(topic='loid_events', event_type='ss.%s' % action_name, request=request, context=context)
        event.add('request_url', request.fullpath)
        for (k, v) in loid.to_dict().iteritems():
            event.add(k, v)
        self.save_event(event)

    def login_event(self, action_name, error_msg, user_name=None, email=None, remember_me=None, newsletter=None, email_verified=None, signature=None, request=None, context=None):
        if False:
            for i in range(10):
                print('nop')
        "Create a 'login' event for event-collector.\n\n        action_name: login_attempt, register_attempt, password_reset\n        error_msg: error message string if there was an error\n        user_name: user entered username string\n        email: user entered email string (register, password reset)\n        remember_me:  boolean state of remember me checkbox (login, register)\n        newsletter: boolean state of newsletter checkbox (register only)\n        email_verified: boolean value for email verification state, requires\n            email (password reset only)\n        request, context: Should be pylons.request & pylons.c respectively\n\n        "
        event = Event(topic='login_events', event_type='ss.%s' % action_name, request=request, context=context)
        if error_msg:
            event.add('successful', False)
            event.add('process_notes', error_msg)
        else:
            event.add('successful', True)
        event.add('user_name', user_name)
        event.add('email', email)
        event.add('remember_me', remember_me)
        event.add('newsletter', newsletter)
        event.add('email_verified', email_verified)
        if signature:
            event.add('signed', True)
            event.add('signature_platform', signature.platform)
            event.add('signature_version', signature.version)
            event.add('signature_valid', signature.is_valid())
            sigerror = ', '.join(('%s_%s' % (field, code) for (code, field) in signature.errors))
            event.add('signature_errors', sigerror)
            if signature.epoch:
                event.add('signature_age', int(time.time()) - signature.epoch)
        self.save_event(event)

    def bucketing_event(self, experiment_id, experiment_name, variant, user, loid):
        if False:
            while True:
                i = 10
        'Send an event recording an experiment bucketing.\n\n        experiment_id: an integer representing the experiment\n        experiment_name: a human-readable name representing the experiment\n        variant: a string representing the variant name\n        user: the Account that has been put into the variant\n        '
        event = Event(topic='bucketing_events', event_type='bucket')
        event.add('experiment_id', experiment_id)
        event.add('experiment_name', experiment_name)
        event.add('variant', variant)
        if user is not None:
            event.add('user_id', user._id)
            event.add('user_name', user.name)
        if loid:
            for (k, v) in loid.to_dict().iteritems():
                event.add(k, v)
        self.save_event(event)

    def page_bucketing_event(self, experiment_id, experiment_name, variant, content_id, request, context=None):
        if False:
            i = 10
            return i + 15
        'Send an event recording bucketing of a page for a page-based\n        experiment.\n\n        experiment_id: an integer representing the experiment\n        experiment_name: a human-readable name representing the experiment\n        variant: a string representing the variant name\n        content_id: the primary content fullname for the page being bucketed\n        '
        event = Event(topic='bucketing_events', event_type='bucket_page', request=request, context=context)
        event.add('experiment_id', experiment_id)
        event.add('experiment_name', experiment_name)
        event.add('variant', variant)
        event.add('bucketing_fullname', content_id)
        event.add('crawler_name', g.pool_name)
        event.add('url', request.fullurl)
        self.save_event(event)

class Event(object):

    def __init__(self, topic, event_type, time=None, uuid=None, request=None, context=None, testing=False, data=None, obfuscated_data=None, truncatable_field=None):
        if False:
            print('Hello World!')
        'Create a new event for event-collector.\n\n        topic: Used to filter events into appropriate streams for processing\n        event_type: Used for grouping and sub-categorizing events\n        time: Should be a datetime.datetime object in UTC timezone\n        uuid: Should be a UUID object\n        request, context: Should be pylons.request & pylons.c respectively\n        testing: Whether to send the event to the test endpoint\n        data: A dict of field names/values to initialize the payload with\n        obfuscated_data: Same as `data`, but fields that need obfuscation\n        truncatable_field: Field to truncate if the event is too large\n        '
        self.topic = topic
        self.event_type = event_type
        self.testing = testing or g.debug
        self.truncatable_field = truncatable_field
        if not time:
            time = datetime.datetime.now(pytz.UTC)
        self.timestamp = _datetime_to_millis(time)
        if not uuid:
            uuid = uuid4()
        self.uuid = str(uuid)
        self.payload = {}
        if data:
            self.payload.update(data)
        self.obfuscated_data = {}
        if obfuscated_data:
            self.obfuscated_data.update(obfuscated_data)
        if context and request:
            context_data = self.get_context_data(request, context)
            new_context_data = {k: v for (k, v) in context_data.items() if k not in self.payload}
            self.payload.update(new_context_data)
            context_data = self.get_sensitive_context_data(request, context)
            new_context_data = {k: v for (k, v) in context_data.items() if k not in self.obfuscated_data}
            self.obfuscated_data.update(new_context_data)

    def add(self, field, value, obfuscate=False):
        if False:
            print('Hello World!')
        if value is None or value == '':
            return
        if obfuscate:
            self.obfuscated_data[field] = value
        else:
            self.payload[field] = value

    def add_text(self, key, value, obfuscate=False):
        if False:
            return 10
        self.add(key, value, obfuscate=obfuscate)
        for (k, v) in charset_summary(value).iteritems():
            self.add('{}_{}'.format(key, k), v)

    def add_target_fields(self, target):
        if False:
            for i in range(10):
                print('nop')
        if not target:
            return
        from r2.models import Comment, Link, Message
        self.add('target_id', target._id)
        self.add('target_fullname', target._fullname)
        self.add('target_age_seconds', target._age.total_seconds())
        target_type = target.__class__.__name__.lower()
        if target_type == 'link' and target.is_self:
            target_type = 'self'
        self.add('target_type', target_type)
        if hasattr(target, 'name'):
            self.add('target_name', target.name)
        if isinstance(target, (Comment, Link, Message)):
            author = target.author_slow
            if target._deleted or author._deleted:
                self.add('target_author_id', 0)
                self.add('target_author_name', '[deleted]')
            else:
                self.add('target_author_id', author._id)
                self.add('target_author_name', author.name)
        if isinstance(target, Link):
            self.add_text('target_title', target.title)
            if not target.is_self:
                self.add('target_url', target.url)
                self.add('target_url_domain', target.link_domain())
        if isinstance(target, Comment):
            link_fullname = Link._fullname_from_id36(to36(target.link_id))
            self.add('link_id', target.link_id)
            self.add('link_fullname', link_fullname)
        if isinstance(target, (Comment, Link)):
            self.add('target_created_ts', to_epoch_milliseconds(target._date))
        hooks.get_hook('eventcollector.add_target_fields').call(event=self, target=target)

    def add_subreddit_fields(self, subreddit):
        if False:
            i = 10
            return i + 15
        if not subreddit:
            return
        self.add('sr_id', subreddit._id)
        self.add('sr_name', subreddit.name)

    def get(self, field, obfuscated=False):
        if False:
            while True:
                i = 10
        if obfuscated:
            return self.obfuscated_data.get(field, None)
        else:
            return self.payload.get(field, None)

    @classmethod
    def get_context_data(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        'Extract common data from the current request and context\n\n        This is generally done explicitly in `__init__`, but is done by hand for\n        votes before the request context is lost by the queuing.\n\n        request, context: Should be pylons.request & pylons.c respectively\n        '
        data = {}
        if context.user_is_loggedin:
            data['user_id'] = context.user._id
            data['user_name'] = context.user.name
        elif context.loid:
            data.update(context.loid.to_dict())
        oauth2_client = getattr(context, 'oauth2_client', None)
        if oauth2_client:
            data['oauth2_client_id'] = oauth2_client._id
            data['oauth2_client_name'] = oauth2_client.name
            data['oauth2_client_app_type'] = oauth2_client.app_type
        data['geoip_country'] = get_request_location(request, context)
        data['domain'] = request.host
        data['user_agent'] = request.user_agent
        data['user_agent_parsed'] = request.parsed_agent.to_dict()
        http_referrer = request.headers.get('Referer', None)
        if http_referrer:
            data['referrer_url'] = http_referrer
            data['referrer_domain'] = domain(http_referrer)
        hooks.get_hook('eventcollector.context_data').call(data=data, user=context.user, request=request, context=context)
        return data

    @classmethod
    def get_sensitive_context_data(self, request, context):
        if False:
            return 10
        data = {}
        ip = getattr(request, 'ip', None)
        if ip:
            data['client_ip'] = ip
            if '.' in ip:
                octets = ip.split('.')
                data['client_ipv4_24'] = '.'.join(octets[:3])
                data['client_ipv4_16'] = '.'.join(octets[:2])
        return data

    def dump(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the JSON representation of the event.'
        data = {'event_topic': self.topic, 'event_type': self.event_type, 'event_ts': self.timestamp, 'uuid': self.uuid, 'payload': self.payload}
        if self.obfuscated_data:
            data['payload']['obfuscated_data'] = self.obfuscated_data
        return json.dumps(data)

class PublishableEvent(object):

    def __init__(self, data, truncatable_field=None):
        if False:
            print('Hello World!')
        self.data = data
        self.truncatable_field = truncatable_field

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.data)

    def truncate_data(self, target_len):
        if False:
            for i in range(10):
                print('nop')
        if not self.truncatable_field:
            return
        if len(self.data) <= target_len:
            return
        oversize_by = len(self.data) - target_len
        oversize_by += len('"is_truncated": true, ')
        deserialized_data = json.loads(self.data)
        original = deserialized_data['payload'][self.truncatable_field]
        truncated = original[:-oversize_by]
        deserialized_data['payload'][self.truncatable_field] = truncated
        deserialized_data['payload']['is_truncated'] = True
        self.data = json.dumps(deserialized_data)
        g.stats.simple_event('eventcollector.oversize_truncated')

class EventPublisher(object):
    MAX_EVENT_SIZE = 100 * 1024
    MAX_CONTENT_LENGTH = 500 * 1024

    def __init__(self, url, signature_key, secret, user_agent, stats, max_event_size=MAX_EVENT_SIZE, max_content_length=MAX_CONTENT_LENGTH, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        self.url = url
        self.signature_key = signature_key
        self.secret = secret
        self.user_agent = user_agent
        self.timeout = timeout
        self.stats = stats
        self.max_event_size = max_event_size
        self.max_content_length = max_content_length
        self.session = requests.Session()

    def _make_signature(self, payload):
        if False:
            return 10
        mac = hmac.new(self.secret, payload, hashlib.sha256).hexdigest()
        return 'key={key}, mac={mac}'.format(key=self.signature_key, mac=mac)

    def _publish(self, events):
        if False:
            while True:
                i = 10
        data = '[' + ', '.join(events) + ']'
        headers = {'Date': _make_http_date(), 'User-Agent': self.user_agent, 'Content-Type': 'application/json', 'X-Signature': self._make_signature(data)}
        use_gzip = g.live_config.get('events_collector_use_gzip_chance', 0) > random.random()
        if use_gzip:
            f = StringIO()
            gzip.GzipFile(fileobj=f, mode='wb').write(data)
            data = f.getvalue()
            headers['Content-Encoding'] = 'gzip'
        with self.stats.get_timer('providers.event_collector'):
            resp = self.session.post(self.url, data=data, headers=headers, timeout=self.timeout)
            return resp

    def _chunk_events(self, events):
        if False:
            return 10
        'Break a PublishableEvent list into chunks to obey size limits.\n\n        Note that this yields lists of strings (the serialized data) to\n        publish directly, not PublishableEvent objects.\n\n        '
        to_send = []
        send_size = 0
        for event in events:
            event.truncate_data(self.max_event_size)
            if len(event) > self.max_event_size:
                g.log.warning('Event too large (%s); dropping', len(event))
                g.log.warning('%r', event.data)
                g.stats.simple_event('eventcollector.oversize_dropped')
                continue
            send_size += len(event)
            if len(to_send) > 0:
                send_size += len(', ')
            if send_size + 2 >= self.max_content_length:
                yield to_send
                to_send = []
                send_size = len(event)
            to_send.append(event.data)
        if to_send:
            yield to_send

    def publish(self, events):
        if False:
            i = 10
            return i + 15
        for some_events in self._chunk_events(events):
            resp = self._publish(some_events)
            ignored = resp.content
            yield (resp, some_events)

def _get_reason(response):
    if False:
        i = 10
        return i + 15
    return getattr(response, 'reason', None) or getattr(response.raw, 'reason', '{unknown}')

def process_events(g, timeout=5.0, **kw):
    if False:
        for i in range(10):
            print('nop')
    publisher = EventPublisher(g.events_collector_url, g.secrets['events_collector_key'], g.secrets['events_collector_secret'], g.useragent, g.stats, timeout=timeout)
    test_publisher = EventPublisher(g.events_collector_test_url, g.secrets['events_collector_key'], g.secrets['events_collector_secret'], g.useragent, g.stats, timeout=timeout)

    @g.stats.amqp_processor('event_collector')
    def processor(msgs, chan):
        if False:
            print('Hello World!')
        events = []
        test_events = []
        for msg in msgs:
            headers = msg.properties.get('application_headers', {})
            truncatable_field = headers.get('truncatable_field')
            event = PublishableEvent(msg.body, truncatable_field)
            if msg.delivery_info['routing_key'] == 'event_collector_test':
                test_events.append(event)
            else:
                events.append(event)
        to_publish = itertools.chain(publisher.publish(events), test_publisher.publish(test_events))
        for (response, sent) in to_publish:
            if response.ok:
                g.log.info('Published %s events', len(sent))
            else:
                g.log.warning('Event send failed %s - %s', response.status_code, _get_reason(response))
                g.log.warning('Response headers: %r', response.headers)
                if response.status_code == 413:
                    for event in sent:
                        amqp.add_item('event_collector_failed', event)
                else:
                    response.raise_for_status()
    amqp.handle_items('event_collector', processor, **kw)