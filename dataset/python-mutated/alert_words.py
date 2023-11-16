from typing import Dict, Iterable, List
import ahocorasick
from django.db import transaction
from zerver.lib.cache import cache_with_key, realm_alert_words_automaton_cache_key, realm_alert_words_cache_key
from zerver.models import AlertWord, Realm, UserProfile, flush_realm_alert_words

@cache_with_key(lambda realm: realm_alert_words_cache_key(realm.id), timeout=3600 * 24)
def alert_words_in_realm(realm: Realm) -> Dict[int, List[str]]:
    if False:
        for i in range(10):
            print('nop')
    user_ids_and_words = AlertWord.objects.filter(realm=realm, user_profile__is_active=True).values('user_profile_id', 'word')
    user_ids_with_words: Dict[int, List[str]] = {}
    for id_and_word in user_ids_and_words:
        user_ids_with_words.setdefault(id_and_word['user_profile_id'], [])
        user_ids_with_words[id_and_word['user_profile_id']].append(id_and_word['word'])
    return user_ids_with_words

@cache_with_key(lambda realm: realm_alert_words_automaton_cache_key(realm.id), timeout=3600 * 24)
def get_alert_word_automaton(realm: Realm) -> ahocorasick.Automaton:
    if False:
        print('Hello World!')
    user_id_with_words = alert_words_in_realm(realm)
    alert_word_automaton = ahocorasick.Automaton()
    for (user_id, alert_words) in user_id_with_words.items():
        for alert_word in alert_words:
            alert_word_lower = alert_word.lower()
            if alert_word_automaton.exists(alert_word_lower):
                (key, user_ids_for_alert_word) = alert_word_automaton.get(alert_word_lower)
                user_ids_for_alert_word.add(user_id)
            else:
                alert_word_automaton.add_word(alert_word_lower, (alert_word_lower, {user_id}))
    alert_word_automaton.make_automaton()
    if alert_word_automaton.kind != ahocorasick.AHOCORASICK:
        return None
    return alert_word_automaton

def user_alert_words(user_profile: UserProfile) -> List[str]:
    if False:
        while True:
            i = 10
    return list(AlertWord.objects.filter(user_profile=user_profile).values_list('word', flat=True))

@transaction.atomic
def add_user_alert_words(user_profile: UserProfile, new_words: Iterable[str]) -> List[str]:
    if False:
        while True:
            i = 10
    existing_words_lower = {word.lower() for word in user_alert_words(user_profile)}
    word_dict: Dict[str, str] = {}
    for word in new_words:
        if word.lower() in existing_words_lower:
            continue
        word_dict[word.lower()] = word
    AlertWord.objects.bulk_create((AlertWord(user_profile=user_profile, word=word, realm=user_profile.realm) for word in word_dict.values()))
    flush_realm_alert_words(user_profile.realm_id)
    return user_alert_words(user_profile)

@transaction.atomic
def remove_user_alert_words(user_profile: UserProfile, delete_words: Iterable[str]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    for delete_word in delete_words:
        AlertWord.objects.filter(user_profile=user_profile, word__iexact=delete_word).delete()
    return user_alert_words(user_profile)