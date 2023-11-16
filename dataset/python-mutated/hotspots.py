from typing import Dict, List
from django.conf import settings
from django.utils.translation import gettext_lazy
from django_stubs_ext import StrPromise
from zerver.models import UserHotspot, UserProfile
INTRO_HOTSPOTS: Dict[str, Dict[str, StrPromise]] = {'intro_streams': {'title': gettext_lazy('Catch up on a stream'), 'description': gettext_lazy('Messages sent to a stream are seen by everyone subscribed to that stream. Try clicking on one of the stream links below.')}, 'intro_topics': {'title': gettext_lazy('Topics'), 'description': gettext_lazy('Every message has a topic. Topics keep conversations easy to follow, and make it easy to reply to conversations that start while you are offline.')}, 'intro_gear': {'title': gettext_lazy('Settings'), 'description': gettext_lazy('Go to Settings to configure your notifications and preferences.')}, 'intro_compose': {'title': gettext_lazy('Compose'), 'description': gettext_lazy('Click here to start a new conversation. Pick a topic (2-3 words is best), and give it a go!')}}
ALL_HOTSPOTS: Dict[str, Dict[str, StrPromise]] = {**INTRO_HOTSPOTS}

def get_next_hotspots(user: UserProfile) -> List[Dict[str, object]]:
    if False:
        for i in range(10):
            print('nop')
    if settings.ALWAYS_SEND_ALL_HOTSPOTS:
        return [{'name': hotspot, 'title': str(ALL_HOTSPOTS[hotspot]['title']), 'description': str(ALL_HOTSPOTS[hotspot]['description']), 'delay': 0} for hotspot in ALL_HOTSPOTS]
    if not settings.TUTORIAL_ENABLED:
        return []
    if user.tutorial_status == UserProfile.TUTORIAL_FINISHED:
        return []
    seen_hotspots = frozenset(UserHotspot.objects.filter(user=user).values_list('hotspot', flat=True))
    for hotspot in INTRO_HOTSPOTS:
        if hotspot not in seen_hotspots:
            return [{'name': hotspot, 'title': str(INTRO_HOTSPOTS[hotspot]['title']), 'description': str(INTRO_HOTSPOTS[hotspot]['description']), 'delay': 0.5}]
    user.tutorial_status = UserProfile.TUTORIAL_FINISHED
    user.save(update_fields=['tutorial_status'])
    return []

def copy_hotspots(source_profile: UserProfile, target_profile: UserProfile) -> None:
    if False:
        for i in range(10):
            print('nop')
    for userhotspot in frozenset(UserHotspot.objects.filter(user=source_profile)):
        UserHotspot.objects.create(user=target_profile, hotspot=userhotspot.hotspot, timestamp=userhotspot.timestamp)
    target_profile.tutorial_status = source_profile.tutorial_status
    target_profile.onboarding_steps = source_profile.onboarding_steps
    target_profile.save(update_fields=['tutorial_status', 'onboarding_steps'])