import copy
import html
import threading
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from xml.etree import ElementTree
import requests
from plexapi import BASE_HEADERS, CONFIG, TIMEOUT, X_PLEX_ENABLE_FAST_CONNECT, X_PLEX_IDENTIFIER, log, logfilter, utils
from plexapi.base import PlexObject
from plexapi.client import PlexClient
from plexapi.exceptions import BadRequest, NotFound, Unauthorized
from plexapi.library import LibrarySection
from plexapi.server import PlexServer
from plexapi.sonos import PlexSonosClient
from plexapi.sync import SyncItem, SyncList
from requests.status_codes import _codes as codes

class MyPlexAccount(PlexObject):
    """ MyPlex account and profile information. This object represents the data found Account on
        the myplex.tv servers at the url https://plex.tv/api/v2/user. You may create this object
        directly by passing in your username & password (or token). There is also a convenience
        method provided at :class:`~plexapi.server.PlexServer.myPlexAccount()` which will create
        and return this object.

        Parameters:
            username (str): Plex login username if not using a token.
            password (str): Plex login password if not using a token.
            token (str): Plex authentication token instead of username and password.
            session (requests.Session, optional): Use your own session object if you want to
                cache the http responses from PMS.
            timeout (int): timeout in seconds on initial connect to myplex (default config.TIMEOUT).
            code (str): Two-factor authentication code to use when logging in with username and password.
            remember (bool): Remember the account token for 14 days (Default True).

        Attributes:
            key (str): 'https://plex.tv/api/v2/user'
            adsConsent (str): Unknown.
            adsConsentReminderAt (str): Unknown.
            adsConsentSetAt (str): Unknown.
            anonymous (str): Unknown.
            authToken (str): The account token.
            backupCodesCreated (bool): If the two-factor authentication backup codes have been created.
            confirmed (bool): If the account has been confirmed.
            country (str): The account country.
            email (str): The account email address.
            emailOnlyAuth (bool): If login with email only is enabled.
            experimentalFeatures (bool): If experimental features are enabled.
            friendlyName (str): Your account full name.
            entitlements (List<str>): List of devices your allowed to use with this account.
            guest (bool): If the account is a Plex Home guest user.
            hasPassword (bool): If the account has a password.
            home (bool): If the account is a Plex Home user.
            homeAdmin (bool): If the account is the Plex Home admin.
            homeSize (int): The number of accounts in the Plex Home.
            id (int): The Plex account ID.
            joinedAt (datetime): Date the account joined Plex.
            locale (str): the account locale
            mailingListActive (bool): If you are subscribed to the Plex newsletter.
            mailingListStatus (str): Your current mailing list status.
            maxHomeSize (int): The maximum number of accounts allowed in the Plex Home.
            pin (str): The hashed Plex Home PIN.
            profileAutoSelectAudio (bool): If the account has automatically select audio and subtitle tracks enabled.
            profileDefaultAudioLanguage (str): The preferred audio language for the account.
            profileDefaultSubtitleLanguage (str): The preferred subtitle language for the account.
            profileAutoSelectSubtitle (int): The auto-select subtitle mode
                (0 = Manually selected, 1 = Shown with foreign audio, 2 = Always enabled).
            profileDefaultSubtitleAccessibility (int): The subtitles for the deaf or hard-of-hearing (SDH) searches mode
                (0 = Prefer non-SDH subtitles, 1 = Prefer SDH subtitles, 2 = Only show SDH subtitles,
                3 = Only shown non-SDH subtitles).
            profileDefaultSubtitleForced (int): The forced subtitles searches mode
                (0 = Prefer non-forced subtitles, 1 = Prefer forced subtitles, 2 = Only show forced subtitles,
                3 = Only show non-forced subtitles).
            protected (bool): If the account has a Plex Home PIN enabled.
            rememberExpiresAt (datetime): Date the token expires.
            restricted (bool): If the account is a Plex Home managed user.
            roles: (List<str>) Lit of account roles. Plexpass membership listed here.
            scrobbleTypes (List<int>): Unknown.
            subscriptionActive (bool): If the account's Plex Pass subscription is active.
            subscriptionDescription (str): Description of the Plex Pass subscription.
            subscriptionFeatures: (List<str>) List of features allowed on your Plex Pass subscription.
            subscriptionPaymentService (str): Payment service used for your Plex Pass subscription.
            subscriptionPlan (str): Name of Plex Pass subscription plan.
            subscriptionStatus (str): String representation of ``subscriptionActive``.
            subscriptionSubscribedAt (datetime): Date the account subscribed to Plex Pass.
            thumb (str): URL of the account thumbnail.
            title (str): The title of the account (username or friendly name).
            twoFactorEnabled (bool): If two-factor authentication is enabled.
            username (str): The account username.
            uuid (str): The account UUID.
    """
    FRIENDINVITE = 'https://plex.tv/api/servers/{machineId}/shared_servers'
    HOMEUSERS = 'https://plex.tv/api/home/users'
    HOMEUSERCREATE = 'https://plex.tv/api/home/users?title={title}'
    EXISTINGUSER = 'https://plex.tv/api/home/users?invitedEmail={username}'
    FRIENDSERVERS = 'https://plex.tv/api/servers/{machineId}/shared_servers/{serverId}'
    PLEXSERVERS = 'https://plex.tv/api/servers/{machineId}'
    FRIENDUPDATE = 'https://plex.tv/api/friends/{userId}'
    HOMEUSER = 'https://plex.tv/api/home/users/{userId}'
    MANAGEDHOMEUSER = 'https://plex.tv/api/v2/home/users/restricted/{userId}'
    SIGNIN = 'https://plex.tv/api/v2/users/signin'
    SIGNOUT = 'https://plex.tv/api/v2/users/signout'
    WEBHOOKS = 'https://plex.tv/api/v2/user/webhooks'
    OPTOUTS = 'https://plex.tv/api/v2/user/{userUUID}/settings/opt_outs'
    LINK = 'https://plex.tv/api/v2/pins/link'
    VIEWSTATESYNC = 'https://plex.tv/api/v2/user/view_state_sync'
    VOD = 'https://vod.provider.plex.tv'
    MUSIC = 'https://music.provider.plex.tv'
    DISCOVER = 'https://discover.provider.plex.tv'
    METADATA = 'https://metadata.provider.plex.tv'
    key = 'https://plex.tv/api/v2/user'

    def __init__(self, username=None, password=None, token=None, session=None, timeout=None, code=None, remember=True):
        if False:
            print('Hello World!')
        self._token = logfilter.add_secret(token or CONFIG.get('auth.server_token'))
        self._session = session or requests.Session()
        self._timeout = timeout or TIMEOUT
        self._sonos_cache = []
        self._sonos_cache_timestamp = 0
        (data, initpath) = self._signin(username, password, code, remember, timeout)
        super(MyPlexAccount, self).__init__(self, data, initpath)

    def _signin(self, username, password, code, remember, timeout):
        if False:
            return 10
        if self._token:
            return (self.query(self.key), self.key)
        payload = {'login': username or CONFIG.get('auth.myplex_username'), 'password': password or CONFIG.get('auth.myplex_password'), 'rememberMe': remember}
        if code:
            payload['verificationCode'] = code
        data = self.query(self.SIGNIN, method=self._session.post, data=payload, timeout=timeout)
        return (data, self.SIGNIN)

    def signout(self):
        if False:
            return 10
        ' Sign out of the Plex account. Invalidates the authentication token. '
        return self.query(self.SIGNOUT, method=self._session.delete)

    def _loadData(self, data):
        if False:
            for i in range(10):
                print('nop')
        ' Load attribute values from Plex XML response. '
        self._data = data
        self._token = logfilter.add_secret(data.attrib.get('authToken'))
        self._webhooks = []
        self.adsConsent = data.attrib.get('adsConsent')
        self.adsConsentReminderAt = data.attrib.get('adsConsentReminderAt')
        self.adsConsentSetAt = data.attrib.get('adsConsentSetAt')
        self.anonymous = data.attrib.get('anonymous')
        self.authToken = self._token
        self.backupCodesCreated = utils.cast(bool, data.attrib.get('backupCodesCreated'))
        self.confirmed = utils.cast(bool, data.attrib.get('confirmed'))
        self.country = data.attrib.get('country')
        self.email = data.attrib.get('email')
        self.emailOnlyAuth = utils.cast(bool, data.attrib.get('emailOnlyAuth'))
        self.experimentalFeatures = utils.cast(bool, data.attrib.get('experimentalFeatures'))
        self.friendlyName = data.attrib.get('friendlyName')
        self.guest = utils.cast(bool, data.attrib.get('guest'))
        self.hasPassword = utils.cast(bool, data.attrib.get('hasPassword'))
        self.home = utils.cast(bool, data.attrib.get('home'))
        self.homeAdmin = utils.cast(bool, data.attrib.get('homeAdmin'))
        self.homeSize = utils.cast(int, data.attrib.get('homeSize'))
        self.id = utils.cast(int, data.attrib.get('id'))
        self.joinedAt = utils.toDatetime(data.attrib.get('joinedAt'))
        self.locale = data.attrib.get('locale')
        self.mailingListActive = utils.cast(bool, data.attrib.get('mailingListActive'))
        self.mailingListStatus = data.attrib.get('mailingListStatus')
        self.maxHomeSize = utils.cast(int, data.attrib.get('maxHomeSize'))
        self.pin = data.attrib.get('pin')
        self.protected = utils.cast(bool, data.attrib.get('protected'))
        self.rememberExpiresAt = utils.toDatetime(data.attrib.get('rememberExpiresAt'))
        self.restricted = utils.cast(bool, data.attrib.get('restricted'))
        self.scrobbleTypes = [utils.cast(int, x) for x in data.attrib.get('scrobbleTypes').split(',')]
        self.thumb = data.attrib.get('thumb')
        self.title = data.attrib.get('title')
        self.twoFactorEnabled = utils.cast(bool, data.attrib.get('twoFactorEnabled'))
        self.username = data.attrib.get('username')
        self.uuid = data.attrib.get('uuid')
        subscription = data.find('subscription')
        self.subscriptionActive = utils.cast(bool, subscription.attrib.get('active'))
        self.subscriptionDescription = data.attrib.get('subscriptionDescription')
        self.subscriptionFeatures = self.listAttrs(subscription, 'id', rtag='features', etag='feature')
        self.subscriptionPaymentService = subscription.attrib.get('paymentService')
        self.subscriptionPlan = subscription.attrib.get('plan')
        self.subscriptionStatus = subscription.attrib.get('status')
        self.subscriptionSubscribedAt = utils.toDatetime(subscription.attrib.get('subscribedAt') or None, '%Y-%m-%d %H:%M:%S %Z')
        profile = data.find('profile')
        self.profileAutoSelectAudio = utils.cast(bool, profile.attrib.get('autoSelectAudio'))
        self.profileDefaultAudioLanguage = profile.attrib.get('defaultAudioLanguage')
        self.profileDefaultSubtitleLanguage = profile.attrib.get('defaultSubtitleLanguage')
        self.profileAutoSelectSubtitle = utils.cast(int, profile.attrib.get('autoSelectSubtitle'))
        self.profileDefaultSubtitleAccessibility = utils.cast(int, profile.attrib.get('defaultSubtitleAccessibility'))
        self.profileDefaultSubtitleForces = utils.cast(int, profile.attrib.get('defaultSubtitleForces'))
        self.entitlements = self.listAttrs(data, 'id', rtag='entitlements', etag='entitlement')
        self.roles = self.listAttrs(data, 'id', rtag='roles', etag='role')
        self.services = None

    @property
    def authenticationToken(self):
        if False:
            print('Hello World!')
        ' Returns the authentication token for the account. Alias for ``authToken``. '
        return self.authToken

    def _reload(self, key=None, **kwargs):
        if False:
            return 10
        ' Perform the actual reload. '
        data = self.query(self.key)
        self._loadData(data)
        return self

    def _headers(self, **kwargs):
        if False:
            print('Hello World!')
        ' Returns dict containing base headers for all requests to the server. '
        headers = BASE_HEADERS.copy()
        if self._token:
            headers['X-Plex-Token'] = self._token
        headers.update(kwargs)
        return headers

    def query(self, url, method=None, headers=None, timeout=None, **kwargs):
        if False:
            while True:
                i = 10
        method = method or self._session.get
        timeout = timeout or self._timeout
        log.debug('%s %s %s', method.__name__.upper(), url, kwargs.get('json', ''))
        headers = self._headers(**headers or {})
        response = method(url, headers=headers, timeout=timeout, **kwargs)
        if response.status_code not in (200, 201, 204):
            codename = codes.get(response.status_code)[0]
            errtext = response.text.replace('\n', ' ')
            message = f'({response.status_code}) {codename}; {response.url} {errtext}'
            if response.status_code == 401:
                raise Unauthorized(message)
            elif response.status_code == 404:
                raise NotFound(message)
            elif response.status_code == 422 and 'Invalid token' in response.text:
                raise Unauthorized(message)
            else:
                raise BadRequest(message)
        if 'application/json' in response.headers.get('Content-Type', ''):
            return response.json()
        elif 'text/plain' in response.headers.get('Content-Type', ''):
            return response.text.strip()
        data = response.text.encode('utf8')
        return ElementTree.fromstring(data) if data.strip() else None

    def device(self, name=None, clientId=None):
        if False:
            while True:
                i = 10
        ' Returns the :class:`~plexapi.myplex.MyPlexDevice` that matches the name specified.\n\n            Parameters:\n                name (str): Name to match against.\n                clientId (str): clientIdentifier to match against.\n        '
        for device in self.devices():
            if name and device.name.lower() == name.lower() or device.clientIdentifier == clientId:
                return device
        raise NotFound(f'Unable to find device {name}')

    def devices(self):
        if False:
            return 10
        ' Returns a list of all :class:`~plexapi.myplex.MyPlexDevice` objects connected to the server. '
        data = self.query(MyPlexDevice.key)
        return [MyPlexDevice(self, elem) for elem in data]

    def resource(self, name):
        if False:
            print('Hello World!')
        ' Returns the :class:`~plexapi.myplex.MyPlexResource` that matches the name specified.\n\n            Parameters:\n                name (str): Name to match against.\n        '
        for resource in self.resources():
            if resource.name.lower() == name.lower():
                return resource
        raise NotFound(f'Unable to find resource {name}')

    def resources(self):
        if False:
            for i in range(10):
                print('nop')
        ' Returns a list of all :class:`~plexapi.myplex.MyPlexResource` objects connected to the server. '
        data = self.query(MyPlexResource.key)
        return [MyPlexResource(self, elem) for elem in data]

    def sonos_speakers(self):
        if False:
            while True:
                i = 10
        if 'companions_sonos' not in self.subscriptionFeatures:
            return []
        t = time.time()
        if t - self._sonos_cache_timestamp > 5:
            self._sonos_cache_timestamp = t
            data = self.query('https://sonos.plex.tv/resources')
            self._sonos_cache = [PlexSonosClient(self, elem) for elem in data]
        return self._sonos_cache

    def sonos_speaker(self, name):
        if False:
            i = 10
            return i + 15
        return next((x for x in self.sonos_speakers() if x.title.split('+')[0].strip() == name), None)

    def sonos_speaker_by_id(self, identifier):
        if False:
            i = 10
            return i + 15
        return next((x for x in self.sonos_speakers() if x.machineIdentifier.startswith(identifier)), None)

    def inviteFriend(self, user, server, sections=None, allowSync=False, allowCameraUpload=False, allowChannels=False, filterMovies=None, filterTelevision=None, filterMusic=None):
        if False:
            return 10
        " Share library content with the specified user.\n\n            Parameters:\n                user (:class:`~plexapi.myplex.MyPlexUser`): `MyPlexUser` object, username, or email\n                    of the user to be added.\n                server (:class:`~plexapi.server.PlexServer`): `PlexServer` object, or machineIdentifier\n                    containing the library sections to share.\n                sections (List<:class:`~plexapi.library.LibrarySection`>): List of `LibrarySection` objects, or names\n                    to be shared (default None). `sections` must be defined in order to update shared libraries.\n                allowSync (Bool): Set True to allow user to sync content.\n                allowCameraUpload (Bool): Set True to allow user to upload photos.\n                allowChannels (Bool): Set True to allow user to utilize installed channels.\n                filterMovies (Dict): Dict containing key 'contentRating' and/or 'label' each set to a list of\n                    values to be filtered. ex: `{'contentRating':['G'], 'label':['foo']}`\n                filterTelevision (Dict): Dict containing key 'contentRating' and/or 'label' each set to a list of\n                    values to be filtered. ex: `{'contentRating':['G'], 'label':['foo']}`\n                filterMusic (Dict): Dict containing key 'label' set to a list of values to be filtered.\n                    ex: `{'label':['foo']}`\n        "
        username = user.username if isinstance(user, MyPlexUser) else user
        machineId = server.machineIdentifier if isinstance(server, PlexServer) else server
        sectionIds = self._getSectionIds(machineId, sections)
        params = {'server_id': machineId, 'shared_server': {'library_section_ids': sectionIds, 'invited_email': username}, 'sharing_settings': {'allowSync': '1' if allowSync else '0', 'allowCameraUpload': '1' if allowCameraUpload else '0', 'allowChannels': '1' if allowChannels else '0', 'filterMovies': self._filterDictToStr(filterMovies or {}), 'filterTelevision': self._filterDictToStr(filterTelevision or {}), 'filterMusic': self._filterDictToStr(filterMusic or {})}}
        headers = {'Content-Type': 'application/json'}
        url = self.FRIENDINVITE.format(machineId=machineId)
        return self.query(url, self._session.post, json=params, headers=headers)

    def createHomeUser(self, user, server, sections=None, allowSync=False, allowCameraUpload=False, allowChannels=False, filterMovies=None, filterTelevision=None, filterMusic=None):
        if False:
            while True:
                i = 10
        " Share library content with the specified user.\n\n            Parameters:\n                user (:class:`~plexapi.myplex.MyPlexUser`): `MyPlexUser` object, username, or email\n                    of the user to be added.\n                server (:class:`~plexapi.server.PlexServer`): `PlexServer` object, or machineIdentifier\n                    containing the library sections to share.\n                sections (List<:class:`~plexapi.library.LibrarySection`>): List of `LibrarySection` objects, or names\n                    to be shared (default None). `sections` must be defined in order to update shared libraries.\n                allowSync (Bool): Set True to allow user to sync content.\n                allowCameraUpload (Bool): Set True to allow user to upload photos.\n                allowChannels (Bool): Set True to allow user to utilize installed channels.\n                filterMovies (Dict): Dict containing key 'contentRating' and/or 'label' each set to a list of\n                    values to be filtered. ex: `{'contentRating':['G'], 'label':['foo']}`\n                filterTelevision (Dict): Dict containing key 'contentRating' and/or 'label' each set to a list of\n                    values to be filtered. ex: `{'contentRating':['G'], 'label':['foo']}`\n                filterMusic (Dict): Dict containing key 'label' set to a list of values to be filtered.\n                    ex: `{'label':['foo']}`\n        "
        machineId = server.machineIdentifier if isinstance(server, PlexServer) else server
        sectionIds = self._getSectionIds(server, sections)
        headers = {'Content-Type': 'application/json'}
        url = self.HOMEUSERCREATE.format(title=user)
        user_creation = self.query(url, self._session.post, headers=headers)
        userIds = {}
        for elem in user_creation.findall('.'):
            userIds['id'] = elem.attrib.get('id')
        log.debug(userIds)
        params = {'server_id': machineId, 'shared_server': {'library_section_ids': sectionIds, 'invited_id': userIds['id']}, 'sharing_settings': {'allowSync': '1' if allowSync else '0', 'allowCameraUpload': '1' if allowCameraUpload else '0', 'allowChannels': '1' if allowChannels else '0', 'filterMovies': self._filterDictToStr(filterMovies or {}), 'filterTelevision': self._filterDictToStr(filterTelevision or {}), 'filterMusic': self._filterDictToStr(filterMusic or {})}}
        url = self.FRIENDINVITE.format(machineId=machineId)
        library_assignment = self.query(url, self._session.post, json=params, headers=headers)
        return (user_creation, library_assignment)

    def createExistingUser(self, user, server, sections=None, allowSync=False, allowCameraUpload=False, allowChannels=False, filterMovies=None, filterTelevision=None, filterMusic=None):
        if False:
            print('Hello World!')
        " Share library content with the specified user.\n\n            Parameters:\n                user (:class:`~plexapi.myplex.MyPlexUser`): `MyPlexUser` object, username, or email\n                    of the user to be added.\n                server (:class:`~plexapi.server.PlexServer`): `PlexServer` object, or machineIdentifier\n                    containing the library sections to share.\n                sections (List<:class:`~plexapi.library.LibrarySection`>): List of `LibrarySection` objects, or names\n                    to be shared (default None). `sections` must be defined in order to update shared libraries.\n                allowSync (Bool): Set True to allow user to sync content.\n                allowCameraUpload (Bool): Set True to allow user to upload photos.\n                allowChannels (Bool): Set True to allow user to utilize installed channels.\n                filterMovies (Dict): Dict containing key 'contentRating' and/or 'label' each set to a list of\n                    values to be filtered. ex: `{'contentRating':['G'], 'label':['foo']}`\n                filterTelevision (Dict): Dict containing key 'contentRating' and/or 'label' each set to a list of\n                    values to be filtered. ex: `{'contentRating':['G'], 'label':['foo']}`\n                filterMusic (Dict): Dict containing key 'label' set to a list of values to be filtered.\n                    ex: `{'label':['foo']}`\n        "
        headers = {'Content-Type': 'application/json'}
        if isinstance(user, MyPlexUser):
            username = user.username
        elif user in [_user.username for _user in self.users()]:
            username = self.user(user).username
        else:
            newUser = user
            url = self.EXISTINGUSER.format(username=newUser)
            user_creation = self.query(url, self._session.post, headers=headers)
            machineId = server.machineIdentifier if isinstance(server, PlexServer) else server
            sectionIds = self._getSectionIds(server, sections)
            params = {'server_id': machineId, 'shared_server': {'library_section_ids': sectionIds, 'invited_email': newUser}, 'sharing_settings': {'allowSync': '1' if allowSync else '0', 'allowCameraUpload': '1' if allowCameraUpload else '0', 'allowChannels': '1' if allowChannels else '0', 'filterMovies': self._filterDictToStr(filterMovies or {}), 'filterTelevision': self._filterDictToStr(filterTelevision or {}), 'filterMusic': self._filterDictToStr(filterMusic or {})}}
            url = self.FRIENDINVITE.format(machineId=machineId)
            library_assignment = self.query(url, self._session.post, json=params, headers=headers)
            return (user_creation, library_assignment)
        url = self.EXISTINGUSER.format(username=username)
        return self.query(url, self._session.post, headers=headers)

    def removeFriend(self, user):
        if False:
            i = 10
            return i + 15
        ' Remove the specified user from your friends.\n\n            Parameters:\n                user (:class:`~plexapi.myplex.MyPlexUser` or str): :class:`~plexapi.myplex.MyPlexUser`,\n                    username, or email of the user to be removed.\n        '
        user = user if isinstance(user, MyPlexUser) else self.user(user)
        url = self.FRIENDUPDATE.format(userId=user.id)
        return self.query(url, self._session.delete)

    def removeHomeUser(self, user):
        if False:
            return 10
        ' Remove the specified user from your home users.\n\n            Parameters:\n                user (:class:`~plexapi.myplex.MyPlexUser` or str): :class:`~plexapi.myplex.MyPlexUser`,\n                    username, or email of the user to be removed.\n        '
        user = user if isinstance(user, MyPlexUser) else self.user(user)
        url = self.HOMEUSER.format(userId=user.id)
        return self.query(url, self._session.delete)

    def switchHomeUser(self, user, pin=None):
        if False:
            i = 10
            return i + 15
        " Returns a new :class:`~plexapi.myplex.MyPlexAccount` object switched to the given home user.\n\n            Parameters:\n                user (:class:`~plexapi.myplex.MyPlexUser` or str): :class:`~plexapi.myplex.MyPlexUser`,\n                    username, or email of the home user to switch to.\n                pin (str): PIN for the home user (required if the home user has a PIN set).\n\n            Example:\n\n                .. code-block:: python\n\n                    from plexapi.myplex import MyPlexAccount\n                    # Login to a Plex Home account\n                    account = MyPlexAccount('<USERNAME>', '<PASSWORD>')\n                    # Switch to a different Plex Home user\n                    userAccount = account.switchHomeUser('Username')\n\n        "
        user = user if isinstance(user, MyPlexUser) else self.user(user)
        url = f'{self.HOMEUSERS}/{user.id}/switch'
        params = {}
        if pin:
            params['pin'] = pin
        data = self.query(url, self._session.post, params=params)
        userToken = data.attrib.get('authenticationToken')
        return MyPlexAccount(token=userToken, session=self._session)

    def setPin(self, newPin, currentPin=None):
        if False:
            while True:
                i = 10
        ' Set a new Plex Home PIN for the account.\n\n            Parameters:\n                newPin (str): New PIN to set for the account.\n                currentPin (str): Current PIN for the account (required to change the PIN).\n        '
        url = self.HOMEUSER.format(userId=self.id)
        params = {'pin': newPin}
        if currentPin:
            params['currentPin'] = currentPin
        return self.query(url, self._session.put, params=params)

    def removePin(self, currentPin):
        if False:
            while True:
                i = 10
        ' Remove the Plex Home PIN for the account.\n\n            Parameters:\n                currentPin (str): Current PIN for the account (required to remove the PIN).\n        '
        return self.setPin('', currentPin)

    def setManagedUserPin(self, user, newPin):
        if False:
            for i in range(10):
                print('nop')
        ' Set a new Plex Home PIN for a managed home user. This must be done from the Plex Home admin account.\n\n            Parameters:\n                user (:class:`~plexapi.myplex.MyPlexUser` or str): :class:`~plexapi.myplex.MyPlexUser`\n                    or username of the managed home user.\n                newPin (str): New PIN to set for the managed home user.\n        '
        user = user if isinstance(user, MyPlexUser) else self.user(user)
        url = self.MANAGEDHOMEUSER.format(userId=user.id)
        params = {'pin': newPin}
        return self.query(url, self._session.post, params=params)

    def removeManagedUserPin(self, user):
        if False:
            print('Hello World!')
        ' Remove the Plex Home PIN for a managed home user. This must be done from the Plex Home admin account.\n\n            Parameters:\n                user (:class:`~plexapi.myplex.MyPlexUser` or str): :class:`~plexapi.myplex.MyPlexUser`\n                    or username of the managed home user.\n        '
        user = user if isinstance(user, MyPlexUser) else self.user(user)
        url = self.MANAGEDHOMEUSER.format(userId=user.id)
        params = {'removePin': 1}
        return self.query(url, self._session.post, params=params)

    def acceptInvite(self, user):
        if False:
            while True:
                i = 10
        ' Accept a pending friend invite from the specified user.\n\n            Parameters:\n                user (:class:`~plexapi.myplex.MyPlexInvite` or str): :class:`~plexapi.myplex.MyPlexInvite`,\n                    username, or email of the friend invite to accept.\n        '
        invite = user if isinstance(user, MyPlexInvite) else self.pendingInvite(user, includeSent=False)
        params = {'friend': int(invite.friend), 'home': int(invite.home), 'server': int(invite.server)}
        url = MyPlexInvite.REQUESTS + f'/{invite.id}' + utils.joinArgs(params)
        return self.query(url, self._session.put)

    def cancelInvite(self, user):
        if False:
            i = 10
            return i + 15
        ' Cancel a pending firend invite for the specified user.\n\n            Parameters:\n                user (:class:`~plexapi.myplex.MyPlexInvite` or str): :class:`~plexapi.myplex.MyPlexInvite`,\n                    username, or email of the friend invite to cancel.\n        '
        invite = user if isinstance(user, MyPlexInvite) else self.pendingInvite(user, includeReceived=False)
        params = {'friend': int(invite.friend), 'home': int(invite.home), 'server': int(invite.server)}
        url = MyPlexInvite.REQUESTED + f'/{invite.id}' + utils.joinArgs(params)
        return self.query(url, self._session.delete)

    def updateFriend(self, user, server, sections=None, removeSections=False, allowSync=None, allowCameraUpload=None, allowChannels=None, filterMovies=None, filterTelevision=None, filterMusic=None):
        if False:
            i = 10
            return i + 15
        " Update the specified user's share settings.\n\n            Parameters:\n                user (:class:`~plexapi.myplex.MyPlexUser`): `MyPlexUser` object, username, or email\n                    of the user to be updated.\n                server (:class:`~plexapi.server.PlexServer`): `PlexServer` object, or machineIdentifier\n                    containing the library sections to share.\n                sections (List<:class:`~plexapi.library.LibrarySection`>): List of `LibrarySection` objects, or names\n                    to be shared (default None). `sections` must be defined in order to update shared libraries.\n                removeSections (Bool): Set True to remove all shares. Supersedes sections.\n                allowSync (Bool): Set True to allow user to sync content.\n                allowCameraUpload (Bool): Set True to allow user to upload photos.\n                allowChannels (Bool): Set True to allow user to utilize installed channels.\n                filterMovies (Dict): Dict containing key 'contentRating' and/or 'label' each set to a list of\n                    values to be filtered. ex: `{'contentRating':['G'], 'label':['foo']}`\n                filterTelevision (Dict): Dict containing key 'contentRating' and/or 'label' each set to a list of\n                    values to be filtered. ex: `{'contentRating':['G'], 'label':['foo']}`\n                filterMusic (Dict): Dict containing key 'label' set to a list of values to be filtered.\n                    ex: `{'label':['foo']}`\n        "
        response_filters = ''
        response_servers = ''
        user = user if isinstance(user, MyPlexUser) else self.user(user)
        machineId = server.machineIdentifier if isinstance(server, PlexServer) else server
        sectionIds = self._getSectionIds(machineId, sections)
        headers = {'Content-Type': 'application/json'}
        user_servers = [s for s in user.servers if s.machineIdentifier == machineId]
        if user_servers and sectionIds:
            serverId = user_servers[0].id
            params = {'server_id': machineId, 'shared_server': {'library_section_ids': sectionIds}}
            url = self.FRIENDSERVERS.format(machineId=machineId, serverId=serverId)
        else:
            params = {'server_id': machineId, 'shared_server': {'library_section_ids': sectionIds, 'invited_id': user.id}}
            url = self.FRIENDINVITE.format(machineId=machineId)
        if not user_servers or sectionIds:
            if removeSections is True:
                response_servers = self.query(url, self._session.delete, json=params, headers=headers)
            elif 'invited_id' in params.get('shared_server', ''):
                response_servers = self.query(url, self._session.post, json=params, headers=headers)
            else:
                response_servers = self.query(url, self._session.put, json=params, headers=headers)
        else:
            log.warning('Section name, number of section object is required changing library sections')
        url = self.FRIENDUPDATE.format(userId=user.id)
        params = {}
        if isinstance(allowSync, bool):
            params['allowSync'] = '1' if allowSync else '0'
        if isinstance(allowCameraUpload, bool):
            params['allowCameraUpload'] = '1' if allowCameraUpload else '0'
        if isinstance(allowChannels, bool):
            params['allowChannels'] = '1' if allowChannels else '0'
        if isinstance(filterMovies, dict):
            params['filterMovies'] = self._filterDictToStr(filterMovies or {})
        if isinstance(filterTelevision, dict):
            params['filterTelevision'] = self._filterDictToStr(filterTelevision or {})
        if isinstance(allowChannels, dict):
            params['filterMusic'] = self._filterDictToStr(filterMusic or {})
        if params:
            url += utils.joinArgs(params)
            response_filters = self.query(url, self._session.put)
        return (response_servers, response_filters)

    def user(self, username):
        if False:
            while True:
                i = 10
        ' Returns the :class:`~plexapi.myplex.MyPlexUser` that matches the specified username or email.\n\n            Parameters:\n                username (str): Username, email or id of the user to return.\n        '
        username = str(username)
        for user in self.users():
            if username.lower() == user.title.lower():
                return user
            elif user.username and user.email and user.id and (username.lower() in (user.username.lower(), user.email.lower(), str(user.id))):
                return user
        raise NotFound(f'Unable to find user {username}')

    def users(self):
        if False:
            while True:
                i = 10
        ' Returns a list of all :class:`~plexapi.myplex.MyPlexUser` objects connected to your account.\n        '
        elem = self.query(MyPlexUser.key)
        return self.findItems(elem, cls=MyPlexUser)

    def pendingInvite(self, username, includeSent=True, includeReceived=True):
        if False:
            while True:
                i = 10
        ' Returns the :class:`~plexapi.myplex.MyPlexInvite` that matches the specified username or email.\n            Note: This can be a pending invite sent from your account or received to your account.\n\n            Parameters:\n                username (str): Username, email or id of the user to return.\n                includeSent (bool): True to include sent invites.\n                includeReceived (bool): True to include received invites.\n        '
        username = str(username)
        for invite in self.pendingInvites(includeSent, includeReceived):
            if invite.username and invite.email and invite.id and (username.lower() in (invite.username.lower(), invite.email.lower(), str(invite.id))):
                return invite
        raise NotFound(f'Unable to find invite {username}')

    def pendingInvites(self, includeSent=True, includeReceived=True):
        if False:
            print('Hello World!')
        ' Returns a list of all :class:`~plexapi.myplex.MyPlexInvite` objects connected to your account.\n            Note: This includes all pending invites sent from your account and received to your account.\n\n            Parameters:\n                includeSent (bool): True to include sent invites.\n                includeReceived (bool): True to include received invites.\n        '
        invites = []
        if includeSent:
            elem = self.query(MyPlexInvite.REQUESTED)
            invites += self.findItems(elem, cls=MyPlexInvite)
        if includeReceived:
            elem = self.query(MyPlexInvite.REQUESTS)
            invites += self.findItems(elem, cls=MyPlexInvite)
        return invites

    def _getSectionIds(self, server, sections):
        if False:
            for i in range(10):
                print('nop')
        ' Converts a list of section objects or names to sectionIds needed for library sharing. '
        if not sections:
            return []
        allSectionIds = {}
        machineIdentifier = server.machineIdentifier if isinstance(server, PlexServer) else server
        url = self.PLEXSERVERS.format(machineId=machineIdentifier)
        data = self.query(url, self._session.get)
        for elem in data[0]:
            _id = utils.cast(int, elem.attrib.get('id'))
            _key = utils.cast(int, elem.attrib.get('key'))
            _title = elem.attrib.get('title', '').lower()
            allSectionIds[_id] = _id
            allSectionIds[_key] = _id
            allSectionIds[_title] = _id
        log.debug(allSectionIds)
        sectionIds = []
        for section in sections:
            sectionKey = section.key if isinstance(section, LibrarySection) else section.lower()
            sectionIds.append(allSectionIds[sectionKey])
        return sectionIds

    def _filterDictToStr(self, filterDict):
        if False:
            for i in range(10):
                print('nop')
        ' Converts friend filters to a string representation for transport. '
        values = []
        for (key, vals) in filterDict.items():
            if key not in ('contentRating', 'label', 'contentRating!', 'label!'):
                raise BadRequest(f'Unknown filter key: {key}')
            values.append(f"{key}={'%2C'.join(vals)}")
        return '|'.join(values)

    def addWebhook(self, url):
        if False:
            print('Hello World!')
        urls = self._webhooks[:] + [url]
        return self.setWebhooks(urls)

    def deleteWebhook(self, url):
        if False:
            return 10
        urls = copy.copy(self._webhooks)
        if url not in urls:
            raise BadRequest(f'Webhook does not exist: {url}')
        urls.remove(url)
        return self.setWebhooks(urls)

    def setWebhooks(self, urls):
        if False:
            while True:
                i = 10
        log.info('Setting webhooks: %s', urls)
        data = {'urls[]': urls} if len(urls) else {'urls': ''}
        data = self.query(self.WEBHOOKS, self._session.post, data=data)
        self._webhooks = self.listAttrs(data, 'url', etag='webhook')
        return self._webhooks

    def webhooks(self):
        if False:
            i = 10
            return i + 15
        data = self.query(self.WEBHOOKS)
        self._webhooks = self.listAttrs(data, 'url', etag='webhook')
        return self._webhooks

    def optOut(self, playback=None, library=None):
        if False:
            i = 10
            return i + 15
        ' Opt in or out of sharing stuff with plex.\n            See: https://www.plex.tv/about/privacy-legal/\n        '
        params = {}
        if playback is not None:
            params['optOutPlayback'] = int(playback)
        if library is not None:
            params['optOutLibraryStats'] = int(library)
        url = 'https://plex.tv/api/v2/user/privacy'
        return self.query(url, method=self._session.put, data=params)

    def syncItems(self, client=None, clientId=None):
        if False:
            i = 10
            return i + 15
        " Returns an instance of :class:`~plexapi.sync.SyncList` for specified client.\n\n            Parameters:\n                client (:class:`~plexapi.myplex.MyPlexDevice`): a client to query SyncItems for.\n                clientId (str): an identifier of a client to query SyncItems for.\n\n            If both `client` and `clientId` provided the client would be preferred.\n            If neither `client` nor `clientId` provided the clientId would be set to current clients's identifier.\n        "
        if client:
            clientId = client.clientIdentifier
        elif clientId is None:
            clientId = X_PLEX_IDENTIFIER
        data = self.query(SyncList.key.format(clientId=clientId))
        return SyncList(self, data)

    def sync(self, sync_item, client=None, clientId=None):
        if False:
            return 10
        " Adds specified sync item for the client. It's always easier to use methods defined directly in the media\n            objects, e.g. :func:`~plexapi.video.Video.sync`, :func:`~plexapi.audio.Audio.sync`.\n\n            Parameters:\n                client (:class:`~plexapi.myplex.MyPlexDevice`): a client for which you need to add SyncItem to.\n                clientId (str): an identifier of a client for which you need to add SyncItem to.\n                sync_item (:class:`~plexapi.sync.SyncItem`): prepared SyncItem object with all fields set.\n\n            If both `client` and `clientId` provided the client would be preferred.\n            If neither `client` nor `clientId` provided the clientId would be set to current clients's identifier.\n\n            Returns:\n                :class:`~plexapi.sync.SyncItem`: an instance of created syncItem.\n\n            Raises:\n                :exc:`~plexapi.exceptions.BadRequest`: When client with provided clientId wasn't found.\n                :exc:`~plexapi.exceptions.BadRequest`: Provided client doesn't provides `sync-target`.\n        "
        if not client and (not clientId):
            clientId = X_PLEX_IDENTIFIER
        if not client:
            for device in self.devices():
                if device.clientIdentifier == clientId:
                    client = device
                    break
            if not client:
                raise BadRequest(f'Unable to find client by clientId={clientId}')
        if 'sync-target' not in client.provides:
            raise BadRequest("Received client doesn't provides sync-target")
        params = {'SyncItem[title]': sync_item.title, 'SyncItem[rootTitle]': sync_item.rootTitle, 'SyncItem[metadataType]': sync_item.metadataType, 'SyncItem[machineIdentifier]': sync_item.machineIdentifier, 'SyncItem[contentType]': sync_item.contentType, 'SyncItem[Policy][scope]': sync_item.policy.scope, 'SyncItem[Policy][unwatched]': str(int(sync_item.policy.unwatched)), 'SyncItem[Policy][value]': str(sync_item.policy.value if hasattr(sync_item.policy, 'value') else 0), 'SyncItem[Location][uri]': sync_item.location, 'SyncItem[MediaSettings][audioBoost]': str(sync_item.mediaSettings.audioBoost), 'SyncItem[MediaSettings][maxVideoBitrate]': str(sync_item.mediaSettings.maxVideoBitrate), 'SyncItem[MediaSettings][musicBitrate]': str(sync_item.mediaSettings.musicBitrate), 'SyncItem[MediaSettings][photoQuality]': str(sync_item.mediaSettings.photoQuality), 'SyncItem[MediaSettings][photoResolution]': sync_item.mediaSettings.photoResolution, 'SyncItem[MediaSettings][subtitleSize]': str(sync_item.mediaSettings.subtitleSize), 'SyncItem[MediaSettings][videoQuality]': str(sync_item.mediaSettings.videoQuality), 'SyncItem[MediaSettings][videoResolution]': sync_item.mediaSettings.videoResolution}
        url = SyncList.key.format(clientId=client.clientIdentifier)
        data = self.query(url, method=self._session.post, params=params)
        return SyncItem(self, data, None, clientIdentifier=client.clientIdentifier)

    def claimToken(self):
        if False:
            print('Hello World!')
        ' Returns a str, a new "claim-token", which you can use to register your new Plex Server instance to your\n            account.\n            See: https://hub.docker.com/r/plexinc/pms-docker/, https://www.plex.tv/claim/\n        '
        response = self._session.get('https://plex.tv/api/claim/token.json', headers=self._headers(), timeout=TIMEOUT)
        if response.status_code not in (200, 201, 204):
            codename = codes.get(response.status_code)[0]
            errtext = response.text.replace('\n', ' ')
            raise BadRequest(f'({response.status_code}) {codename} {response.url}; {errtext}')
        return response.json()['token']

    def history(self, maxresults=None, mindate=None):
        if False:
            while True:
                i = 10
        ' Get Play History for all library sections on all servers for the owner.\n\n            Parameters:\n                maxresults (int): Only return the specified number of results (optional).\n                mindate (datetime): Min datetime to return results from.\n        '
        servers = [x for x in self.resources() if x.provides == 'server' and x.owned]
        hist = []
        for server in servers:
            conn = server.connect()
            hist.extend(conn.history(maxresults=maxresults, mindate=mindate, accountID=1))
        return hist

    def onlineMediaSources(self):
        if False:
            for i in range(10):
                print('nop')
        ' Returns a list of user account Online Media Sources settings :class:`~plexapi.myplex.AccountOptOut`\n        '
        url = self.OPTOUTS.format(userUUID=self.uuid)
        elem = self.query(url)
        return self.findItems(elem, cls=AccountOptOut, etag='optOut')

    def videoOnDemand(self):
        if False:
            while True:
                i = 10
        ' Returns a list of VOD Hub items :class:`~plexapi.library.Hub`\n        '
        data = self.query(f'{self.VOD}/hubs')
        return self.findItems(data)

    def tidal(self):
        if False:
            print('Hello World!')
        ' Returns a list of tidal Hub items :class:`~plexapi.library.Hub`\n        '
        data = self.query(f'{self.MUSIC}/hubs')
        return self.findItems(data)

    def watchlist(self, filter=None, sort=None, libtype=None, maxresults=None, **kwargs):
        if False:
            return 10
        " Returns a list of :class:`~plexapi.video.Movie` and :class:`~plexapi.video.Show` items in the user's watchlist.\n            Note: The objects returned are from Plex's online metadata. To get the matching item on a Plex server,\n            search for the media using the guid.\n\n            Parameters:\n                filter (str, optional): 'available' or 'released' to only return items that are available or released,\n                    otherwise return all items.\n                sort (str, optional): In the format ``field:dir``. Available fields are ``watchlistedAt`` (Added At),\n                    ``titleSort`` (Title), ``originallyAvailableAt`` (Release Date), or ``rating`` (Critic Rating).\n                    ``dir`` can be ``asc`` or ``desc``.\n                libtype (str, optional): 'movie' or 'show' to only return movies or shows, otherwise return all items.\n                maxresults (int, optional): Only return the specified number of results.\n                **kwargs (dict): Additional custom filters to apply to the search results.\n\n\n            Example:\n\n                .. code-block:: python\n\n                    # Watchlist for released movies sorted by critic rating in descending order\n                    watchlist = account.watchlist(filter='released', sort='rating:desc', libtype='movie')\n                    item = watchlist[0]  # First item in the watchlist\n\n                    # Search for the item on a Plex server\n                    result = plex.library.search(guid=item.guid, libtype=item.type)\n\n        "
        params = {'includeCollections': 1, 'includeExternalMedia': 1}
        if not filter:
            filter = 'all'
        if sort:
            params['sort'] = sort
        if libtype:
            params['type'] = utils.searchType(libtype)
        params.update(kwargs)
        key = f'{self.METADATA}/library/sections/watchlist/{filter}{utils.joinArgs(params)}'
        return self._toOnlineMetadata(self.fetchItems(key, maxresults=maxresults), **kwargs)

    def onWatchlist(self, item):
        if False:
            return 10
        " Returns True if the item is on the user's watchlist.\n\n            Parameters:\n                item (:class:`~plexapi.video.Movie` or :class:`~plexapi.video.Show`): Item to check\n                    if it is on the user's watchlist.\n        "
        return bool(self.userState(item).watchlistedAt)

    def addToWatchlist(self, items):
        if False:
            i = 10
            return i + 15
        " Add media items to the user's watchlist\n\n            Parameters:\n                items (List): List of :class:`~plexapi.video.Movie` or :class:`~plexapi.video.Show`\n                    objects to be added to the watchlist.\n\n            Raises:\n                :exc:`~plexapi.exceptions.BadRequest`: When trying to add invalid or existing\n                    media to the watchlist.\n        "
        if not isinstance(items, list):
            items = [items]
        for item in items:
            if self.onWatchlist(item):
                raise BadRequest(f'"{item.title}" is already on the watchlist')
            ratingKey = item.guid.rsplit('/', 1)[-1]
            self.query(f'{self.METADATA}/actions/addToWatchlist?ratingKey={ratingKey}', method=self._session.put)
        return self

    def removeFromWatchlist(self, items):
        if False:
            while True:
                i = 10
        " Remove media items from the user's watchlist\n\n            Parameters:\n                items (List): List of :class:`~plexapi.video.Movie` or :class:`~plexapi.video.Show`\n                    objects to be added to the watchlist.\n\n            Raises:\n                :exc:`~plexapi.exceptions.BadRequest`: When trying to remove invalid or non-existing\n                    media to the watchlist.\n        "
        if not isinstance(items, list):
            items = [items]
        for item in items:
            if not self.onWatchlist(item):
                raise BadRequest(f'"{item.title}" is not on the watchlist')
            ratingKey = item.guid.rsplit('/', 1)[-1]
            self.query(f'{self.METADATA}/actions/removeFromWatchlist?ratingKey={ratingKey}', method=self._session.put)
        return self

    def userState(self, item):
        if False:
            i = 10
            return i + 15
        ' Returns a :class:`~plexapi.myplex.UserState` object for the specified item.\n\n            Parameters:\n                item (:class:`~plexapi.video.Movie` or :class:`~plexapi.video.Show`): Item to return the user state.\n        '
        ratingKey = item.guid.rsplit('/', 1)[-1]
        data = self.query(f'{self.METADATA}/library/metadata/{ratingKey}/userState')
        return self.findItem(data, cls=UserState)

    def isPlayed(self, item):
        if False:
            for i in range(10):
                print('nop')
        ' Return True if the item is played on Discover.\n\n            Parameters:\n                item (:class:`~plexapi.video.Movie`,\n                :class:`~plexapi.video.Show`, :class:`~plexapi.video.Season` or\n                :class:`~plexapi.video.Episode`): Object from searchDiscover().\n                Can be also result from Plex Movie or Plex TV Series agent.\n        '
        userState = self.userState(item)
        return bool(userState.viewCount > 0) if userState.viewCount else False

    def markPlayed(self, item):
        if False:
            for i in range(10):
                print('nop')
        ' Mark the Plex object as played on Discover.\n\n            Parameters:\n                item (:class:`~plexapi.video.Movie`,\n                :class:`~plexapi.video.Show`, :class:`~plexapi.video.Season` or\n                :class:`~plexapi.video.Episode`): Object from searchDiscover().\n                Can be also result from Plex Movie or Plex TV Series agent.\n        '
        key = f'{self.METADATA}/actions/scrobble'
        ratingKey = item.guid.rsplit('/', 1)[-1]
        params = {'key': ratingKey, 'identifier': 'com.plexapp.plugins.library'}
        self.query(key, params=params)
        return self

    def markUnplayed(self, item):
        if False:
            i = 10
            return i + 15
        ' Mark the Plex object as unplayed on Discover.\n\n            Parameters:\n                item (:class:`~plexapi.video.Movie`,\n                :class:`~plexapi.video.Show`, :class:`~plexapi.video.Season` or\n                :class:`~plexapi.video.Episode`): Object from searchDiscover().\n                Can be also result from Plex Movie or Plex TV Series agent.\n        '
        key = f'{self.METADATA}/actions/unscrobble'
        ratingKey = item.guid.rsplit('/', 1)[-1]
        params = {'key': ratingKey, 'identifier': 'com.plexapp.plugins.library'}
        self.query(key, params=params)
        return self

    def searchDiscover(self, query, limit=30, libtype=None):
        if False:
            for i in range(10):
                print('nop')
        " Search for movies and TV shows in Discover.\n            Returns a list of :class:`~plexapi.video.Movie` and :class:`~plexapi.video.Show` objects.\n\n            Parameters:\n                query (str): Search query.\n                limit (int, optional): Limit to the specified number of results. Default 30.\n                libtype (str, optional): 'movie' or 'show' to only return movies or shows, otherwise return all items.\n        "
        libtypes = {'movie': 'movies', 'show': 'tv'}
        libtype = libtypes.get(libtype, 'movies,tv')
        headers = {'Accept': 'application/json'}
        params = {'query': query, 'limit': limit, 'searchTypes': libtype, 'includeMetadata': 1}
        data = self.query(f'{self.DISCOVER}/library/search', headers=headers, params=params)
        searchResults = data['MediaContainer'].get('SearchResults', [])
        searchResult = next((s.get('SearchResult', []) for s in searchResults if s.get('id') == 'external'), [])
        results = []
        for result in searchResult:
            metadata = result['Metadata']
            type = metadata['type']
            if type == 'movie':
                tag = 'Video'
            elif type == 'show':
                tag = 'Directory'
            else:
                continue
            attrs = ''.join((f'{k}="{html.escape(str(v))}" ' for (k, v) in metadata.items()))
            xml = f'<{tag} {attrs}/>'
            results.append(self._manuallyLoadXML(xml))
        return self._toOnlineMetadata(results)

    @property
    def viewStateSync(self):
        if False:
            i = 10
            return i + 15
        ' Returns True or False if syncing of watch state and ratings\n            is enabled or disabled, respectively, for the account.\n        '
        headers = {'Accept': 'application/json'}
        data = self.query(self.VIEWSTATESYNC, headers=headers)
        return data.get('consent')

    def enableViewStateSync(self):
        if False:
            i = 10
            return i + 15
        ' Enable syncing of watch state and ratings for the account. '
        self._updateViewStateSync(True)

    def disableViewStateSync(self):
        if False:
            while True:
                i = 10
        ' Disable syncing of watch state and ratings for the account. '
        self._updateViewStateSync(False)

    def _updateViewStateSync(self, consent):
        if False:
            while True:
                i = 10
        ' Enable or disable syncing of watch state and ratings for the account.\n\n            Parameters:\n                consent (bool): True to enable, False to disable.\n        '
        params = {'consent': consent}
        self.query(self.VIEWSTATESYNC, method=self._session.put, params=params)

    def link(self, pin):
        if False:
            while True:
                i = 10
        ' Link a device to the account using a pin code.\n\n            Parameters:\n                pin (str): The 4 digit link pin code.\n        '
        headers = {'Content-Type': 'application/x-www-form-urlencoded', 'X-Plex-Product': 'Plex SSO'}
        data = {'code': pin}
        self.query(self.LINK, self._session.put, headers=headers, data=data)

    def _toOnlineMetadata(self, objs, **kwargs):
        if False:
            return 10
        ' Convert a list of media objects to online metadata objects. '
        server = PlexServer(self.METADATA, self._token, session=self._session)
        includeUserState = int(bool(kwargs.pop('includeUserState', True)))
        if not isinstance(objs, list):
            objs = [objs]
        for obj in objs:
            obj._server = server
            url = urlsplit(obj._details_key)
            query = dict(parse_qsl(url.query))
            query['includeUserState'] = includeUserState
            query.pop('includeFields', None)
            obj._details_key = urlunsplit((url.scheme, url.netloc, url.path, urlencode(query), url.fragment))
        return objs

    def publicIP(self):
        if False:
            while True:
                i = 10
        ' Returns your public IP address. '
        return self.query('https://plex.tv/:/ip')

    def geoip(self, ip_address):
        if False:
            return 10
        " Returns a :class:`~plexapi.myplex.GeoLocation` object with geolocation information\n            for an IP address using Plex's GeoIP database.\n\n            Parameters:\n                ip_address (str): IP address to lookup.\n        "
        params = {'ip_address': ip_address}
        data = self.query('https://plex.tv/api/v2/geoip', params=params)
        return GeoLocation(self, data)

class MyPlexUser(PlexObject):
    """ This object represents non-signed in users such as friends and linked
        accounts. NOTE: This should not be confused with the :class:`~plexapi.myplex.MyPlexAccount`
        which is your specific account. The raw xml for the data presented here
        can be found at: https://plex.tv/api/users/

        Attributes:
            TAG (str): 'User'
            key (str): 'https://plex.tv/api/users/'
            allowCameraUpload (bool): True if this user can upload images.
            allowChannels (bool): True if this user has access to channels.
            allowSync (bool): True if this user can sync.
            email (str): User's email address (user@gmail.com).
            filterAll (str): Unknown.
            filterMovies (str): Unknown.
            filterMusic (str): Unknown.
            filterPhotos (str): Unknown.
            filterTelevision (str): Unknown.
            home (bool): Unknown.
            id (int): User's Plex account ID.
            protected (False): Unknown (possibly SSL enabled?).
            recommendationsPlaylistId (str): Unknown.
            restricted (str): Unknown.
            servers (List<:class:`~plexapi.myplex.<MyPlexServerShare`>)): Servers shared with the user.
            thumb (str): Link to the users avatar.
            title (str): Seems to be an alias for username.
            username (str): User's username.
    """
    TAG = 'User'
    key = 'https://plex.tv/api/users/'

    def _loadData(self, data):
        if False:
            print('Hello World!')
        ' Load attribute values from Plex XML response. '
        self._data = data
        self.friend = self._initpath == self.key
        self.allowCameraUpload = utils.cast(bool, data.attrib.get('allowCameraUpload'))
        self.allowChannels = utils.cast(bool, data.attrib.get('allowChannels'))
        self.allowSync = utils.cast(bool, data.attrib.get('allowSync'))
        self.email = data.attrib.get('email')
        self.filterAll = data.attrib.get('filterAll')
        self.filterMovies = data.attrib.get('filterMovies')
        self.filterMusic = data.attrib.get('filterMusic')
        self.filterPhotos = data.attrib.get('filterPhotos')
        self.filterTelevision = data.attrib.get('filterTelevision')
        self.home = utils.cast(bool, data.attrib.get('home'))
        self.id = utils.cast(int, data.attrib.get('id'))
        self.protected = utils.cast(bool, data.attrib.get('protected'))
        self.recommendationsPlaylistId = data.attrib.get('recommendationsPlaylistId')
        self.restricted = data.attrib.get('restricted')
        self.thumb = data.attrib.get('thumb')
        self.title = data.attrib.get('title', '')
        self.username = data.attrib.get('username', '')
        self.servers = self.findItems(data, MyPlexServerShare)
        for server in self.servers:
            server.accountID = self.id

    def get_token(self, machineIdentifier):
        if False:
            print('Hello World!')
        try:
            for item in self._server.query(self._server.FRIENDINVITE.format(machineId=machineIdentifier)):
                if utils.cast(int, item.attrib.get('userID')) == self.id:
                    return item.attrib.get('accessToken')
        except Exception:
            log.exception('Failed to get access token for %s', self.title)

    def server(self, name):
        if False:
            while True:
                i = 10
        ' Returns the :class:`~plexapi.myplex.MyPlexServerShare` that matches the name specified.\n\n            Parameters:\n                name (str): Name of the server to return.\n        '
        for server in self.servers:
            if name.lower() == server.name.lower():
                return server
        raise NotFound(f'Unable to find server {name}')

    def history(self, maxresults=None, mindate=None):
        if False:
            for i in range(10):
                print('nop')
        ' Get all Play History for a user in all shared servers.\n            Parameters:\n                maxresults (int): Only return the specified number of results (optional).\n                mindate (datetime): Min datetime to return results from.\n        '
        hist = []
        for server in self.servers:
            hist.extend(server.history(maxresults=maxresults, mindate=mindate))
        return hist

class MyPlexInvite(PlexObject):
    """ This object represents pending friend invites.

        Attributes:
            TAG (str): 'Invite'
            createdAt (datetime): Datetime the user was invited.
            email (str): User's email address (user@gmail.com).
            friend (bool): True or False if the user is invited as a friend.
            friendlyName (str): The user's friendly name.
            home (bool): True or False if the user is invited to a Plex Home.
            id (int): User's Plex account ID.
            server (bool): True or False if the user is invited to any servers.
            servers (List<:class:`~plexapi.myplex.<MyPlexServerShare`>)): Servers shared with the user.
            thumb (str): Link to the users avatar.
            username (str): User's username.
    """
    TAG = 'Invite'
    REQUESTS = 'https://plex.tv/api/invites/requests'
    REQUESTED = 'https://plex.tv/api/invites/requested'

    def _loadData(self, data):
        if False:
            return 10
        ' Load attribute values from Plex XML response. '
        self._data = data
        self.createdAt = utils.toDatetime(data.attrib.get('createdAt'))
        self.email = data.attrib.get('email')
        self.friend = utils.cast(bool, data.attrib.get('friend'))
        self.friendlyName = data.attrib.get('friendlyName')
        self.home = utils.cast(bool, data.attrib.get('home'))
        self.id = utils.cast(int, data.attrib.get('id'))
        self.server = utils.cast(bool, data.attrib.get('server'))
        self.servers = self.findItems(data, MyPlexServerShare)
        self.thumb = data.attrib.get('thumb')
        self.username = data.attrib.get('username', '')
        for server in self.servers:
            server.accountID = self.id

class Section(PlexObject):
    """ This refers to a shared section. The raw xml for the data presented here
        can be found at: https://plex.tv/api/servers/{machineId}/shared_servers

        Attributes:
            TAG (str): section
            id (int): The shared section ID
            key (int): The shared library section key
            shared (bool): If this section is shared with the user
            title (str): Title of the section
            type (str): movie, tvshow, artist

    """
    TAG = 'Section'

    def _loadData(self, data):
        if False:
            print('Hello World!')
        self._data = data
        self.id = utils.cast(int, data.attrib.get('id'))
        self.key = utils.cast(int, data.attrib.get('key'))
        self.shared = utils.cast(bool, data.attrib.get('shared', '0'))
        self.title = data.attrib.get('title')
        self.type = data.attrib.get('type')
        self.sectionId = self.id
        self.sectionKey = self.key

    def history(self, maxresults=None, mindate=None):
        if False:
            print('Hello World!')
        ' Get all Play History for a user for this section in this shared server.\n            Parameters:\n                maxresults (int): Only return the specified number of results (optional).\n                mindate (datetime): Min datetime to return results from.\n        '
        server = self._server._server.resource(self._server.name).connect()
        return server.history(maxresults=maxresults, mindate=mindate, accountID=self._server.accountID, librarySectionID=self.sectionKey)

class MyPlexServerShare(PlexObject):
    """ Represents a single user's server reference. Used for library sharing.

        Attributes:
            id (int): id for this share
            serverId (str): what id plex uses for this.
            machineIdentifier (str): The servers machineIdentifier
            name (str): The servers name
            lastSeenAt (datetime): Last connected to the server?
            numLibraries (int): Total number of libraries
            allLibraries (bool): True if all libraries is shared with this user.
            owned (bool): 1 if the server is owned by the user
            pending (bool): True if the invite is pending.

    """
    TAG = 'Server'

    def _loadData(self, data):
        if False:
            print('Hello World!')
        ' Load attribute values from Plex XML response. '
        self._data = data
        self.id = utils.cast(int, data.attrib.get('id'))
        self.accountID = utils.cast(int, data.attrib.get('accountID'))
        self.serverId = utils.cast(int, data.attrib.get('serverId'))
        self.machineIdentifier = data.attrib.get('machineIdentifier')
        self.name = data.attrib.get('name')
        self.lastSeenAt = utils.toDatetime(data.attrib.get('lastSeenAt'))
        self.numLibraries = utils.cast(int, data.attrib.get('numLibraries'))
        self.allLibraries = utils.cast(bool, data.attrib.get('allLibraries'))
        self.owned = utils.cast(bool, data.attrib.get('owned'))
        self.pending = utils.cast(bool, data.attrib.get('pending'))

    def section(self, name):
        if False:
            i = 10
            return i + 15
        ' Returns the :class:`~plexapi.myplex.Section` that matches the name specified.\n\n            Parameters:\n                name (str): Name of the section to return.\n        '
        for section in self.sections():
            if name.lower() == section.title.lower():
                return section
        raise NotFound(f'Unable to find section {name}')

    def sections(self):
        if False:
            while True:
                i = 10
        ' Returns a list of all :class:`~plexapi.myplex.Section` objects shared with this user.\n        '
        url = MyPlexAccount.FRIENDSERVERS.format(machineId=self.machineIdentifier, serverId=self.id)
        data = self._server.query(url)
        return self.findItems(data, Section, rtag='SharedServer')

    def history(self, maxresults=9999999, mindate=None):
        if False:
            i = 10
            return i + 15
        ' Get all Play History for a user in this shared server.\n            Parameters:\n                maxresults (int): Only return the specified number of results (optional).\n                mindate (datetime): Min datetime to return results from.\n        '
        server = self._server.resource(self.name).connect()
        return server.history(maxresults=maxresults, mindate=mindate, accountID=self.accountID)

class MyPlexResource(PlexObject):
    """ This object represents resources connected to your Plex server that can provide
        content such as Plex Media Servers, iPhone or Android clients, etc. The raw xml
        for the data presented here can be found at:
        https://plex.tv/api/v2/resources?includeHttps=1&includeRelay=1

        Attributes:
            TAG (str): 'Device'
            key (str): 'https://plex.tv/api/v2/resources?includeHttps=1&includeRelay=1'
            accessToken (str): This resource's Plex access token.
            clientIdentifier (str): Unique ID for this resource.
            connections (list): List of :class:`~plexapi.myplex.ResourceConnection` objects
                for this resource.
            createdAt (datetime): Timestamp this resource first connected to your server.
            device (str): Best guess on the type of device this is (PS, iPhone, Linux, etc).
            dnsRebindingProtection (bool): True if the server had DNS rebinding protection.
            home (bool): Unknown
            httpsRequired (bool): True if the resource requires https.
            lastSeenAt (datetime): Timestamp this resource last connected.
            name (str): Descriptive name of this resource.
            natLoopbackSupported (bool): True if the resource supports NAT loopback.
            owned (bool): True if this resource is one of your own (you logged into it).
            ownerId (int): ID of the user that owns this resource (shared resources only).
            platform (str): OS the resource is running (Linux, Windows, Chrome, etc.)
            platformVersion (str): Version of the platform.
            presence (bool): True if the resource is online
            product (str): Plex product (Plex Media Server, Plex for iOS, Plex Web, etc.)
            productVersion (str): Version of the product.
            provides (str): List of services this resource provides (client, server,
                player, pubsub-player, etc.)
            publicAddressMatches (bool): True if the public IP address matches the client's public IP address.
            relay (bool): True if this resource has the Plex Relay enabled.
            sourceTitle (str): Username of the user that owns this resource (shared resources only).
            synced (bool): Unknown (possibly True if the resource has synced content?)
    """
    TAG = 'resource'
    key = 'https://plex.tv/api/v2/resources?includeHttps=1&includeRelay=1'
    DEFAULT_LOCATION_ORDER = ['local', 'remote', 'relay']
    DEFAULT_SCHEME_ORDER = ['https', 'http']

    def _loadData(self, data):
        if False:
            i = 10
            return i + 15
        self._data = data
        self.accessToken = logfilter.add_secret(data.attrib.get('accessToken'))
        self.clientIdentifier = data.attrib.get('clientIdentifier')
        self.connections = self.findItems(data, ResourceConnection, rtag='connections')
        self.createdAt = utils.toDatetime(data.attrib.get('createdAt'), '%Y-%m-%dT%H:%M:%SZ')
        self.device = data.attrib.get('device')
        self.dnsRebindingProtection = utils.cast(bool, data.attrib.get('dnsRebindingProtection'))
        self.home = utils.cast(bool, data.attrib.get('home'))
        self.httpsRequired = utils.cast(bool, data.attrib.get('httpsRequired'))
        self.lastSeenAt = utils.toDatetime(data.attrib.get('lastSeenAt'), '%Y-%m-%dT%H:%M:%SZ')
        self.name = data.attrib.get('name')
        self.natLoopbackSupported = utils.cast(bool, data.attrib.get('natLoopbackSupported'))
        self.owned = utils.cast(bool, data.attrib.get('owned'))
        self.ownerId = utils.cast(int, data.attrib.get('ownerId', 0))
        self.platform = data.attrib.get('platform')
        self.platformVersion = data.attrib.get('platformVersion')
        self.presence = utils.cast(bool, data.attrib.get('presence'))
        self.product = data.attrib.get('product')
        self.productVersion = data.attrib.get('productVersion')
        self.provides = data.attrib.get('provides')
        self.publicAddressMatches = utils.cast(bool, data.attrib.get('publicAddressMatches'))
        self.relay = utils.cast(bool, data.attrib.get('relay'))
        self.sourceTitle = data.attrib.get('sourceTitle')
        self.synced = utils.cast(bool, data.attrib.get('synced'))

    def preferred_connections(self, ssl=None, locations=None, schemes=None):
        if False:
            i = 10
            return i + 15
        ' Returns a sorted list of the available connection addresses for this resource.\n            Often times there is more than one address specified for a server or client.\n            Default behavior will prioritize local connections before remote or relay and HTTPS before HTTP.\n\n            Parameters:\n                ssl (bool, optional): Set True to only connect to HTTPS connections. Set False to\n                    only connect to HTTP connections. Set None (default) to connect to any\n                    HTTP or HTTPS connection.\n        '
        if locations is None:
            locations = self.DEFAULT_LOCATION_ORDER[:]
        if schemes is None:
            schemes = self.DEFAULT_SCHEME_ORDER[:]
        connections_dict = {location: {scheme: [] for scheme in schemes} for location in locations}
        for connection in self.connections:
            if self.owned or (not self.owned and (not connection.local)):
                location = 'relay' if connection.relay else 'local' if connection.local else 'remote'
                if location not in locations:
                    continue
                if 'http' in schemes:
                    connections_dict[location]['http'].append(connection.httpuri)
                if 'https' in schemes:
                    connections_dict[location]['https'].append(connection.uri)
        if ssl is True:
            schemes.remove('http')
        elif ssl is False:
            schemes.remove('https')
        connections = []
        for location in locations:
            for scheme in schemes:
                connections.extend(connections_dict[location][scheme])
        return connections

    def connect(self, ssl=None, timeout=None, locations=None, schemes=None):
        if False:
            while True:
                i = 10
        ' Returns a new :class:`~plexapi.server.PlexServer` or :class:`~plexapi.client.PlexClient` object.\n            Uses `MyPlexResource.preferred_connections()` to generate the priority order of connection addresses.\n            After trying to connect to all available addresses for this resource and\n            assuming at least one connection was successful, the PlexServer object is built and returned.\n\n            Parameters:\n                ssl (bool, optional): Set True to only connect to HTTPS connections. Set False to\n                    only connect to HTTP connections. Set None (default) to connect to any\n                    HTTP or HTTPS connection.\n                timeout (int, optional): The timeout in seconds to attempt each connection.\n\n            Raises:\n                :exc:`~plexapi.exceptions.NotFound`: When unable to connect to any addresses for this resource.\n        '
        if locations is None:
            locations = self.DEFAULT_LOCATION_ORDER[:]
        if schemes is None:
            schemes = self.DEFAULT_SCHEME_ORDER[:]
        connections = self.preferred_connections(ssl, locations, schemes)
        cls = PlexServer if 'server' in self.provides else PlexClient
        listargs = [[cls, url, self.accessToken, self._server._session, timeout] for url in connections]
        log.debug('Testing %s resource connections..', len(listargs))
        results = utils.threaded(_connect, listargs)
        return _chooseConnection('Resource', self.name, results)

class ResourceConnection(PlexObject):
    """ Represents a Resource Connection object found within the
        :class:`~plexapi.myplex.MyPlexResource` objects.

        Attributes:
            TAG (str): 'Connection'
            address (str): The connection IP address
            httpuri (str): Full HTTP URL
            ipv6 (bool): True if the address is IPv6
            local (bool): True if the address is local
            port (int): The connection port
            protocol (str): HTTP or HTTPS
            relay (bool): True if the address uses the Plex Relay
            uri (str): Full connetion URL
    """
    TAG = 'connection'

    def _loadData(self, data):
        if False:
            i = 10
            return i + 15
        self._data = data
        self.address = data.attrib.get('address')
        self.ipv6 = utils.cast(bool, data.attrib.get('IPv6'))
        self.local = utils.cast(bool, data.attrib.get('local'))
        self.port = utils.cast(int, data.attrib.get('port'))
        self.protocol = data.attrib.get('protocol')
        self.relay = utils.cast(bool, data.attrib.get('relay'))
        self.uri = data.attrib.get('uri')
        self.httpuri = f'http://{self.address}:{self.port}'

class MyPlexDevice(PlexObject):
    """ This object represents resources connected to your Plex server that provide
        playback ability from your Plex Server, iPhone or Android clients, Plex Web,
        this API, etc. The raw xml for the data presented here can be found at:
        https://plex.tv/devices.xml

        Attributes:
            TAG (str): 'Device'
            key (str): 'https://plex.tv/devices.xml'
            clientIdentifier (str): Unique ID for this resource.
            connections (list): List of connection URIs for the device.
            device (str): Best guess on the type of device this is (Linux, iPad, AFTB, etc).
            id (str): MyPlex ID of the device.
            model (str): Model of the device (bueller, Linux, x86_64, etc.)
            name (str): Hostname of the device.
            platform (str): OS the resource is running (Linux, Windows, Chrome, etc.)
            platformVersion (str): Version of the platform.
            product (str): Plex product (Plex Media Server, Plex for iOS, Plex Web, etc.)
            productVersion (string): Version of the product.
            provides (str): List of services this resource provides (client, controller,
                sync-target, player, pubsub-player).
            publicAddress (str): Public IP address.
            screenDensity (str): Unknown
            screenResolution (str): Screen resolution (750x1334, 1242x2208, etc.)
            token (str): Plex authentication token for the device.
            vendor (str): Device vendor (ubuntu, etc).
            version (str): Unknown (1, 2, 1.3.3.3148-b38628e, 1.3.15, etc.)
    """
    TAG = 'Device'
    key = 'https://plex.tv/devices.xml'

    def _loadData(self, data):
        if False:
            for i in range(10):
                print('nop')
        self._data = data
        self.name = data.attrib.get('name')
        self.publicAddress = data.attrib.get('publicAddress')
        self.product = data.attrib.get('product')
        self.productVersion = data.attrib.get('productVersion')
        self.platform = data.attrib.get('platform')
        self.platformVersion = data.attrib.get('platformVersion')
        self.device = data.attrib.get('device')
        self.model = data.attrib.get('model')
        self.vendor = data.attrib.get('vendor')
        self.provides = data.attrib.get('provides')
        self.clientIdentifier = data.attrib.get('clientIdentifier')
        self.version = data.attrib.get('version')
        self.id = data.attrib.get('id')
        self.token = logfilter.add_secret(data.attrib.get('token'))
        self.screenResolution = data.attrib.get('screenResolution')
        self.screenDensity = data.attrib.get('screenDensity')
        self.createdAt = utils.toDatetime(data.attrib.get('createdAt'))
        self.lastSeenAt = utils.toDatetime(data.attrib.get('lastSeenAt'))
        self.connections = self.listAttrs(data, 'uri', etag='Connection')

    def connect(self, timeout=None):
        if False:
            return 10
        ' Returns a new :class:`~plexapi.client.PlexClient` or :class:`~plexapi.server.PlexServer`\n            Sometimes there is more than one address specified for a server or client.\n            After trying to connect to all available addresses for this client and assuming\n            at least one connection was successful, the PlexClient object is built and returned.\n\n            Raises:\n                :exc:`~plexapi.exceptions.NotFound`: When unable to connect to any addresses for this device.\n        '
        cls = PlexServer if 'server' in self.provides else PlexClient
        listargs = [[cls, url, self.token, self._server._session, timeout] for url in self.connections]
        log.debug('Testing %s device connections..', len(listargs))
        results = utils.threaded(_connect, listargs)
        return _chooseConnection('Device', self.name, results)

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        ' Remove this device from your account. '
        key = f'https://plex.tv/devices/{self.id}.xml'
        self._server.query(key, self._server._session.delete)

    def syncItems(self):
        if False:
            print('Hello World!')
        " Returns an instance of :class:`~plexapi.sync.SyncList` for current device.\n\n            Raises:\n                :exc:`~plexapi.exceptions.BadRequest`: when the device doesn't provides `sync-target`.\n        "
        if 'sync-target' not in self.provides:
            raise BadRequest('Requested syncList for device which do not provides sync-target')
        return self._server.syncItems(client=self)

class MyPlexPinLogin:
    """
        MyPlex PIN login class which supports getting the four character PIN which the user must
        enter on https://plex.tv/link to authenticate the client and provide an access token to
        create a :class:`~plexapi.myplex.MyPlexAccount` instance.
        This helper class supports a polling, threaded and callback approach.

        - The polling approach expects the developer to periodically check if the PIN login was
          successful using :func:`~plexapi.myplex.MyPlexPinLogin.checkLogin`.
        - The threaded approach expects the developer to call
          :func:`~plexapi.myplex.MyPlexPinLogin.run` and then at a later time call
          :func:`~plexapi.myplex.MyPlexPinLogin.waitForLogin` to wait for and check the result.
        - The callback approach is an extension of the threaded approach and expects the developer
          to pass the `callback` parameter to the call to :func:`~plexapi.myplex.MyPlexPinLogin.run`.
          The callback will be called when the thread waiting for the PIN login to succeed either
          finishes or expires. The parameter passed to the callback is the received authentication
          token or `None` if the login expired.

        Parameters:
            session (requests.Session, optional): Use your own session object if you want to
                cache the http responses from PMS
            requestTimeout (int): timeout in seconds on initial connect to plex.tv (default config.TIMEOUT).
            headers (dict): A dict of X-Plex headers to send with requests.
            oauth (bool): True to use Plex OAuth instead of PIN login.

        Attributes:
            PINS (str): 'https://plex.tv/api/v2/pins'
            CHECKPINS (str): 'https://plex.tv/api/v2/pins/{pinid}'
            POLLINTERVAL (int): 1
            finished (bool): Whether the pin login has finished or not.
            expired (bool): Whether the pin login has expired or not.
            token (str): Token retrieved through the pin login.
            pin (str): Pin to use for the login on https://plex.tv/link.
    """
    PINS = 'https://plex.tv/api/v2/pins'
    CHECKPINS = 'https://plex.tv/api/v2/pins/{pinid}'
    POLLINTERVAL = 1

    def __init__(self, session=None, requestTimeout=None, headers=None, oauth=False):
        if False:
            i = 10
            return i + 15
        super(MyPlexPinLogin, self).__init__()
        self._session = session or requests.Session()
        self._requestTimeout = requestTimeout or TIMEOUT
        self.headers = headers
        self._oauth = oauth
        self._loginTimeout = None
        self._callback = None
        self._thread = None
        self._abort = False
        self._id = None
        self._code = None
        self._getCode()
        self.finished = False
        self.expired = False
        self.token = None

    @property
    def pin(self):
        if False:
            return 10
        ' Return the 4 character PIN used for linking a device at https://plex.tv/link. '
        if self._oauth:
            raise BadRequest('Cannot use PIN for Plex OAuth login')
        return self._code

    def oauthUrl(self, forwardUrl=None):
        if False:
            while True:
                i = 10
        ' Return the Plex OAuth url for login.\n\n            Parameters:\n                forwardUrl (str, optional): The url to redirect the client to after login.\n        '
        if not self._oauth:
            raise BadRequest('Must use "MyPlexPinLogin(oauth=True)" for Plex OAuth login.')
        headers = self._headers()
        params = {'clientID': headers['X-Plex-Client-Identifier'], 'context[device][product]': headers['X-Plex-Product'], 'context[device][version]': headers['X-Plex-Version'], 'context[device][platform]': headers['X-Plex-Platform'], 'context[device][platformVersion]': headers['X-Plex-Platform-Version'], 'context[device][device]': headers['X-Plex-Device'], 'context[device][deviceName]': headers['X-Plex-Device-Name'], 'code': self._code}
        if forwardUrl:
            params['forwardUrl'] = forwardUrl
        return f'https://app.plex.tv/auth/#!?{urlencode(params)}'

    def run(self, callback=None, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        ' Starts the thread which monitors the PIN login state.\n            Parameters:\n                callback (Callable[str]): Callback called with the received authentication token (optional).\n                timeout (int): Timeout in seconds waiting for the PIN login to succeed (optional).\n\n            Raises:\n                :class:`RuntimeError`: If the thread is already running.\n                :class:`RuntimeError`: If the PIN login for the current PIN has expired.\n        '
        if self._thread and (not self._abort):
            raise RuntimeError('MyPlexPinLogin thread is already running')
        if self.expired:
            raise RuntimeError('MyPlexPinLogin has expired')
        self._loginTimeout = timeout
        self._callback = callback
        self._abort = False
        self.finished = False
        self._thread = threading.Thread(target=self._pollLogin, name='plexapi.myplex.MyPlexPinLogin')
        self._thread.start()

    def waitForLogin(self):
        if False:
            return 10
        ' Waits for the PIN login to succeed or expire.\n            Parameters:\n                callback (Callable[str]): Callback called with the received authentication token (optional).\n                timeout (int): Timeout in seconds waiting for the PIN login to succeed (optional).\n\n            Returns:\n                `True` if the PIN login succeeded or `False` otherwise.\n        '
        if not self._thread or self._abort:
            return False
        self._thread.join()
        if self.expired or not self.token:
            return False
        return True

    def stop(self):
        if False:
            while True:
                i = 10
        ' Stops the thread monitoring the PIN login state. '
        if not self._thread or self._abort:
            return
        self._abort = True
        self._thread.join()

    def checkLogin(self):
        if False:
            while True:
                i = 10
        ' Returns `True` if the PIN login has succeeded. '
        if self._thread:
            return False
        try:
            return self._checkLogin()
        except Exception:
            self.expired = True
            self.finished = True
        return False

    def _getCode(self):
        if False:
            return 10
        url = self.PINS
        if self._oauth:
            params = {'strong': True}
        else:
            params = None
        response = self._query(url, self._session.post, params=params)
        if response is None:
            return None
        self._id = response.attrib.get('id')
        self._code = response.attrib.get('code')
        return self._code

    def _checkLogin(self):
        if False:
            print('Hello World!')
        if not self._id:
            return False
        if self.token:
            return True
        url = self.CHECKPINS.format(pinid=self._id)
        response = self._query(url)
        if response is None:
            return False
        token = response.attrib.get('authToken')
        if not token:
            return False
        self.token = token
        self.finished = True
        return True

    def _pollLogin(self):
        if False:
            print('Hello World!')
        try:
            start = time.time()
            while not self._abort and (not self._loginTimeout or time.time() - start < self._loginTimeout):
                try:
                    result = self._checkLogin()
                except Exception:
                    self.expired = True
                    break
                if result:
                    break
                time.sleep(self.POLLINTERVAL)
            if self.token and self._callback:
                self._callback(self.token)
        finally:
            self.finished = True

    def _headers(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Returns dict containing base headers for all requests for pin login. '
        headers = BASE_HEADERS.copy()
        if self.headers:
            headers.update(self.headers)
        headers.update(kwargs)
        return headers

    def _query(self, url, method=None, headers=None, **kwargs):
        if False:
            i = 10
            return i + 15
        method = method or self._session.get
        log.debug('%s %s', method.__name__.upper(), url)
        headers = headers or self._headers()
        response = method(url, headers=headers, timeout=self._requestTimeout, **kwargs)
        if not response.ok:
            codename = codes.get(response.status_code)[0]
            errtext = response.text.replace('\n', ' ')
            raise BadRequest(f'({response.status_code}) {codename} {response.url}; {errtext}')
        data = response.text.encode('utf8')
        return ElementTree.fromstring(data) if data.strip() else None

def _connect(cls, url, token, session, timeout, results, i, job_is_done_event=None):
    if False:
        for i in range(10):
            print('nop')
    " Connects to the specified cls with url and token. Stores the connection\n        information to results[i] in a threadsafe way.\n\n        Arguments:\n            cls: the class which is responsible for establishing connection, basically it's\n                 :class:`~plexapi.client.PlexClient` or :class:`~plexapi.server.PlexServer`\n            url (str): url which should be passed as `baseurl` argument to cls.__init__()\n            session (requests.Session): session which sould be passed as `session` argument to cls.__init()\n            token (str): authentication token which should be passed as `baseurl` argument to cls.__init__()\n            timeout (int): timeout which should be passed as `baseurl` argument to cls.__init__()\n            results (list): pre-filled list for results\n            i (int): index of current job, should be less than len(results)\n            job_is_done_event (:class:`~threading.Event`): is X_PLEX_ENABLE_FAST_CONNECT is True then the\n                  event would be set as soon the connection is established\n    "
    starttime = time.time()
    try:
        device = cls(baseurl=url, token=token, session=session, timeout=timeout)
        runtime = int(time.time() - starttime)
        results[i] = (url, token, device, runtime)
        if X_PLEX_ENABLE_FAST_CONNECT and job_is_done_event:
            job_is_done_event.set()
    except Exception as err:
        runtime = int(time.time() - starttime)
        log.error('%s: %s', url, err)
        results[i] = (url, token, None, runtime)

def _chooseConnection(ctype, name, results):
    if False:
        print('Hello World!')
    ' Chooses the first (best) connection from the given _connect results. '
    for (url, token, result, runtime) in results:
        okerr = 'OK' if result else 'ERR'
        log.debug('%s connection %s (%ss): %s?X-Plex-Token=%s', ctype, okerr, runtime, url, token)
    results = [r[2] for r in results if r and r[2] is not None]
    if results:
        log.debug('Connecting to %s: %s?X-Plex-Token=%s', ctype, results[0]._baseurl, results[0]._token)
        return results[0]
    raise NotFound(f'Unable to connect to {ctype.lower()}: {name}')

class AccountOptOut(PlexObject):
    """ Represents a single AccountOptOut
        'https://plex.tv/api/v2/user/{userUUID}/settings/opt_outs'

        Attributes:
            TAG (str): optOut
            key (str): Online Media Source key
            value (str): Online Media Source opt_in, opt_out, or opt_out_managed
    """
    TAG = 'optOut'
    CHOICES = {'opt_in', 'opt_out', 'opt_out_managed'}

    def _loadData(self, data):
        if False:
            while True:
                i = 10
        self.key = data.attrib.get('key')
        self.value = data.attrib.get('value')

    def _updateOptOut(self, option):
        if False:
            print('Hello World!')
        ' Sets the Online Media Sources option.\n\n            Parameters:\n                option (str): see CHOICES\n\n            Raises:\n                :exc:`~plexapi.exceptions.NotFound`: ``option`` str not found in CHOICES.\n        '
        if option not in self.CHOICES:
            raise NotFound(f'{option} not found in available choices: {self.CHOICES}')
        url = self._server.OPTOUTS.format(userUUID=self._server.uuid)
        params = {'key': self.key, 'value': option}
        self._server.query(url, method=self._server._session.post, params=params)
        self.value = option

    def optIn(self):
        if False:
            while True:
                i = 10
        ' Sets the Online Media Source to "Enabled". '
        self._updateOptOut('opt_in')

    def optOut(self):
        if False:
            return 10
        ' Sets the Online Media Source to "Disabled". '
        self._updateOptOut('opt_out')

    def optOutManaged(self):
        if False:
            for i in range(10):
                print('nop')
        ' Sets the Online Media Source to "Disabled for Managed Users".\n\n            Raises:\n                :exc:`~plexapi.exceptions.BadRequest`: When trying to opt out music.\n        '
        if self.key == 'tv.plex.provider.music':
            raise BadRequest(f'{self.key} does not have the option to opt out managed users.')
        self._updateOptOut('opt_out_managed')

class UserState(PlexObject):
    """ Represents a single UserState

        Attributes:
            TAG (str): UserState
            lastViewedAt (datetime): Datetime the item was last played.
            ratingKey (str): Unique key identifying the item.
            type (str): The media type of the item.
            viewCount (int): Count of times the item was played.
            viewedLeafCount (int): Number of items marked as played in the show/season.
            viewOffset (int): Time offset in milliseconds from the start of the content
            viewState (bool): True or False if the item has been played.
            watchlistedAt (datetime): Datetime the item was added to the watchlist.
    """
    TAG = 'UserState'

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'<{self.__class__.__name__}:{self.ratingKey}>'

    def _loadData(self, data):
        if False:
            i = 10
            return i + 15
        self.lastViewedAt = utils.toDatetime(data.attrib.get('lastViewedAt'))
        self.ratingKey = data.attrib.get('ratingKey')
        self.type = data.attrib.get('type')
        self.viewCount = utils.cast(int, data.attrib.get('viewCount', 0))
        self.viewedLeafCount = utils.cast(int, data.attrib.get('viewedLeafCount', 0))
        self.viewOffset = utils.cast(int, data.attrib.get('viewOffset', 0))
        self.viewState = data.attrib.get('viewState') == 'complete'
        self.watchlistedAt = utils.toDatetime(data.attrib.get('watchlistedAt'))

class GeoLocation(PlexObject):
    """ Represents a signle IP address geolocation

        Attributes:
            TAG (str): location
            city (str): City name
            code (str): Country code
            continentCode (str): Continent code
            coordinates (Tuple<float>): Latitude and longitude
            country (str): Country name
            europeanUnionMember (bool): True if the country is a member of the European Union
            inPrivacyRestrictedCountry (bool): True if the country is privacy restricted
            postalCode (str): Postal code
            subdivisions (str): Subdivision name
            timezone (str): Timezone
    """
    TAG = 'location'

    def _loadData(self, data):
        if False:
            print('Hello World!')
        self._data = data
        self.city = data.attrib.get('city')
        self.code = data.attrib.get('code')
        self.continentCode = data.attrib.get('continent_code')
        self.coordinates = tuple((utils.cast(float, coord) for coord in (data.attrib.get('coordinates') or ',').split(',')))
        self.country = data.attrib.get('country')
        self.postalCode = data.attrib.get('postal_code')
        self.subdivisions = data.attrib.get('subdivisions')
        self.timezone = data.attrib.get('time_zone')
        europeanUnionMember = data.attrib.get('european_union_member')
        self.europeanUnionMember = False if europeanUnionMember == 'Unknown' else utils.cast(bool, europeanUnionMember)
        inPrivacyRestrictedCountry = data.attrib.get('in_privacy_restricted_country')
        self.inPrivacyRestrictedCountry = False if inPrivacyRestrictedCountry == 'Unknown' else utils.cast(bool, inPrivacyRestrictedCountry)