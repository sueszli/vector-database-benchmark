
//
// Copyright (C) 2009 Nortel, certain elements licensed under a Contributor Agreement.
// Contributors retain copyright to elements licensed under a Contributor Agreement.
// Licensed to the User under the LGPL license.
//
// $$
//////////////////////////////////////////////////////////////////////////////

// SYSTEM INCLUDES

// APPLICATION INCLUDES
#include <utl/UtlRegex.h>
#include <utl/UtlBool.h>
#include "net/SipXauthIdentity.h"
#include "net/XmlRpcRequest.h"
#include "net/XmlRpcMethod.h"
#include "net/XmlRpcDispatch.h"
#include "net/XmlRpcResponse.h"
#include "net/NetBase64Codec.h"
#include "sipXecsService/SipXecsService.h"
#include "sipXecsService/SharedSecret.h"
#include "os/OsDateTime.h"
#include "os/OsLock.h"
#include "os/OsLogger.h"
#include "sipdb/ResultSet.h"
#include "SipRedirectorPresenceRouting.h"
#include "registry/SipRedirectServer.h"
#include "sipdb/EntityDB.h"
#include "registry/SipRegistrar.h"

// DEFINES
#define CONFIG_SETTING_REALM "REALM"
#define CONFIG_SETTING_SIP_DOMAIN "SIP_DOMAIN"
#define CONFIG_SETTING_VOICEMAIL_ON_BUSY "VOICEMAIL_ON_BUSY"
#define CONFIG_SETTING_USER_PREFS_FILE "USER_PREFS_FILE"
#define CONFIG_OPENFIRE_PRESENCE_SERVER_URL "OPENFIRE_PRESENCE_SERVER_URL"
#define CONFIG_PRESENCE_MONITOR_SERVER_URL  "LOCAL_PRESENCE_MONITOR_SERVER_URL"
#define UNIFIED_PRESENCE_CHANGED_REQUEST_NAME "UnifiedPresenceChangeListener.unifiedPresenceChanged"

#define OPENFIRE_PING_TIMER_IN_SECS  (20)

#define VOICEMAIL_CONTACT_PREFIX ("~~vm~")

#define QUERY_PRESENCE_METHOD            "presenceServer.getUnifiedPresenceInfo"
#define REGISTER_PRESENCE_MONITOR_METHOD "presenceServer.registerPresenceMonitor"
#define PING_METHOD                      "presenceServer.ping"
#define XML_RPC_PROTOCOL                 "xmlrpc"

// sipXopenfire presence plug-in  XML-RPC response member names
#define STATUS_CODE "status-code"   // string representing status of request "ERROR" or "OK"
#define INSTANCE_HANDLE "instance-handle"
#define NIL_INSTANCE_HANDLE_VALUE  "nil_handle"
#define STATUS_CODE_VALUE_OK    "OK"
#define STATUS_CODE_VALUE_ERROR "ERROR"
#define ERROR_CODE  "error-code"     // int providing a detailed error code.  Provided only when "status-code" is "ERROR"
#define ERROR_INFO  "error-string"  // string giving a plaintext explaination of the error. Provided only when "status-code" is "ERROR"
#define TELEPHONY_PRESENCE "sip-presence" // string representing telephony presence.  Can be "idle", "busy" or "undetermied"
#define XMPP_PRESENCE "xmpp-presence" // string representing XMPP presence.  Can be "available", "away", "xa", "dnd" or "offline"
#define UNIFIED_PRESENCE "unified-presence" // string representing unified XMPP presence which is a merge of the telephony and XMPP presences
#define CUSTOM_PRESENCE_MESSAGE "custom-presence-message" // string representing user-supplied cusomt message linked to presence state.  Can be an empty string
#define SIP_RESOURCE_ID "sipId"
#define JABBER_ID "jabber-id"

UnifiedPresence::UnifiedPresence( UtlString aor ) :
    UtlString( aor )
{
}

const UtlString& UnifiedPresence::getXmppPresence( void ) const
{
    return mXmppPresence;
}

void UnifiedPresence::setXmppPresence( const UtlString& xmppPresence )
{
    mXmppPresence = xmppPresence;
}

const UtlString& UnifiedPresence::getXmppStatusMessage( void ) const
{
    return mXmppStatusMessage;
}

void UnifiedPresence::setXmppStatusMessage( const UtlString& xmppStatusMessage )
{
    mXmppStatusMessage = xmppStatusMessage;
}

const UtlString& UnifiedPresence::getSipState( void ) const
{
    return mSipState;
}

void UnifiedPresence::setSipState( const UtlString& sipState )
{
    mSipState = sipState;
}

const UtlString& UnifiedPresence::getUnifiedPresence( void ) const
{
    return mUnifiedPresence;
}

void UnifiedPresence::setUnifiedPresence( const UtlString& unifiedPresence )
{
    mUnifiedPresence = unifiedPresence;
}

UnifiedPresenceContainer* UnifiedPresenceContainer::pInstance = 0;

UtlString SipRedirectorPresenceRouting::sLocalDomain;

UnifiedPresenceContainer* UnifiedPresenceContainer::getInstance( void )
{
    if( pInstance == 0 )
    {
        pInstance = new UnifiedPresenceContainer();
    }
    return pInstance;
}

void UnifiedPresenceContainer::insert( UtlString* pAor, UnifiedPresence* pUnifiedPresence )
{
    OsLock lock( mMutex );
    UnifiedPresence* pOldUnifiedPresence = (UnifiedPresence*)mUnifiedPresences.remove( pAor );
    if (pOldUnifiedPresence)
      delete pOldUnifiedPresence;

    mUnifiedPresences.insertKeyAndValue( pAor, pUnifiedPresence );
    Os::Logger::instance().log(FAC_SIP, PRI_DEBUG, "UnifiedPresenceContainer::insert "
                  "Presence information for '%s':\r\n"
                  "    Telephony presence: '%s'"
                  "    XMPP presence: '%s'"
                  "    Custom presence message: '%s'"
                  "    Unified message: '%s'",
                  pAor->data(), pUnifiedPresence->getSipState().data(),
                  pUnifiedPresence->getXmppPresence().data(),
                  pUnifiedPresence->getXmppStatusMessage().data(),
                  pUnifiedPresence->getUnifiedPresence().data());
}

const UnifiedPresence* UnifiedPresenceContainer::lookup( UtlString* pAor )
{
    OsLock lock( mMutex );
    return (UnifiedPresence* )mUnifiedPresences.findValue( pAor );

}

void UnifiedPresenceContainer::reset( void )
{
    OsLock lock( mMutex );
    mUnifiedPresences.destroyAll();
}

UnifiedPresenceContainer::UnifiedPresenceContainer() :
    mMutex( OsMutex::Q_FIFO )
{
}


// MACROS
// EXTERNAL FUNCTIONS
// EXTERNAL VARIABLES
// CONSTANTS
// STRUCTS
// TYPEDEFS
// FORWARD DECLARATIONS


const RegEx TelUri( "tel:([\\+\\.\\(\\)0-9\\-]+?)(,|\\s|\\Z)" );
const RegEx SipUri( "(sip:.+@.+?)(,|\\s|\\Z)" );

// Static factory function.
extern "C" RedirectPlugin* getRedirectPlugin(const UtlString& instanceName)
{
    return new SipRedirectorPresenceRouting(instanceName);
}

// Constructor
SipRedirectorPresenceRouting::SipRedirectorPresenceRouting(const UtlString& instanceName) :
    RedirectPlugin(instanceName),
    mPingTimer( *this ),
    mbRegisteredWithOpenfire( false ),
    mOpenfireInstanceHandle(NIL_INSTANCE_HANDLE_VALUE)
{
    mLogName.append("[");
    mLogName.append(instanceName);
    mLogName.append("] SipRedirectorPresenceRouting");
}

// Destructor
SipRedirectorPresenceRouting::~SipRedirectorPresenceRouting()
{
   mPingTimer.stop();
}

// Read config information.
void SipRedirectorPresenceRouting::readConfig(OsConfigDb& configDb)
{
    // extract the realm information from the config DB - we need this part
    // to do the credentials db look-up.
    if ((configDb.get(CONFIG_SETTING_REALM, mRealm) != OS_SUCCESS) ||
         mRealm.isNull())
    {
        Os::Logger::instance().log(FAC_SIP, PRI_ERR,
                     "%s::readConfig No Realm specified in the configuration",
                     mLogName.data());
    }
    else
    {
        Os::Logger::instance().log(FAC_SIP, PRI_INFO,
                    "%s::readConfig mRealm = '%s'",
                    mLogName.data(), mRealm.data() );
    }

    // extract the SIP domain information from the config DB - we need this part
    // to transform the tel URI into a SIP URI
    if ((configDb.get(CONFIG_SETTING_SIP_DOMAIN, mSipDomain) != OS_SUCCESS) ||
         mSipDomain.isNull())
    {
        Os::Logger::instance().log(FAC_SIP, PRI_ERR,
                     "%s::readConfig No SIP domain specified in the configuration",
                     mLogName.data());
    }
    else
    {
        Os::Logger::instance().log(FAC_SIP, PRI_INFO,
                    "%s::readConfig mSipDomain = '%s'",
                    mLogName.data(), mSipDomain.data() );
    }

    mbForwardToVmOnBusy = configDb.getBoolean(CONFIG_SETTING_VOICEMAIL_ON_BUSY, FALSE);
    Os::Logger::instance().log(FAC_SIP, PRI_INFO,
                 "%s::readConfig mbForwardToVmOnBusy = %d",
                 mLogName.data(), mbForwardToVmOnBusy);

    
    UtlString prefsFilename;
    configDb.get(CONFIG_SETTING_USER_PREFS_FILE, prefsFilename);
    Os::Logger::instance().log(FAC_SIP, PRI_INFO,
                 "%s::readConfig prefsFilename = %s",
                 mLogName.data(), prefsFilename.data());
    mUserPrefs.setFileName( &prefsFilename );
    
    UtlString openFirePresenceServerUrlAsString;
    if ((configDb.get(CONFIG_OPENFIRE_PRESENCE_SERVER_URL, openFirePresenceServerUrlAsString) != OS_SUCCESS) ||
          openFirePresenceServerUrlAsString.isNull())
    {
       Os::Logger::instance().log(FAC_SIP, PRI_ERR,
                     "%s::readConfig No URL specified for openfire presence server in the configuration",
                     mLogName.data());
    }
    else
    {
        Os::Logger::instance().log(FAC_SIP, PRI_INFO,
                    "%s::readConfig openFirePresenceServerUrlAsString = '%s'",
                    mLogName.data(), openFirePresenceServerUrlAsString.data() );
        mOpenFirePresenceServerUrl.fromString( openFirePresenceServerUrlAsString );
    }

    UtlString presenceMonitorServerUrlAsString;
    if ((configDb.get(CONFIG_PRESENCE_MONITOR_SERVER_URL, presenceMonitorServerUrlAsString) != OS_SUCCESS) ||
           presenceMonitorServerUrlAsString.isNull())
    {
         Os::Logger::instance().log(FAC_SIP, PRI_ERR,
                       "%s::readConfig No URL specified for local presence monitor server in the configuration",
                       mLogName.data());
    }
    else
    {
        Os::Logger::instance().log(FAC_SIP, PRI_INFO,
                      "%s::readConfig presenceMonitorServerUrlAsString = '%s'",
                      mLogName.data(), presenceMonitorServerUrlAsString.data() );
        mLocalPresenceMonitorServerUrl.fromString( presenceMonitorServerUrlAsString );
    }

    // read the domain configuration
    OsConfigDb domainConfiguration;
    domainConfiguration.loadFromFile(SipXecsService::domainConfigPath());
   // get the shared secret for generating signatures
    SharedSecret secret(domainConfiguration);

    // Set secret for signing SipXauthIdentity
    SipXauthIdentity::setSecret(secret.data());

    // openfire incorrectly uses base64 encoded version of shared secret for validation, so send that here
    UtlString base64SharedSecret;
    domainConfiguration.get(SipXecsService::DomainDbKey::SHARED_SECRET, base64SharedSecret);
    SipRedirectorPresenceRouting::basicAuthCreds(mXmlApcApiCreds, "nil", base64SharedSecret.data());
}

void
SipRedirectorPresenceRouting::basicAuthCreds(std::string& creds, const char *user, const char *sharedSecret)
{
  std::string unencoded;
  unencoded.append(user).append(":").append(sharedSecret);
  int encodedSize = NetBase64Codec::encodedSize(unencoded.length());
  char* encoded = new char[encodedSize];
  NetBase64Codec::encode(unencoded.length(), unencoded.c_str(), encodedSize, encoded);
  creds.append("Basic ").append(encoded, encodedSize);
  delete encoded;
}


// Initialize
OsStatus
SipRedirectorPresenceRouting::initialize(OsConfigDb& configDb,
                                 int redirectorNo,
                                 const UtlString& localDomainName)
{
   Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                 "%s::SipRedirectorPresenceRouting::initialize", mLogName.data() );

   OsStatus rc = OS_FAILED;
  
   sLocalDomain = localDomainName;
   
   if( startPresenceMonitorXmlRpcServer() == OS_SUCCESS )
   {
	   if( registerPresenceMonitorServerWithOpenfire() == OS_SUCCESS )
	   {
	      mbRegisteredWithOpenfire = true;
	   }
      OsTime pingTimerPeriod( OPENFIRE_PING_TIMER_IN_SECS, 0 );
      mPingTimer.periodicEvery( pingTimerPeriod, pingTimerPeriod );
      rc = OS_SUCCESS;
   }
   return rc;
}


// Finalize
void
SipRedirectorPresenceRouting::finalize()
{
}

RedirectPlugin::LookUpStatus
SipRedirectorPresenceRouting::lookUp(
   const SipMessage& message,
   UtlString& requestString,
   Url& requestUri,
   const UtlString& method,
   ContactList& contactList,
   RequestSeqNo requestSeqNo,
   int redirectorNo,
   SipRedirectorPrivateStorage*& privateStorage,
   ErrorDescriptor& errorDescriptor)
{
   RedirectPlugin::LookUpStatus rc = RedirectPlugin::SUCCESS;

   // this is a new request that we need to process.
   // If the called party is a local user,
   // query the unified presence container to find out
   // the presence state of the called party.

   UtlString realm;
   UtlString authType;

   // If the request URI can be found in the identity column of the credentials
   // database, then a request is to a local user, find out its presence state...
   EntityRecord entity;
   EntityDB* entityDb = SipRegistrar::getInstance(NULL)->getEntityDB();
   if (entityDb->findByIdentity(requestUri, entity))
   {
       realm = entity.realm().c_str();
       authType = entity.authType().c_str();
       
      // If url param sipx-userforward = false, do not redirect...
      UtlString disableForwarding;
      requestUri.getUrlParameter("sipx-userforward", disableForwarding);
      if (disableForwarding.compareTo("false", UtlString::ignoreCase) == 0)
      {
         Os::Logger::instance().log(FAC_SIP, PRI_DEBUG, "%s::lookUp user forwarding disabled by parameter",
                       mLogName.data());
      }
      else
      {
         Os::Logger::instance().log(FAC_SIP, PRI_DEBUG, "%s::lookUp identity '%s'",
                       mLogName.data(), requestString.data());
         rc = doLookUp( requestUri, message, contactList );
      }
   }
   return rc;
}

RedirectPlugin::LookUpStatus
SipRedirectorPresenceRouting::doLookUp(
   const Url& requestUri,
   const SipMessage& message,
   ContactList& contactList)
{
   // check if we have unified presence info for this user
   const UnifiedPresence* pUp;
   UtlString identity;
   UtlString username;
   requestUri.getIdentity( identity );
   requestUri.getUserId( username );
   Os::Logger::instance().log(FAC_SIP, PRI_INFO, "%s::LookUpStatus is looking up '%s'",
                                    mLogName.data(),identity.data() );
   pUp = UnifiedPresenceContainer::getInstance()->lookup( &identity );

   if( pUp )
   {
      // unified presence data is available for the called party.
      // Use it to make call routing decisions.
      Os::Logger::instance().log(FAC_SIP, PRI_INFO, "%s::LookUpStatus "
                                       "Presence information for '%s':\r\n"
                                       "    Telephony presence: '%s'"
                                       "    XMPP presence: '%s'"
                                       "    Custom presence message: '%s'",
                                       mLogName.data(),
                                       identity.data(), pUp->getSipState().data(),
                                       pUp->getXmppPresence().data(),
                                       pUp->getXmppStatusMessage().data() );

      // look for tel URI in the custom presence message
      RegEx telUri( TelUri );
      telUri.Search( pUp->getXmppStatusMessage().data() );
      UtlString target;
      if( !mSipDomain.isNull() && telUri.MatchString( &target, 1 ) )
      {
         // prepend 'sip:', append the SIP domain and add resulting target as contact 
         target.insert( 0, "sip:" );
         target.append( '@' );
         target.append( mSipDomain );
         addContactToContactList( target, requestUri, message, contactList );
      }
      else
      {
         // no tel URI - look for a sip URI then
         RegEx sipUri( SipUri );
         sipUri.Search( pUp->getXmppStatusMessage().data() );
         if( sipUri.MatchString( &target, 1 ) )
         {
            addContactToContactList( target, requestUri, message, contactList );
         }
         else
         {
            // no tel or sip URI, check to see if we need to send the call directly
            // to voicemail based on seetings, presence and user preferences
            if( ( pUp->getSipState().compareTo("BUSY", UtlString::ignoreCase ) == 0 && mbForwardToVmOnBusy ) ||
                ( pUp->getXmppPresence().compareTo("BUSY", UtlString::ignoreCase ) == 0 && mUserPrefs.forwardToVoicemailOnDnd( username ) ) )
            {
               // prune all non-voicemail contacts from the list
               removeNonVoicemailContacts( contactList );
            }
         }
      }   
   }
   return RedirectPlugin::SUCCESS;
}

void SipRedirectorPresenceRouting::addContactToContactList( const UtlString& target,
                                                            const Url& requestUri,
                                                            const SipMessage& message,
                                                            ContactList& contactList )
{
   UtlString requestIdentity;
   requestUri.getIdentity( requestIdentity );
   SipXauthIdentity authIdentity;
   authIdentity.setIdentity( requestIdentity );
   Url targetUri( target );
   // Encode AuthIdentity into the URI
   authIdentity.encodeUri(targetUri, message);
   contactList.add( targetUri, *this );
}

void SipRedirectorPresenceRouting::removeNonVoicemailContacts( ContactList& contactList )
{
   // walk the list to find the contact entry for voicemail
   size_t index;
   Url contactUrl;
   bool bVoicemailContactFound = false;
   for( index = 0; index < contactList.entries(); index++ )
   {
      if( contactList.get( index, contactUrl ) )
      {
         UtlString userPart;
         contactUrl.getUserId( userPart );
         if( userPart.index( VOICEMAIL_CONTACT_PREFIX ) == 0 )
         {
            bVoicemailContactFound = true;
            break;
         }
      }
   }

   // if vm contact found, remove all and put vm contact back in.
   if( bVoicemailContactFound )
   {
      contactList.removeAll( *this );
      contactList.add( contactUrl, *this );
   }
}

OsStatus SipRedirectorPresenceRouting::startPresenceMonitorXmlRpcServer( void )
{
    UtlString path;
    mLocalPresenceMonitorServerUrl.getPath( path );

    XmlRpcDispatch* pXmlRpcServer = new XmlRpcDispatch( mLocalPresenceMonitorServerUrl.getHostPort(),
                                                        mLocalPresenceMonitorServerUrl.getScheme() == Url::HttpsUrlScheme,
                                                        path );
    pXmlRpcServer->addMethod( UNIFIED_PRESENCE_CHANGED_REQUEST_NAME, UnifiedPresenceChangedMethod::get);
    return OS_SUCCESS;
}

OsStatus SipRedirectorPresenceRouting::registerPresenceMonitorServerWithOpenfire( void )
{
    OsStatus rc = OS_FAILED;
    UtlString presenceMonitorUrlAsString = mLocalPresenceMonitorServerUrl.toString();

    XmlRpcRequest registerPresenceMonitorRequest( mOpenFirePresenceServerUrl, REGISTER_PRESENCE_MONITOR_METHOD );
    registerPresenceMonitorRequest.setHeaderField("Authorization", mXmlApcApiCreds.c_str());
    UtlString protocol = XML_RPC_PROTOCOL;
    registerPresenceMonitorRequest.addParam( &protocol );
    registerPresenceMonitorRequest.addParam( &presenceMonitorUrlAsString );
    XmlRpcResponse registerPresenceMonitorResponse;
    if( registerPresenceMonitorRequest.execute( registerPresenceMonitorResponse ) == true )
    {
       UtlContainable* pValue = NULL;
       if ( !registerPresenceMonitorResponse.getResponse( pValue ) || !pValue )
       {
          Os::Logger::instance().log(FAC_NAT, PRI_ERR, "SipRedirectorPresenceRouting::registerPresenceMonitorRequestWithOpenfire response had no result.");
       }
       else
       {
          UtlString keyName;
          UtlHashMap* pMap = dynamic_cast<UtlHashMap*>( pValue );
          if ( !pMap )
          {
             Os::Logger::instance().log(FAC_NAT, PRI_ERR,
                           "SipRedirectorPresenceRouting::registerPresenceMonitorRequestWithOpenfire response result had unexpected type: %s",
                           pValue->getContainableType() );
          }
          else
          {
             // extract status code and check it.
             keyName = STATUS_CODE;
             UtlString* pStatusCode = dynamic_cast<UtlString*>( pMap->findValue( &keyName ) );
             if( pStatusCode->compareTo( STATUS_CODE_VALUE_OK, UtlString::ignoreCase ) != 0 )
             {
                // status-code is not "OK", some error happened - extract error information.
                keyName = ERROR_CODE;
                UtlString* pErrorCode = dynamic_cast<UtlString*>( pMap->findValue( &keyName ) );
                keyName = ERROR_INFO;
                UtlString* pErrorInfo = dynamic_cast<UtlString*>( pMap->findValue( &keyName ) );
                if( pErrorCode && pErrorInfo )
                {
                   Os::Logger::instance().log(FAC_NAT, PRI_ERR, "SipRedirectorPresenceRouting::registerPresenceMonitorRequestWithOpenfire request failed: %s:'%s'", pErrorCode->data(), pErrorInfo->data() );
                }
             }
             else
             {
                keyName = INSTANCE_HANDLE;
                UtlString* pInstanceHandle = dynamic_cast<UtlString*>( pMap->findValue( &keyName ) );
                if( pInstanceHandle )
                {
                   // set openfire instance value 
                   mOpenfireInstanceHandle = *pInstanceHandle;
                }
                rc = OS_SUCCESS;
             }
          }
       }
    }
    else
    {
       // Check if the request failed because of a failed connection.
       // That error can sometimes happen when the server closed the TCP
       // connection we were using to communicate to it and can usually

        // be recovered by sending the request again.  Try to send the
        // request once more to see if it will fly this time.
        int faultCode;
        UtlString faultString;
        registerPresenceMonitorResponse.getFault( &faultCode, faultString );
        Os::Logger::instance().log( FAC_NAT, PRI_ERR,
                       "SipRedirectorPresenceRouting::registerPresenceMonitorRequestWithOpenfire failed to execute() request: %d : %s",
                       faultCode, faultString.data() );
    }
    return rc;
}

OsStatus SipRedirectorPresenceRouting::pingOpenfire( void )
{
    OsStatus rc = OS_FAILED;
    UtlString presenceMonitorUrlAsString = mLocalPresenceMonitorServerUrl.toString();
    XmlRpcRequest pingRequest( mOpenFirePresenceServerUrl, PING_METHOD );
    pingRequest.setHeaderField("Authorization", mXmlApcApiCreds.c_str());
    pingRequest.addParam( &mLogName );
    XmlRpcResponse pingResponse;
    if( pingRequest.execute( pingResponse ) == true )
    {
       UtlContainable* pValue = NULL;
       if ( !pingResponse.getResponse( pValue ) || !pValue )
       {
          Os::Logger::instance().log(FAC_NAT, PRI_CRIT, "SipRedirectorPresenceRouting::pingOpenfire response had no result.");
       }
       else
       {
          UtlString keyName;
          UtlHashMap* pMap = dynamic_cast<UtlHashMap*>( pValue );
          if ( !pMap )
          {
             Os::Logger::instance().log(FAC_NAT, PRI_ERR,
                           "SipRedirectorPresenceRouting::pingOpenfire response result had unexpected type: %s",
                           pValue->getContainableType() );
          }
          else
          {
             // extract status code and check it.
             keyName = STATUS_CODE;
             UtlString* pStatusCode = dynamic_cast<UtlString*>( pMap->findValue( &keyName ) );
             if( pStatusCode->compareTo( STATUS_CODE_VALUE_OK, UtlString::ignoreCase ) == 0 )
             {
                keyName = INSTANCE_HANDLE;
                UtlString* pInstanceHandle = dynamic_cast<UtlString*>( pMap->findValue( &keyName ) );
                if( pInstanceHandle )
                {
                   if( mOpenfireInstanceHandle.compareTo( NIL_INSTANCE_HANDLE_VALUE ) == 0 )
                   {
                      // This is the first instance handle we get from openfire, save it
                      mOpenfireInstanceHandle = *pInstanceHandle;
                      rc = OS_SUCCESS;
                   }
                   else
                   {
                      // check if the openfire handle we received matches the one we have on record
                      if( mOpenfireInstanceHandle.compareTo( *pInstanceHandle ) == 0 )
                      {
                         // there is a match; everything is fine.
                         rc = OS_SUCCESS;
                      }
                      else
                      {
                         // mistmatch - likely caused by openfire resetting behind our back.
                         // fail the ping to cause a re-register with openfire
                         mOpenfireInstanceHandle = *pInstanceHandle;
                         rc = OS_FAILED; // rc already set to OS_FAILED; added for readability
                      }
                   }
                }
             }
          }
       }
    }
    return rc;
}

OsStatus SipRedirectorPresenceRouting::signal(intptr_t eventData)
{
   if( mbRegisteredWithOpenfire == false )
   {
      // we are not registered with openfire yet, attempt to register again
      if( registerPresenceMonitorServerWithOpenfire() == OS_SUCCESS )
      {
         mbRegisteredWithOpenfire = true;
         Os::Logger::instance().log(FAC_NAT, PRI_INFO, "%s::signal: reconnected with openfire", mLogName.data() );
      }
      else
      {
         Os::Logger::instance().log(FAC_NAT, PRI_ERR, "%s::signal: failed to reconnect with openfire - retrying later", mLogName.data() );
      }
   }
   else
   {
      // we think we are registered with openfire, try to ping it to ensure it is still there
      if( pingOpenfire() != OS_SUCCESS )
      {
         Os::Logger::instance().log(FAC_NAT, PRI_ERR, "%s::signal: ping to openfire failed - trying to reconnect", mLogName.data());
         // We lost the server.  Flush out the cached presence info as it is now stale.
         // We should not be basing routing decisions on it.
         UnifiedPresenceContainer::getInstance()->reset();
         mbRegisteredWithOpenfire = false;
      }
   }
   
   // Refresh the user presence preferences in case they changed
   mUserPrefs.refresh();
   return OS_SUCCESS;
}

const UtlString& SipRedirectorPresenceRouting::getLocalDomainName( void )
{
   return sLocalDomain;
}



const UtlString& SipRedirectorPresenceRouting::name( void ) const
{
   return mLogName;
}

XmlRpcMethod* UnifiedPresenceChangedMethod::get()
{
    return new UnifiedPresenceChangedMethod();
}


 bool UnifiedPresenceChangedMethod::execute(const HttpRequestContext& requestContext,
                      UtlSList& params,
                      void* userData,
                      XmlRpcResponse& response,
                      ExecutionStatus& status)
{
    status = XmlRpcMethod::FAILED;
    if( params.entries() == 6 )
    {
        // According to set XML-RPC i/f definition:
        //param[0] = XMPP Presence
        //param[1] = Jabber ID;
        //param[2] = SIP AOR;
        //param[3] = SIP State;
        //param[4] = Unified Presence;
        //param[5] = XMPP Status Message;
        UnifiedPresence* pUp = new  UnifiedPresence( ((UtlString*)params.at(2))->data() );
        pUp->setXmppPresence     ( ((UtlString*)params.at(0))->data() );
        pUp->setXmppStatusMessage( ((UtlString*)params.at(5))->data() );
        pUp->setSipState         ( ((UtlString*)params.at(3))->data() );
        pUp->setUnifiedPresence  ( ((UtlString*)params.at(4))->data() );

        UtlString* pAor = new UtlString( ((UtlString*)params.at(2))->data() );
        // make sure that the SIP AOR really has a domain part; if not, add it.
        if( pAor->index('@') == UTL_NOT_FOUND )
        {
           pAor->append( '@' );
           pAor->append( SipRedirectorPresenceRouting::getLocalDomainName() );
        }
        UnifiedPresenceContainer::getInstance()->insert( pAor, pUp );
    }
    UtlString responseString = "ok";
    response.setResponse( &responseString );
    status = XmlRpcMethod::OK;
    return true;
}

PresenceRoutingUserPreferences::PresenceRoutingUserPreferences() :
   mMutex( OsMutex::Q_FIFO )
{
}

OsStatus PresenceRoutingUserPreferences::initialize()
{
   OsStatus currentStatus = OS_SUCCESS;
   return currentStatus;
}

#if 0
OsStatus PresenceRoutingUserPreferences::parseDocument( TiXmlDocument* pDoc )
{
   OsStatus rc = OS_SUCCESS;
   OsLock lock( mMutex );
   TiXmlNode* presenceRoutingNode;
   mUserVmOnDndPreferences.destroyAll();
   if( (presenceRoutingNode = pDoc->FirstChild("presenceRoutingPrefs")) != NULL &&
         presenceRoutingNode->Type() == TiXmlNode::ELEMENT)
   {
      // Find all the <user> elements.
      for( TiXmlNode* userNode = 0;
      (userNode = presenceRoutingNode->IterateChildren( "user", userNode )); ) 
      {
         if (userNode->Type() == TiXmlNode::ELEMENT)
         {
            TiXmlNode* pChildNode;
            if( ( pChildNode = userNode->FirstChild( "userName" ) ) && pChildNode->FirstChild() )
            {
               UtlString* pUsername = new UtlString( pChildNode->FirstChild()->Value() );
               if( ( pChildNode = userNode->FirstChild( "vmOnDnd" ) ) && pChildNode->FirstChild() )
               {
                  UtlString vmOnDndAsString = pChildNode->FirstChild()->Value();
                  UtlBool* pbVmOnDnd = new UtlBool( FALSE );
                  if( vmOnDndAsString.compareTo("true", UtlString::ignoreCase) == 0 )
                  {
                     pbVmOnDnd->setValue(TRUE);
                  }
                  mUserVmOnDndPreferences.insertKeyAndValue( pUsername, pbVmOnDnd );
                  Os::Logger::instance().log( FAC_NAT, PRI_DEBUG, "PresenceRoutingUserPreferences::parseDocument added %s %d"
                                ,pUsername->data(), pbVmOnDnd->getValue());
               }
            }
         }
      }
   }
   else
   {
      Os::Logger::instance().log( FAC_NAT, PRI_ERR, "PresenceRoutingUserPreferences::parseDocument could not find node 'presenceRoutingPrefs'");
      rc = OS_FAILED;
   }
   return rc;
}
#endif

OsStatus PresenceRoutingUserPreferences::parseDocument()
{
    // TODO: Code is flawed, the list of users will change and their vm dnd settings will change,
    // and this code will not know about the changes.  We need to port this to be more dynamic
    // and use mongo queries for lookups. --Douglas

//  EntityDB::Entities entities;
//  EntityDB::defaultCollection().collection().getAllEntities(entities);
//  for(EntityDB::Entities::iterator iter = entities.begin(); iter != entities.end(); iter++)
//  {
//      if( !iter->userId().empty() )
//      {
//          UtlString* pUsername = new UtlString(iter->userId().c_str());
//          UtlBool* pbVmOnDnd = new UtlBool( FALSE );
//          if (iter->vmOnDnd())
//              pbVmOnDnd->setValue(TRUE);
//          mUserVmOnDndPreferences.insertKeyAndValue( pUsername, pbVmOnDnd );
//      }
//  }

  return OS_SUCCESS;
}

bool PresenceRoutingUserPreferences::forwardToVoicemailOnDnd(const UtlString& sipUsername )
{
   EntityRecord entity;
   EntityDB* entityDb = SipRegistrar::getInstance(NULL)->getEntityDB();
   if (entityDb->findByUserId(sipUsername.str(), entity))
   {
      return entity.vmOnDnd();
   }
   else
   {
      return false;
   }
}
