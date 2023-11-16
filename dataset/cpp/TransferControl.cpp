// 
// Copyright (C) 2007 Pingtel Corp., certain elements licensed under a Contributor Agreement.  
// Contributors retain copyright to elements licensed under a Contributor Agreement.
// Licensed to the User under the LGPL license.
// 
//////////////////////////////////////////////////////////////////////////////

// SYSTEM INCLUDES
#include <sipxproxy/SipRouter.h>
#include "os/OsLogger.h"
#include "os/OsConfigDb.h"

// APPLICATION INCLUDES
#include "net/Url.h"
#include "net/SipMessage.h"
#include "net/SipXauthIdentity.h"
#include "net/SipXlocationInfo.h"
#include "net/NetMd5Codec.h"
#include "TransferControl.h"

// DEFINES
// CONSTANTS
const char* TransferControl::RecognizerConfigKey1 = "EXCHANGE_SERVER_FQDN";
const char* TransferControl::RecognizerConfigKey2 = "ADDITIONAL_EXCHANGE_SERVER_FQDN";

const char* SIP_METHOD_URI_PARAMETER = "method";
const char* SIP_SIPX_REFERROR_HEADER = "X-sipX-referror";

// TYPEDEFS
// FORWARD DECLARATIONS

UtlString server1;
UtlString server2;
/// Factory used by PluginHooks to dynamically link the plugin instance
extern "C" AuthPlugin* getAuthPlugin(const UtlString& pluginName)
{
   return new TransferControl(pluginName);
}

/// constructor
TransferControl::TransferControl(const UtlString& pluginName ///< the name for this instance
   )
   : AuthPlugin(pluginName),
     mpSipRouter(NULL)
{
   Os::Logger::instance().log(FAC_SIP,PRI_INFO,"TransferControl plugin instantiated '%s'",
                 mInstanceName.data());
};

void 
TransferControl::announceAssociatedSipRouter( SipRouter* sipRouter )
{
   mpSipRouter = sipRouter;
}

/// Read (or re-read) the authorization rules.
void
TransferControl::readConfig( OsConfigDb& configDb /**< a subhash of the individual configuration
                                                    * parameters for this instance of this plugin. */
                             )
{
   /*
    * @note
    * The parent service may call the readConfig method at any time to
    * indicate that the configuration may have changed.  The plugin
    * should reinitialize itself based on the configuration that exists when
    * this is called.  The fact that it is a subhash means that whatever prefix
    * is used to identify the plugin (see PluginHooks) has been removed (see the
    * examples in PluginHooks::readConfig).
    */
   Os::Logger::instance().log(FAC_SIP, PRI_DEBUG, "TransferControl[%s]::readConfig",
                 mInstanceName.data()
                 );
   if (configDb.get(RecognizerConfigKey1, server1) && !server1.isNull())
   {
      Os::Logger::instance().log(FAC_SIP,PRI_INFO
                    ,"TransferControl[%s]::readConfig "
                    " server %s : '%s'"
                    ,mInstanceName.data(), RecognizerConfigKey1
                    ,server1.data()
                    );
   }
   if (configDb.get(RecognizerConfigKey2, server2) && !server2.isNull())
   {
      Os::Logger::instance().log(FAC_SIP,PRI_INFO
                    ,"TransferControl[%s]::readConfig "
                    " server %s : '%s'"
                    ,mInstanceName.data(), RecognizerConfigKey2
                    ,server2.data()
                    );
   }
}

AuthPlugin::AuthResult
TransferControl::authorizeAndModify(const UtlString& id,    /**< The authenticated identity of the
                                                             *   request originator, if any (the null
                                                             *   string if not).
                                                             *   This is in the form of a SIP uri
                                                             *   identity value as used in the
                                                             *   credentials database (user@domain)
                                                             *   without the scheme or any parameters.
                                                             */
                                    const Url&  requestUri, ///< parsed target Uri
                                    RouteState& routeState, ///< the state for this request.  
                                    const UtlString& method,///< the request method
                                    AuthResult  priorResult,///< results from earlier plugins.
                                    SipMessage& request,    ///< see AuthPlugin wrt modifying
                                    bool bSpiralingRequest, ///< request spiraling indication 
                                    UtlString&  reason      ///< rejection reason
                                    )
{
   AuthResult result = CONTINUE;
   
   // get the call-id to use in logging
   UtlString callId;
   request.getCallIdField(&callId);

   UtlString hostAddress;
   int hostPort;
   UtlString hostProtocols;
   //requestUri.getHostAddress(hostAddress);
   //request.getContactUri(0, &hostAddress);;
   request.getContactAddress(0, &hostAddress,&hostPort,&hostProtocols);
   if (DENY != priorResult)
   {
      if (method.compareTo(SIP_REFER_METHOD) == 0)
      {
         UtlString targetStr;
         if (request.getReferToField(targetStr))
         {
            Url target(targetStr, Url::NameAddr);  // parse the target URL

            UtlString targetMethod; 
            if (   Url::SipUrlScheme == target.getScheme() 
                /* REFER can create requests other than INVITE: we don't care about those       *
                 * so check that the method is INVITE or is unspecified (INVITE is the default) */
                && (   ! target.getUrlParameter(SIP_METHOD_URI_PARAMETER, targetMethod)
                    || (0==targetMethod.compareTo(SIP_INVITE_METHOD, UtlString::ignoreCase))
                    ))
            {
               if (id.isNull())
               {
                  // UnAuthenticated REFER. Do challenge the REFER to confirm the 
                  // identity of the transferor.  Note:  prior to XECS-2487, we used to challenge
                  // only the unauthenticated REFERs that didn't carry a Replaces header.
                  // The fix for XECS-2487 now requires that all unauthenticated REFERs
                  // be challenged so that consultative transfers get routed properly
                  // when user-based gateway section is used.  See tracker for the details
                  if (mpSipRouter->isLocalDomain(target))
                  {
                     //White list of two servets to let Exchange REFER to sipXecs endpoints
                     if (hostAddress.compareTo(server1, UtlString::ignoreCase) == 0 || hostAddress.compareTo(server2, UtlString::ignoreCase) == 0)
                     {
                        Os::Logger::instance().log(FAC_AUTH, PRI_INFO, "TransferControl[%s]::authorizeAndModify "
                            "Whitelist host '%s' in call '%s'",
					                  mInstanceName.data(),hostAddress.data(),callId.data()
                            );
			                  result = ALLOW; //Whitelist matched so allow the transfer
			               }else{
                        Os::Logger::instance().log(FAC_AUTH, PRI_INFO, "TransferControl[%s]::authorizeAndModify "
                            "challenging transfer in call '%s' from host '%s'",
                            mInstanceName.data(), callId.data(),hostAddress.data()
                            );
                        result = DENY; // we need an identity to attach to the Refer-To URI
                     }
                  }
                  else
                  {
                     /*
                      * This is a transfer to a target outside our domain, so let it go
                      * unchallenged.  See XECS-806
                      */
                     Os::Logger::instance().log(FAC_AUTH, PRI_DEBUG, "TransferControl[%s]::authorizeAndModify "
                                   "allowing foriegn transfer in call '%s'",
                                   mInstanceName.data(), callId.data()
                                   );
                     // Add the References to the refer-to. Adding the callId field as a reference
                     // header (will be used in resulting INVITE) in the Refer-To provides us
                     // with enough information to be able to logically tie the calls together.
                     // Useful for CDR records.  
                     UtlString refcallId(callId);
                     refcallId.append(";rel=refer");
                     target.setHeaderParameter(SIP_REFERENCES_FIELD, refcallId.data());
                  
                     Os::Logger::instance().log(FAC_AUTH, PRI_DEBUG, "TransferControl[%s]::authorizeAndModify "
                                   "adding Reference field [%s] to refer-to",
                                   mInstanceName.data(), callId.data()
                                  );
                     request.setReferToField(target.toString().data());

                     result = ALLOW;
                  }
               }
               else
               {
                   UtlString  contactString;
                   request.getContactEntry(0, &contactString);
                   Url contactUri( contactString );
                   UtlString userId;
                   contactUri.getUserId(contactString);

            	     Os::Logger::instance().log(FAC_AUTH, PRI_DEBUG, "TransferControl::authorizeAndModify - Contact field is: %s ", contactString.data());

                   if (contactString != "callcontroller")
                   {
                	   // Authenticated REFER
                	   // annotate the refer-to with the authenticated controller identity
                	   SipXauthIdentity controllerIdentity;
                	   controllerIdentity.setIdentity(id);
                	   controllerIdentity.encodeUri(target);
                	   
                	   //
                     // Add the referror param
                     //
                     target.setUrlParameter(SIP_SIPX_REFERROR_HEADER, id.data());

                	   // add the References to the refer-to.
                	   UtlString refcallId(callId);
                	   refcallId.append(";rel=refer");
                	   target.setHeaderParameter(SIP_REFERENCES_FIELD, refcallId.data());
                  
                	   Os::Logger::instance().log(FAC_AUTH, PRI_DEBUG, "TransferControl[%s]::authorizeAndModify "
                                "adding Reference field [%s] to refer-to",
                                mInstanceName.data(), callId.data()
                               );

                	   if (!mpSipRouter->supportMultipleGatewaysPerLocation())
                	   {
                	     addLocationInfo(id, target);
                	   }

                     Os::Logger::instance().log(FAC_AUTH, PRI_DEBUG, "TransferControl[%s]::authorizeAndModify "
                                "adding Reference field [%s] to refer-to",
                                mInstanceName.data(), callId.data()
                               );

                	   request.setReferToField(target.toString().data());
                   }
               }
            }
            else
            {
               Os::Logger::instance().log(FAC_AUTH, PRI_WARNING, "TransferControl[%s]::authorizeAndModify "
                             "unrecognized refer target '%s' for call '%s'",
                             mInstanceName.data(), targetStr.data(), callId.data()
                             );
            }
         }
         else
         {
            // REFER without a Refer-To header... incorrect, but just ignore it.
            Os::Logger::instance().log(FAC_AUTH, PRI_WARNING,
                          "TransferControl[%s]::authorizeAndModify "
                          "REFER method without Refer-To in call '%s'",
                          mInstanceName.data(), callId.data()
                          );
         }
      }
      else if (method.compareTo(SIP_INVITE_METHOD) == 0)
      {

         UtlString targetCallId;
         UtlString targetFromTag;
         UtlString targetToTag;
         UtlString referrorId;

         requestUri.getUrlParameter(SIP_SIPX_REFERROR_HEADER, referrorId, 0);
         if (!referrorId.isNull())
         {
           //
           // This is a transfer.  Set the parameter as a SIP header so it doesn't get lost during redirections
           //
           request.setHeaderValue(SIP_SIPX_REFERROR_HEADER, referrorId.data(), 0);
           
           OS_LOG_INFO(FAC_SIP, "Setting " << SIP_SIPX_REFERROR_HEADER << ": " << referrorId.data() << " from local domain transfer.");
           //
           // Remove the uri parameter
           //
           UtlString uri;
           UtlString protocol;
           
           Url newUri(requestUri);
           newUri.removeUrlParameter(SIP_SIPX_REFERROR_HEADER);
           newUri.getUri(uri);
           
           request.getRequestProtocol(&protocol);
           
           request.setFirstHeaderLine(method, uri, protocol);
         }

         if (request.getReplacesData(targetCallId, targetToTag, targetFromTag))
         {
            /*
             * This is an INVITE with Replaces: probably either the completion
             * of a call pickup or a consultative transfer.
             * In any case, it will not create a new call - just connect something
             * to an existing call - so we don't need to make any new authorization
             * decisions.
             */
            result = ALLOW;

            if (!bSpiralingRequest)
            {
               // remove any x-sipX-Location-Info header
               SipXSignedHeader::remove(request, SIP_SIPX_LOCATION_INFO);
            }
         }
         else
         {
            // INVITE without Replaces: is not a transfer - ignore it.
         }
      }
      else
      {
         // neither REFER nor INVITE, so is not a transfer - ignore it.
      }
   }
   else
   {
      // Some earlier plugin already denied this - don't waste time figuring it out.
      Os::Logger::instance().log(FAC_AUTH, PRI_DEBUG, "TransferControl[%s]::authorizeAndModify "
                    "prior authorization result %s for call %s",
                    mInstanceName.data(), AuthResultStr(priorResult), callId.data());
   }
   
   return result;
}

/// Boolean indicator that returns true if the plugin wants to process requests
/// that requires no authentication
bool TransferControl::willModifyTrustedRequest() const
{
  return true;
}

/// This method is called by the proxy if willModifyRequest() flag is set to true
/// giving this plugin the opportunity to modify the request even if it requires
/// no authentication
void TransferControl::modifyTrustedRequest(
                                 const Url&  requestUri,  ///< parsed target Uri
                                 SipMessage& request,     ///< see below regarding modifying this
                                 bool bSpiralingRequest  ///< true if request is still spiraling through proxy
                                 )
{
  UtlString method;
  request.getRequestMethod(&method);
  
  if (method.compareTo(SIP_INVITE_METHOD) == 0)
  {
    UtlString referrorId;
    requestUri.getUrlParameter(SIP_SIPX_REFERROR_HEADER, referrorId, 0);
    if (!referrorId.isNull())
    {
      //
      // This is a transfer.  Set the parameter as a SIP header so it doesn't get lost during redirections
      //
      request.setHeaderValue(SIP_SIPX_REFERROR_HEADER, referrorId.data(), 0);
      
      OS_LOG_INFO(FAC_SIP, "Setting " << SIP_SIPX_REFERROR_HEADER << ": " << referrorId.data() << " from non-local domain transfer.");
      
      //
      // Remove the uri parameter
      //
      UtlString uri;
      UtlString protocol;

      Url newUri(requestUri);
      newUri.removeUrlParameter(SIP_SIPX_REFERROR_HEADER);
      newUri.getUri(uri);

      request.getRequestProtocol(&protocol);
      request.setFirstHeaderLine(method, uri, protocol);
    }
  }
}

bool TransferControl::willModifyFinalResponse() const
{
  return true;
}

void TransferControl::modifyFinalResponse(
  SipTransaction* pTransaction, 
  const SipMessage& request, 
  SipMessage& finalResponse)
{
  //
  // Check if the response is a 302
  //
  int statusCode = -1;
  statusCode = finalResponse.getResponseStatusCode();
  bool is3xx = (statusCode >= SIP_3XX_CLASS_CODE && statusCode < SIP_4XX_CLASS_CODE);
  if (!is3xx)
    return;
  
  //
  // Check if the 3xx is from the registrar by checking for sipXecs-CallDest in the contact
  //
  UtlString  contactString;
  UtlString targetId;
  
  finalResponse.getContactEntry(0, &contactString);
  Url contactUri( contactString );
  contactUri.getUserId(targetId);
  if (contactString.first(SIP_SIPX_CALL_DEST_FIELD) != UtlString::UTLSTRING_NOT_FOUND)
    return;
     
  //
  // Check if it already has SIP_SIPX_AUTHIDENTITY
  //
  if (contactString.first(SIP_SIPX_AUTHIDENTITY) != UtlString::UTLSTRING_NOT_FOUND)
    return;
  
  //
  // At this point we already know that this is a 302 redirect that did not come from sipx or is signed by sipx
  //
  OS_LOG_DEBUG(FAC_SIP, "TransferControl::modifyFinalResponse - Evaluating contact " << contactString.data());
 
  //
  // Get the request-uri user
  //
  UtlString stringUri;
  UtlString requestUriUser;
  request.getRequestUri(&stringUri);
  // The requestUri is an addr-spec, not a name-addr.
  Url requestUri(stringUri, TRUE);
  requestUri.getUserId(requestUriUser);
  if (requestUriUser.isNull())
  {
    OS_LOG_ERROR(FAC_SIP, "TransferControl::modifyFinalResponse - Unable to determine user identity.")
    return;
  }
  
  //
  // Get local domain
  //
  UtlString localDomain;
  mpSipRouter->getDomain(localDomain);
  if (localDomain.isNull())
  {
    OS_LOG_ERROR(FAC_SIP, "TransferControl::modifyFinalResponse - Unable to determine local domain.")
    return;
  }
  
  //
  // Create the identity
  //
  std::ostringstream identity;
  identity << requestUriUser.data() << "@" << localDomain.data();
  
  //
  // Get the source address of the response
  //
  UtlString srcAddress;
  int srcPort = PORT_NONE;
  finalResponse.getSendAddress(&srcAddress, &srcPort);
  if (srcAddress.isNull())
  {
    OS_LOG_ERROR(FAC_SIP, "TransferControl::modifyFinalResponse - Unable to determine source address.")
    return;
  }
  
  //
  // Check if the identity is registered to this address
  //
  if  (!mpSipRouter->isRegisteredAddress(identity.str(), srcAddress.str()))
  {
    OS_LOG_WARNING(FAC_SIP, "TransferControl::modifyFinalResponse - " << identity.str() << " is not registered from address " << srcAddress.str());
    return;
  }
  
  //
  // 3XX is from a registered user.  Sign the contact
  //
  SipXauthIdentity authIdentity;
  authIdentity.setIdentity(identity.str().c_str());
  authIdentity.encodeUri(contactUri);
  
  UtlString signedContact;
  contactUri.toString(signedContact);
  
  finalResponse.setContactField(signedContact, 0);
  
  OS_LOG_INFO(FAC_SIP, "TransferControl::modifyFinalResponse - identity=" << identity.str() << " contact was " <<  contactString.data() << " now " <<  signedContact.data());
}

void TransferControl::addLocationInfo(const UtlString& id, /// The authenticated identity of the request originator,
                                                      /// this is expected to not be null
                                           Url& target   /// Refer-To header where to add location info
                                        )
{
  SipXSignedHeader locationInfo(id, SIP_SIPX_LOCATION_INFO);

  UtlString location;
  mpSipRouter->getUserLocation(id, location);
  if (!location.isNull())
  {
    // set location param if any
    locationInfo.setParam(SIPX_SIPXECS_LOCATION_URI_PARAM, location.data());
  }
  else
  {
    Os::Logger::instance().log(FAC_AUTH, PRI_DEBUG, "TransferControl[%s]::authorizeAndModify "
        "identity [%s] has no location specified",
        mInstanceName.data(), id.data());
  }

  if (!locationInfo.encodeUri(target))
  {
    Os::Logger::instance().log(FAC_AUTH, PRI_INFO, "TransferControl[%s]::authorizeAndModify "
       "identity [%s], failed to add location field [] to refer-to ",
       mInstanceName.data(), id.data());
  }
}


/// destructor
TransferControl::~TransferControl()
{
}

