//
// Copyright (C) 2007 Pingtel Corp., certain elements licensed under a Contributor Agreement.
// Contributors retain copyright to elements licensed under a Contributor Agreement.
// Licensed to the User under the LGPL license.
//
// $$
//////////////////////////////////////////////////////////////////////////////

// SYSTEM INCLUDES
// APPLICATION INCLUDES

#include "ResourceListSet.h"
#include "ResourceList.h"
#include "SubscriptionSet.h"
#include "ResourceNotifyReceiver.h"
#include "ResourceSubscriptionReceiver.h"
#include "ResourceListMsg.h"
#include <os/OsLogger.h>
#include <os/OsEventMsg.h>
#include <os/UnixSignals.h>
#include <utl/XmlContent.h>
#include <utl/UtlSListIterator.h>
#include <net/SipDialogEvent.h>
#include <net/NameValueTokenizer.h>
#include <net/NameValuePair.h>
#include <net/HttpMessage.h>
#include <net/SipMessage.h>
#include <xmlparser/tinyxml.h>

// EXTERNAL FUNCTIONS
// EXTERNAL VARIABLES
// CONSTANTS

// URN for the xmlns attribute for Resource List Meta-Information XML.
#define RLMI_XMLNS "urn:ietf:params:xml:ns:rlmi"
// MIME information for RLMI XML.
#define RLMI_CONTENT_TYPE "application/rlmi+xml"

// Resubscription period.
#define RESUBSCRIBE_PERIOD 3600

// STATIC VARIABLE INITIALIZATIONS

const UtlContainableType ResourceListSet::TYPE = "ResourceListSet";
const int ResourceListSet::sSeqNoIncrement = 4;
const int ResourceListSet::sSeqNoMask = 0x3FFFFFFC;
// RFC 4235 specifies a maximum of one RLMI notification per second.
// ::sGapTimeout is an OsTime for the minimum interval allowed by RFC 4235.
const OsTime ResourceListSet::sGapTimeout(1, 0);

/* //////////////////////////// PUBLIC //////////////////////////////////// */

/* ============================ CREATORS ================================== */

// Constructor
ResourceListSet::ResourceListSet(ResourceListServer* resourceListServer) :
   mResourceListServer(resourceListServer),
   mResourceCache(this),
   mNextSeqNo(0),
   mSuspendPublishingCount(0),
   mPublishingTimer(getResourceListServer()->getResourceListTask().
                    getMessageQueue(),
                    (void*)ResourceListSet::PUBLISH_TIMEOUT),
   mPublishOnTimeout(FALSE),
   mVersion(0),
   _subscriptionSetTimers(resourceListServer->getResourceListTask().getMessageQueue())
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet:: this = %p",
                 this);
}

// Destructor
ResourceListSet::~ResourceListSet()
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::~ this = %p",
                 this);
}

// Flag to indicate publish on timeout
UtlBoolean ResourceListSet::publishOnTimeout()
{
   return mPublishOnTimeout;
}

// Start the gap timeout.
void ResourceListSet::startGapTimeout()
{
   // After publishing create a 1 second delay before publishing again.
   mPublishingTimer.oneshotAfter(sGapTimeout);
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::startGapTimeout "
                 "mPublishingTimer.oneshotAfter(ResourceListSet::sGapTimeout = 1 sec)");

   // Don't publish when the gap timeout ends.
   mPublishOnTimeout = FALSE;
}

// Delete all ResourceList's and stop the publishing timer.
void ResourceListSet::finalize()
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::finalize this = %p",
                 this);

   // Make sure the ResourceList's are destroyed so that all
   // references to the ResourceCached's are removed, before
   // destroying the ResourceCache.
   deleteAllResourceLists(false);

   // Make sure the publishing timer is stopped before the ResourceListTask
   // is destroyed, because the timer posts messages to ResourceListTask.
   mPublishingTimer.stop();
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::finalize mPublishingTimer.stop()");
}

/* ============================ MANIPULATORS ============================== */

bool ResourceListSet::addSubscriptionSetByTimer(
        const UtlString& callidContact,
        UtlContainable* handler,
        const OsTime& offset)
{
    Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
            "ResourceListSet::addSubscriptionSetByTimer "
            "this = %p, callidContact = '%s', handler = '%p', offset = '%d'",
                  this, callidContact.data(), handler, offset.cvtToMsecs());

    OsStatus ret = _subscriptionSetTimers.scheduleOneshotAfter(
                        new SubscriptionSetMsg(handler, callidContact),
                        offset);

    if (OS_SUCCESS != ret)
    {
        Os::Logger::instance().log(FAC_RLS, PRI_ERR,
                "ResourceListSet::addSubscriptionSetByTimer failed for "
                "this = %p, callidContact = '%s', handler = '%p', offset = '%d'",
                      this, callidContact.data(), handler, offset.cvtToMsecs());
    }

    return (OS_SUCCESS == ret);
}

// Create and add a resource list.
bool ResourceListSet::addResourceList(const char* user,
                                      const char* userCons,
                                      const char* nameXml)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::addResourceList this = %p, user = '%s', userCons = '%s', nameXml = '%s'",
                 this, user, userCons, nameXml);


   // Check to see if there is already a list with this name.
   ResourceList::Ptr pResource = findResourceList(user);
   if (!pResource)
   {
      // Create the resource list.
      pResource = ResourceList::Ptr(new ResourceList(this, user, userCons));

      // Update the version number for consolidated events if it is too
      // small for an existing subscription to this URI.
      int v =
         getResourceListServer()->getSubscriptionMgr().
         getNextAllowedVersion(*(pResource->getResourceListUriCons()));
      if (v > mVersion)
      {
         mVersion = v;
      }

      // Add the resource list to the set.
      mutex_write_lock lock(_listMutex);
      _resourceLists[user] = pResource;

      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ResourceListSet::addResourceList added ResourceList, mVersion = %d",
                    mVersion);
      return true;
   }
   else
   {
      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ResourceListSet::addResourceList ResourceList '%s' already exists",
                    user);
      return false;
   }

}

// Get the information from resource in a resource list specified
// by its position in the list.
void ResourceListSet::getResourceInfoAt(const char* user,
                                        size_t at,
                                        UtlString& uri,
                                        UtlString& nameXml,
                                        UtlString& display_name)
{
   // Serialize access to the ResourceListSet.
   ResourceList::Ptr pResource = findResourceList(user);
   if (pResource)
   {
      pResource->getResourceInfoAt(at, uri, nameXml, display_name);
   }
   else
   {
      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ResourceListSet::getResourceInfoAt ResourceList '%s' not found",
                    user);
   }
}

// Delete all resource lists.
void ResourceListSet::deleteAllResourceLists(bool abortOnShutdown)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::deleteAllResourceLists "
                 "this = %p, abortOnShutdown = %d",
                 this, abortOnShutdown);

   // Gradually remove elements from the ResourceLists and delete them.
   ResourceList::Ptr rl;
   int changeDelay = getResourceListServer()->getChangeDelay();
   do {
      // Set to true if a ResourceCached was deleted and so we need to delay.
      bool resource_deleted = false;

      {
         // Serialize access to the ResourceListSet.
         mutex_write_lock lock(_listMutex);

         // Get pointer to the first ResourceList.
         if (!_resourceLists.empty())
           rl =  _resourceLists.begin()->second;
         else
           rl = ResourceList::Ptr();

         // If one exists, shrink it.
         if (rl) {
            bool list_empty;
            rl->shrink(list_empty, resource_deleted);
            if (list_empty) {
               // The ResourceList is empty, and so can be removed and deleted.
               _resourceLists.erase(rl->getUserPart()->data());
            }
         }
      }

      if (resource_deleted)
      {
         // Delay to allow the consequent processing to catch up.
         OsTask::delay(changeDelay);
      }
   } while (rl && !Os::UnixSignals::instance().isTerminateSignalReceived());
}

// Delete a resource list.
void ResourceListSet::deleteResourceList(const char* user)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::deleteResourceList this = %p, user = '%s'",
                 this, user);

   // Gradually remove elements from the ResourceList and delete them.
   int changeDelay = getResourceListServer()->getChangeDelay();
   bool exitLoop = false;
   do {
      // Set to true if a ResourceCached was deleted and so we need to delay.
      bool resource_deleted = false;

      {
 
         // Get pointer to the ResourceList.
         ResourceList::Ptr pResource = findResourceList(user);

         // If it exists, shrink it.
         if (pResource) {
            bool list_empty;
            pResource->shrink(list_empty, resource_deleted);
            if (list_empty) {
               // The ResourceList is empty, and so can be removed and deleted.
              // Serialize access to the ResourceListSet.
               mutex_write_lock lock(_listMutex);
               _resourceLists.erase(pResource->getUserPart()->data());
               exitLoop = true;
            }
         }
         else
         {
           exitLoop = true;
         }
      }

      if (resource_deleted)
      {
         // Delay to allow the consequent processing to catch up.
         OsTask::delay(changeDelay);
      }
   } while (exitLoop && !Os::UnixSignals::instance().isTerminateSignalReceived());
}

void ResourceListSet::deleteResourceAt(const char* user,
                                       size_t at)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::deleteResourceAt this = %p, user = '%s', at = %d",
                 this, user, (int) at);


   ResourceList::Ptr pResource = findResourceList(user);
   if (pResource)
   {
      bool resource_deleted = pResource->deleteResourceAt(at);

      if (resource_deleted)
      {
         // Delay to allow the consequent processing to catch up.
         OsTask::delay(getResourceListServer()->getChangeDelay());
      }

      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ResourceListSet::deleteResourceAt done");
   }
   else
   {
      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ResourceListSet::deleteResourceAt ResourceList '%s' not found",
                    user);
   }
}

// Get a list of the user-parts of all resource lists.
void ResourceListSet::getAllResourceLists(UtlSList& list)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::getAllResourceLists this = %p",
                 this);

   // Serialize access to the ResourceListSet.
   mutex_read_lock lock(_listMutex);
   for (ResourceMap::iterator iter = _resourceLists.begin(); iter != _resourceLists.end(); iter++)
   {
     list.append(new UtlString(*(iter->second->getUserPart())));
   }
}

//! Create and add a resource to the resource list.
bool ResourceListSet::addResource(const char* user,
                                  const char* uri,
                                  const char* nameXml,
                                  const char* display_name,
                                  ssize_t no_check_start,
                                  ssize_t no_check_end)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::addResource this = %p, user = '%s', uri = '%s', nameXml = '%s', display_name = '%s', no_check_start = %d, no_check_end = %d",
                 this, user, uri, nameXml, display_name,
                 (int) no_check_start, (int) no_check_end);

   bool resource_added = false;
   ResourceList::Ptr pResource = findResourceList(user);
   if (pResource)
   {
      bool resource_cached_created;
      pResource->addResource(uri, nameXml, display_name,
                                resource_added, resource_cached_created,
                                no_check_start, no_check_end);

      if (resource_cached_created)
      {
         // Delay to allow the consequent processing to catch up.
         OsTask::delay(getResourceListServer()->getChangeDelay());
      }

      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ResourceListSet::addResource resource added");
   }
   else
   {
      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ResourceListSet::addResource ResourceList '%s' not found",
                    user);
   }

   return resource_added;
}

// Get the number of resources in a resource list.
size_t ResourceListSet::getResourceListEntries(const char* user)
{
   // Serialize access to the ResourceListSet.
   size_t ret = 0;

   ResourceList::Ptr pResource = findResourceList(user);
   if (pResource)
   {
      ret = pResource->entries();
   }
   else
   {
      Os::Logger::instance().log(FAC_RLS, PRI_WARNING,
                    "ResourceListSet::getResourceListEntries "
                    "user = '%s' could not be found",
                    user);
   }

   return ret;
}

// Callback routine for subscription state events.
// Called as a callback routine.
void ResourceListSet::subscriptionEventCallbackAsync(
   SipSubscribeClient::SubscriptionState newState,
   const char* earlyDialogHandle,
   const char* dialogHandle,
   void* applicationData,
   int responseCode,
   const char* responseText,
   long expiration,
   const SipMessage* subscribeResponse
   )
{
   // earlyDialogHandle may be NULL for some termination callbacks.
   if (!earlyDialogHandle)
   {
      earlyDialogHandle = "";
   }
   // dialogHandle may be NULL for some termination callbacks.
   if (!dialogHandle)
   {
      dialogHandle = "";
   }
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::subscriptionEventCallbackAsync newState = %d, applicationData = %p, earlyDialogHandle = '%s', dialogHandle = '%s'",
                 newState, applicationData, earlyDialogHandle, dialogHandle);

   // The ResourceListSet concerned is applicationData.
   ResourceListSet* resourceListSet = (ResourceListSet*) applicationData;

   // Determine the subscription state.
   // Currently, this is only "active" or "terminated", which is not
   // very good.  But the real subscription state is carried in the
   // NOTIFY requests. :TODO: Handle subscription set correctly.
   const char* subscription_state;
   if (subscribeResponse)
   {
      int expires;
      subscribeResponse->getExpiresField(&expires);
      subscription_state = expires == 0 ? "terminated" : "active";
   }
   else
   {
      subscription_state = "active";
   }

   // Send a message to the ResourceListTask.
   resourceListSet->getResourceListServer()->getResourceListTask().
      postMessageP(
         new SubscriptionCallbackMsg(earlyDialogHandle, dialogHandle,
                                     newState, subscription_state));
}

// Callback routine for subscription state events.
// Called by ResourceListTask.
void ResourceListSet::subscriptionEventCallbackSync(
   const UtlString* earlyDialogHandle,
   const UtlString* dialogHandle,
   SipSubscribeClient::SubscriptionState newState,
   const UtlString* subscriptionState
   )
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::subscriptionEventCallbackSync earlyDialogHandle = '%s', dialogHandle = '%s', newState = %d, subscriptionState = '%s'",
                 earlyDialogHandle->data(), dialogHandle->data(), newState,
                 subscriptionState->data());



   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::subscriptionEventCallbackSync after mutex_read_lock on semaphore");

   // Look up the ResourceSubscriptionReceiver to notify based on the
   // earlyDialogHandle.
   /* To call the handler, we dynamic_cast the object to
    * (ResourceSubscriptionReceiver*).  Whether this is strictly
    * conformant C++ I'm not sure, since UtlContainanble and
    * ResourceSubscriptionReceiver are not base/derived classes of
    * each other.  But it seems to work in GCC as long as the dynamic
    * type of the object is a subclass of both UtlContainable and
    * ResourceSubscriptionReceiver.
    */
   ResourceSubscriptionReceiver::CallBack::Ptr receiver;
   {
     recursive_mutex_read_lock lock(_subscriptionMutex);
     if (_subscribeMap.find(earlyDialogHandle->data()) != _subscribeMap.end())
       receiver = _subscribeMap[earlyDialogHandle->data()];
   }


   if (receiver)
   {
        Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::subscriptionEventCallbackSync calling receiver->subscriptionEventCallback");
      receiver->subscriptionEventCallback(earlyDialogHandle,
                                          dialogHandle,
                                          newState,
                                          subscriptionState);
        Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::subscriptionEventCallbackSync exiting receiver->subscriptionEventCallback");
   }
   else
   {
      Os::Logger::instance().log(FAC_RLS, PRI_WARNING,
                    "ResourceListSet::subscriptionEventCallbackSync this = %p, no ResourceSubscriptionReceiver found for earlyDialogHandle '%s'",
                    this, earlyDialogHandle->data());
   }
   
      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::subscriptionEventCallbackSync exit function");
}

// Callback routine for NOTIFY events.
// Called as a callback routine.
bool ResourceListSet::notifyEventCallbackAsync(const char* earlyDialogHandle,
                                               const char* dialogHandle,
                                               void* applicationData,
                                               const SipMessage* notifyRequest)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::notifyEventCallbackAsync applicationData = %p, earlyDialogHandle = '%s', dialogHandle = '%s'",
                 applicationData, earlyDialogHandle, dialogHandle);

   // The ResourceListSet concerned is applicationData.
   ResourceListSet* resourceListSet = (ResourceListSet*) applicationData;

   if (resourceListSet && notifyRequest);
    resourceListSet->getResourceListServer()->getResourceListTask().handleNotifyRequest(*notifyRequest);
   // Get the NOTIFY content.
   const char* b;
   ssize_t l;
   const HttpBody* body = notifyRequest->getBody();
   if (body)
   {
      body->getBytes(&b, &l);
   }
   else
   {
      b = NULL;
      l = 0;
   }

   // Send a message to the ResourceListTask.
   resourceListSet->getResourceListServer()->getResourceListTask().
      postMessageP(new NotifyCallbackMsg(dialogHandle, b, l));

   return true;
}

// Callback routine for NOTIFY events.
// Called by ResourceListTask.
void ResourceListSet::notifyEventCallbackSync(const UtlString* dialogHandle,
                                              const UtlString* content)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::notifyEventCallbackSync dialogHandle = '%s'",
                 dialogHandle->data());

   
   

   // Look up the ResourceNotifyReceiver to notify based on the dialogHandle.
   /* To call the handler, we dynamic_cast the object to
    * (ResourceNotifyReceiver*).  Whether this is strictly
    * conformant C++ I'm not sure, since UtlContainanble and
    * ResourceNotifyReceiver are not base/derived classes of
    * each other.  But it seems to work in GCC as long as the dynamic
    * type of the object is a subclass of both UtlContainable and
    * ResourceNotifyReceiver.
    */
   ResourceNotifyReceiver::CallBack::Ptr receiver;
   {
     // Serialize access to the ResourceListSet.
     mutex_read_lock lock(_notifyMutex);
     if (_notifyMap.find(dialogHandle->data()) != _notifyMap.end())
       receiver = _notifyMap[dialogHandle->data()];
   }

   if (receiver)
   {
      receiver->notifyEventCallback(dialogHandle, content);
   }
   else
   {
      Os::Logger::instance().log(FAC_RLS, PRI_WARNING,
                    "ResourceListSet::notifyEventCallbackSync this = %p, no ResourceNotifyReceiver found for dialogHandle '%s'",
                    this, dialogHandle->data());
   }
}

/** Add a mapping for an early dialog handle to its handler for
 *  subscription events.
 */
void ResourceListSet::addSubscribeMapping(UtlString* earlyDialogHandle,
                                          UtlContainable* handler)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::addSubscribeMapping this = %p, earlyDialogHandle = '%s', handler = %p",
                 this, earlyDialogHandle->data(), handler);

   ResourceSubscriptionReceiver* pReceiver = dynamic_cast<ResourceSubscriptionReceiver*>(handler);
   if (pReceiver)
   {
     recursive_mutex_write_lock lock(_subscriptionMutex);
     _subscribeMap[earlyDialogHandle->data()] = pReceiver->getSafeCallBack();
   }
}

/** Delete a mapping for an early dialog handle.
 */
void ResourceListSet::deleteSubscribeMapping(UtlString* earlyDialogHandle)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::deleteSubscribeMapping this = %p, earlyDialogHandle = '%s'",
                 this, earlyDialogHandle->data());

   recursive_mutex_write_lock lock(_subscriptionMutex);
   _subscribeMap.erase(earlyDialogHandle->data());
}

/** Add a mapping for a dialog handle to its handler for
 *  NOTIFY events.
 */
void ResourceListSet::addNotifyMapping(const UtlString& dialogHandle,
                                       UtlContainable* handler)
{
   /* The machinery surrounding dialog handles is broken in that it
    * does not keep straight which tag is local and which is remote,
    * and the order of the tags in dialogHandle is not consistent.
    * Ideally, we would fix the problem (XSL-146), but there are many
    * places in the code where it is sloppy about tracking whether a
    * message is incoming or outgoing when constructing a
    * dialogHandle.  We circumvent this by making the lookup of
    * dialogs by dialogHandle insensitive to reversing the tags.  (See
    * SipDialog::isSameDialog.)
    */

   
   // If we already have a different mapping, report an error, as this
   // addNotifyMapping() should be a duplicate of the mapping we
   // already have.
   UtlContainable* current_handler = 0;
   ResourceNotifyReceiver::CallBack::Ptr receiver;
   {
     // Serialize access to the ResourceListSet.
     mutex_read_lock lock(_notifyMutex);
     if (_notifyMap.find(dialogHandle.data()) != _notifyMap.end())
     {
       receiver = _notifyMap[dialogHandle.data()];
       current_handler = (UtlContainable*)receiver->receiver();
     }
   }

   if (current_handler)
   {
      if (current_handler != handler)
      {
      Os::Logger::instance().log(FAC_RLS, PRI_ERR,
                    "ResourceListSet::addNotifyMapping Adding a different handler for an existing mapping: dialogHandle = '%s', current handler = %p, new handler = %p",
                    dialogHandle.data(), current_handler, handler);
      }
      // Remove the previous mapping in preparation for the new mapping.
      deleteNotifyMapping(&dialogHandle);
   }

   // Construct our copies of the dialog handle and the swapped dialog handle.
   UtlString* dialogHandleP = new UtlString(dialogHandle);
   UtlString* swappedDialogHandleP = new UtlString;
   swapTags(dialogHandle, *swappedDialogHandleP);

   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::addNotifyMapping this = %p, dialogHandle = '%s', swappedDialogHandle = '%s', handler = %p",
                 this,
                 dialogHandleP->data(), swappedDialogHandleP->data(),
                 handler);

   // Make entries in mNotifyMap for both forms of the handle.

   ResourceNotifyReceiver* pNotifier = dynamic_cast<ResourceNotifyReceiver*>(handler);
   if (pNotifier)
   {
      mutex_write_lock lock(_notifyMutex);
      _notifyMap[dialogHandleP->data()] = pNotifier->getSafeCallBack();
      _notifyMap[swappedDialogHandleP->data()] = pNotifier->getSafeCallBack();
   }
}

/** Delete a mapping for a dialog handle.
 */
void ResourceListSet::deleteNotifyMapping(const UtlString* dialogHandle)
{
   // See comment in addNotifyMapping for why we have two entries, one for
   // the dialog handle and one for the swapped dialog handle.
   UtlString swappedDialogHandle;
   swapTags(*dialogHandle, swappedDialogHandle);

   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::deleteNotifyMapping this = %p, dialogHandle = '%s', swappedDialogHandle = '%s'",
                 this, dialogHandle->data(), swappedDialogHandle.data());


   mutex_write_lock lock(_notifyMutex);
   _notifyMap.erase(dialogHandle->data());
   _notifyMap.erase(swappedDialogHandle.data());

}

// Get the next sequence number for objects for the parent ResourceListServer.
int ResourceListSet::getNextSeqNo()
{
   // Update mNextSeqNo.
   mNextSeqNo = (mNextSeqNo + sSeqNoIncrement) & sSeqNoMask;

   // Return the new value.
   return mNextSeqNo;
}

// Returns TRUE if publish() should not have any effect.
UtlBoolean ResourceListSet::publishingSuspended()
{
   return mSuspendPublishingCount > 0;
}

// Suspend the effect of publish().
void ResourceListSet::suspendPublishing()
{
   // Serialize access to the ResourceListSet.
   mutex_read_lock lock(_listMutex);

   // Increment mSuspendPublishingCount.
   mSuspendPublishingCount++;

   // Stop the publishing timer, asynchronously.  This is to prevent
   // it from firing, so ResourceListTask doesn't bother trying to
   // publish the resource lists when it will have no effect, but also
   // so that when publishing is resumed, the publishing timer will be
   // started and eventually fire.
   mPublishingTimer.stop(FALSE);
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::suspendPublishing mPublishingTimer.stop()");

   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::suspendPublishing mSuspendPublishingCount now = %d",
                 mSuspendPublishingCount);
}

// Resume the effect of publish().
void ResourceListSet::resumePublishing()
{
   // Serialize access to the ResourceListSet.
   mutex_read_lock lock(_listMutex);

   // Decrement mSuspendPublishingCount if > 0.
   if (mSuspendPublishingCount > 0)
   {
      mSuspendPublishingCount--;

      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ResourceListSet::resumePublishing mSuspendPublishingCount now = %d",
                    mSuspendPublishingCount);

      // If mSuspendPublishingCount is now 0, publish all the lists.
      if (mSuspendPublishingCount == 0)
      {
         schedulePublishing();
      }
   }
   else
   {
      Os::Logger::instance().log(FAC_RLS, PRI_ERR,
                    "ResourceListSet::resumePublishing called when mSuspendPublishingCount = 0");
   }
}

// Declare that some content has changed and needs to be published.
void ResourceListSet::schedulePublishing()
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::schedulePublishing this = %p",
                 this);

   // If publishing has been suspended, do not start the timer --
   // it will be started when publishing is resumed.
   if (!publishingSuspended())
   {
      OsTime pubDelay = getResourceListServer()->getPublishingDelay();

      // Check if waiting for the gap timeout (rather than the publishing timeout)
      if (mPublishOnTimeout == FALSE)
      {
         OsTimer::OsTimerState tmrState;
         OsTimer::Time tmrExpiresAt;
         UtlBoolean tmrPeriodic;
         OsTimer::Interval tmrPeriod;
         mPublishingTimer.getFullState(tmrState, tmrExpiresAt, tmrPeriodic, tmrPeriod);

         // Check if the timer is currently running.
         if (tmrState == OsTimer::STARTED)
         {
            // Calculate the amount of time before the gap timer expires (in seconds and microseconds).
            OsTimer::Time timeDelta = tmrExpiresAt - OsTimer::now();
            OsTime pubGap(timeDelta / 1000000, timeDelta % 1000000);

            // If the remaining gap timeout is less than the pubDelay
            // then we need to wait for pubDelay before publishing.
            if (pubGap < pubDelay)
            {
               // Cancel the current gap timeout so that oneshotAfter can restart the timer.
               mPublishingTimer.stop();
               Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                             "ResourceListSet::schedulePublishing mPublishingTimer.stop()");
            }
         }
      }

      // Start the timer with the publishing timeout if the timer is not already started.
      // If it is already started, OsTimer::oneshotAfter() does nothing.
      mPublishingTimer.oneshotAfter(pubDelay);
      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ResourceListSet::schedulePublishing mPublishingTimer.oneshotAfter(%d.%06d)",
                    pubDelay.seconds(), pubDelay.usecs());

      // Publish once the publishing timer expires.
      mPublishOnTimeout = TRUE;
   }
}

// Publish all ResourceList's that have changes.
void ResourceListSet::publish()
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::publish this = %p",
                 this);

   

   // If publishing has been suspended, do nothing --
   // publish() will be called again after publishing is resumed.
   if (!publishingSuspended())
   {
      mutex_read_lock lock(_listMutex);
      // Iterate through the resource lists.
      for (ResourceMap::iterator iter = _resourceLists.begin(); iter != _resourceLists.end(); iter++)
      {
        iter->second->publishIfNecessary();
      }

      // Purge dialogs with terminated state and terminated resource
      // instances, now that we have published the fact that they've
      // terminated (and their termination reasons).
      getResourceCache().purgeTerminated();
   }
}

/* ============================ ACCESSORS ================================= */

/* //////////////////////////// PUBLIC //////////////////////////////////// */

/* ============================ INQUIRY =================================== */

/**
 * Get the ContainableType for a UtlContainable-derived class.
 */
UtlContainableType ResourceListSet::getContainableType() const
{
   return ResourceListSet::TYPE;
}

// Split a userData value into the seqNo and "enum notifyCodes".
void ResourceListSet::splitUserData(int userData,
                                    int& seqNo,
                                    enum notifyCodes& type)
{
   seqNo = userData & sSeqNoMask;
   type = (enum notifyCodes) (userData & ~sSeqNoMask);
}

// Retrieve an entry from mEventMap and delete it.
UtlContainable* ResourceListSet::retrieveObjectBySeqNoAndDeleteMapping(int seqNo)
{
   // Search for and possibly delete seqNo.
   UtlInt search_key(seqNo);
   UtlContainable* value;
   UtlContainable* key = mEventMap.removeKeyAndValue(&search_key, value);

   if (key)
   {
      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ResourceListSet::retrieveObjectBySeqNoAndDeleteMapping seqNo = %d, value = %p",
                    seqNo, value);
      delete key;
   }
   else
   {
      Os::Logger::instance().log(FAC_RLS, PRI_WARNING,
                    "ResourceListSet::retrieveObjectBySeqNoAndDeleteMapping seqNo = %d not found",
                    seqNo);
   }

   return value;
}

// Add a mapping for a ResourceCached's sequence number.
void ResourceListSet::addResourceSeqNoMapping(int seqNo,
                                              ResourceCached* resource)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ResourceListSet::addResourceSeqNoMappping seqNo = %d, instanceName = '%s'",
                 seqNo, resource->getUri()->data());

   // Allocate a UtlInt to hold the sequence number.
   UtlInt* i = new UtlInt(seqNo);

   // Put the pair in mEventMap.
   mEventMap.insertKeyAndValue(i, resource);
}

//! Delete a mapping for a Resource's sequence number.
void ResourceListSet::deleteResourceSeqNoMapping(int seqNo)
{
   // Search for and possibly delete seqNo.
   UtlInt search_key(seqNo);
   UtlContainable* value;
   UtlContainable* key = mEventMap.removeKeyAndValue(&search_key, value);

   if (key)
   {
      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ResourceListSet::deleteResourceSeqNoMapping seqNo = %d, instanceName = '%s'",
                    seqNo,
                    (dynamic_cast <ResourceCached*> (value))->getUri()->data());
      delete key;
   }
   else
   {
      Os::Logger::instance().log(FAC_RLS, PRI_WARNING,
                    "ResourceListSet::deleteResourceSeqNoMapping seqNo = %d not found",
                    seqNo);
   }
}

// Dump the object's internal state.
void ResourceListSet::dumpState()
{
   Os::Logger::instance().log(FAC_RLS, PRI_INFO,
                 "\t  ResourceListSet %p mSuspendPublishingCount = %d",
                 this, mSuspendPublishingCount);

   mutex_read_lock lock(_listMutex);
   for (ResourceMap::iterator iter = _resourceLists.begin(); iter != _resourceLists.end(); iter++)
   {
     iter->second->dumpState();
   }

   mResourceCache.dumpState();
}

/* //////////////////////////// PROTECTED ///////////////////////////////// */

// Search for a resource list with a given name (user-part).
ResourceList::Ptr ResourceListSet::findResourceList(const char* user)
{
  mutex_read_lock lock(_listMutex);

  if (_resourceLists.find(user) != _resourceLists.end())
    return _resourceLists[user];

  return ResourceList::Ptr();
}

/* //////////////////////////// PRIVATE /////////////////////////////////// */


/* ============================ FUNCTIONS ================================= */

// Swap the tags in a dialog handle.
void ResourceListSet::swapTags(const UtlString& dialogHandle,
                               UtlString& swappedDialogHandle)
{
   // Find the commas in the dialogHandle.
   ssize_t index1 = dialogHandle.index(',');
   ssize_t index2 = dialogHandle.index(',', index1+1);

   // Copy the call-Id and the first comma.
   swappedDialogHandle.remove(0);
   swappedDialogHandle.append(dialogHandle,
                              index1+1);

   // Copy the second tag.
   swappedDialogHandle.append(dialogHandle,
                              index2+1,
                              dialogHandle.length() - (index2+1));

   // Copy the first comma and the first tag.
   swappedDialogHandle.append(dialogHandle,
                              index1,
                              index2-index1);
}
