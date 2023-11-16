//
// Copyright (C) 2007 Pingtel Corp., certain elements licensed under a Contributor Agreement.
// Contributors retain copyright to elements licensed under a Contributor Agreement.
// Licensed to the User under the LGPL license.
//
// $$
//////////////////////////////////////////////////////////////////////////////

// SYSTEM INCLUDES
// APPLICATION INCLUDES

#include "ResourceListServer.h"
#include "ResourceCached.h"
#include "ContactSet.h"
#include "SubscriptionSet.h"
#include <os/OsLogger.h>
#include <utl/UtlHashMapIterator.h>
#include <utl/UtlHashBagIterator.h>
#include <net/SipMessage.h>
#include <net/HttpBody.h>
#include <net/SipRegEvent.h>
#include <xmlparser/tinyxml.h>
#include <xmlparser/tinystr.h>
#include <xmlparser/ExtractContent.h>

// EXTERNAL FUNCTIONS
// EXTERNAL VARIABLES
// CONSTANTS

// In case a ContactSet needs to initiate multiple subscriptions at the same time
// then multiple timers will be started to delay creation of these SubscriptionSets,
// one timer per each SubscriptionSet. To avoid having these timers firing all at
// the same moment, creating a burst of new dialog subscriptions, the expiration
// time of subsequent timers is increased by a small increment to spread the timers
// over an even interval.
#define SUBSCRIPTION_WAIT_INCR_MSEC 2

// STATIC VARIABLE INITIALIZATIONS

const UtlContainableType ContactSet::TYPE = "ContactSet";


/* //////////////////////////// PUBLIC //////////////////////////////////// */

/* ============================ CREATORS ================================== */

// Constructor
ContactSet::ContactSet(ResourceCached* resource,
                       UtlString& uri,
                       bool allowDirectUriSubscription) :
   mResource(resource),
   mUri(uri)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ContactSet:: this = %p, resource = %p, mUri = '%s'",
                 this, mResource, mUri.data());

   // Set up the subscriptions.
   // Until we have any information from our SUBSCRIBE for "reg" events,
   // and if flag allowDirectUriSubscription permits it, there will be one subscription to mUri.
   updateSubscriptions(allowDirectUriSubscription);

   // Start the subscription.
   UtlBoolean ret;
   UtlString mUriNameAddr = "<" + mUri + ">";
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "SubscriptionSet:: mUri = '%s', mUriNameAddr = '%s'",
                 mUri.data(), mUriNameAddr.data());
   ret = getResourceListServer()->getSubscribeClient().
      addSubscription(mUri.data(),
                      REG_EVENT_TYPE,
                      REG_EVENT_CONTENT_TYPE,
                      getResourceListServer()->getClientFromURI(),
                      mUriNameAddr.data(),
                      getResourceListServer()->getClientContactURI(),
                      getResourceListServer()->getResubscribeInterval(),
                      getResourceListSet(),
                      ResourceListSet::subscriptionEventCallbackAsync,
                      ResourceListSet::notifyEventCallbackAsync,
                      mSubscriptionEarlyDialogHandle);

   if (ret)
   {
      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ContactSet:: addSubscription succeeded mUri = '%s', mSubscriptionEarlyDialogHandle = '%s'",
                    mUri.data(),
                    mSubscriptionEarlyDialogHandle.data());
   }
   else
   {
      Os::Logger::instance().log(FAC_RLS, PRI_WARNING,
                    "ContactSet:: addSubscription failed first send mUri = '%s', mSubscriptionEarlyDialogHandle = '%s'",
                    mUri.data(),
                    mSubscriptionEarlyDialogHandle.data());
   }

   // Add this ContactSet to mSubscribeMap.
   // The mapping is done as long as addSubscription returned a valid handle
   // no matter if the addSubscription above succeeded or failed.
   if (!mSubscriptionEarlyDialogHandle.isNull())
   {
       getResourceListSet()->addSubscribeMapping(&mSubscriptionEarlyDialogHandle,
                                             this);
   }
}


// Destructor
ContactSet::~ContactSet()
{
   // Delete this ContactSet from mSubscribeMap (for the "reg" subscription).
   getResourceListSet()->deleteSubscribeMapping(&mSubscriptionEarlyDialogHandle);

   // End the "reg" subscription.
   UtlBoolean ret;
   ret = getResourceListServer()->getSubscribeClient().
      endSubscriptionGroup(mSubscriptionEarlyDialogHandle);
   Os::Logger::instance().log(FAC_RLS,
                 ret ? PRI_DEBUG : PRI_WARNING,
                 "ContactSet::~ endSubscriptionGroup %s mUri = '%s', mSubscriptionEarlyDialogHandle = '%s'",
                 ret ? "succeeded" : "failed",
                 mUri.data(),
                 mSubscriptionEarlyDialogHandle.data());

   // Remove this ContactSet from mNotifyMap for all subscriptions.
   {
      UtlHashMapIterator itor(mSubscriptions);
      UtlString* handle;
      while ((handle = dynamic_cast <UtlString*> (itor())))
      {
         getResourceListSet()->deleteNotifyMapping(handle);
      }
   }

   // Delete the contents of mSubscriptions.
   mSubscriptions.destroyAll();

   // Delete the subordinate SubscriptionSet's for the contacts.
   {
      // Have to use a loop to remove items individually, because
      // destroyAll() deadlocks with the access to mSubscriptionSets
      // during the the attempt to publish new status for the resource
      // as the ResourceInstances are recursively deleted.
      UtlHashMapIterator itor(mSubscriptionSets);
      UtlContainable* k;
      while ((k = itor()))
      {
         UtlContainable* v = itor.value();
         mSubscriptionSets.removeReference(k);
         delete k;
         delete v;
      }
   }
}

/* ============================ MANIPULATORS ============================== */

void ContactSet::subscriptionEventCallback(
   const UtlString* earlyDialogHandle,
   const UtlString* dialogHandle,
   SipSubscribeClient::SubscriptionState newState,
   const UtlString* subscriptionState)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ContactSet::subscriptionEventCallback mUri = '%s', newState = %d, earlyDialogHandle = '%s', dialogHandle = '%s', subscriptionState = '%s'",
                 mUri.data(),
                 newState, mSubscriptionEarlyDialogHandle.data(),
                 dialogHandle->data(), subscriptionState->data());

   switch (newState)
   {
   case SipSubscribeClient::SUBSCRIPTION_INITIATED:
      break;
   case SipSubscribeClient::SUBSCRIPTION_SETUP:
   {
         // Remember it in mSubscriptions, if there isn't already an entry.
         if (!mSubscriptions.find(dialogHandle))
         {
            // Check that we don't have too many subscriptions.
            if (mSubscriptions.entries() <
                getResourceListServer()->getMaxRegSubscInResource())
            {
               // Add this ContactSet to mNotifyMap for the subscription.
               // (::addNotifyMapping() copies *dialogHandle.)
               getResourceListSet()->addNotifyMapping(*dialogHandle, this);

               // Remember to make a copy of *dialogHandle.
               mSubscriptions.insertKeyAndValue(new UtlString(*dialogHandle),
                                                new UtlHashMap);
            }
            else
            {
               Os::Logger::instance().log(FAC_RLS, PRI_ERR,
                             "ContactSet::subscriptionEventCallback cannot add reg subscription with dialog handle '%s', already %zu in ContactSet '%s'",
                             dialogHandle->data(), mSubscriptions.entries(),
                             mUri.data());
            }
         }
         else
         {
            Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                          "ContactSet::subscriptionEventCallback mSubscriptions element already exists for this dialog handle mUri = '%s', dialogHandle = '%s'",
                          mUri.data(),
                          dialogHandle->data());
         }
   }
   break;
   case SipSubscribeClient::SUBSCRIPTION_TERMINATED:
   {
      // Remove this ContactSet from mNotifyMap for the subscription.
      getResourceListSet()->deleteNotifyMapping(dialogHandle);

      // Delete this subscription from mSubscriptions.
      mSubscriptions.destroy(dialogHandle);

      // Update the subscriptions.
      updateSubscriptions(false);
   }
   break;
   }
}

// Process a notify event callback.
// This involves parsing the content of the callback and revising our record
// of the state for that subscription.  Then, we must regenerate the list
// of contacts and update the set of subscriptions to match the current
// contacts.
void ContactSet::notifyEventCallback(const UtlString* dialogHandle,
                                     const UtlString* content)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ContactSet::notifyEventCallback mUri = '%s', dialogHandle = '%s', content = '%s'",
                 mUri.data(), dialogHandle->data(), content->data());

   // Parse the XML and update the contact status.

   // Find the UtlHashMap for this subscription.
   UtlHashMap* state_from_this_subscr =
      dynamic_cast <UtlHashMap*> (mSubscriptions.findValue(dialogHandle));
   if (!state_from_this_subscr)
   {
      // No state for this dialogHandle, so we need to add one.
      Os::Logger::instance().log(FAC_RLS, PRI_WARNING,
                    "ContactSet::notifyEventCallback mSubscriptions element does not exist for this dialog handle mUri = '%s', dialogHandle = '%s'",
                    mUri.data(),
                    dialogHandle->data());

      // Check that we don't have too many subscriptions.
      if (mSubscriptions.entries() <
          getResourceListServer()->getMaxRegSubscInResource())
      {
         state_from_this_subscr = new UtlHashMap;
         mSubscriptions.insertKeyAndValue(new UtlString(*dialogHandle),
                                          state_from_this_subscr);
      }
      else
      {
         Os::Logger::instance().log(FAC_RLS, PRI_ERR,
                       "ContactSet::notifyEventCallback cannot add reg subscription with dialog handle '%s', already %zu in ContactSet '%s'",
                       dialogHandle->data(), mSubscriptions.entries(),
                       mUri.data());
      }
   }

   // Perform the remainder of the processing if we obtained a hash map
   // from the above processing.
   if (state_from_this_subscr)
   {
      // Initialize Tiny XML document object.
      TiXmlDocument document;
      TiXmlNode* reginfo_node;
      bool allowDirectUriSubscription = false;

      if (
         // Load the XML into it.
         document.Parse(content->data()) &&
         // Find the top element, which should be a <reginfo>.
         (reginfo_node = document.FirstChild("reginfo")) != NULL &&
         reginfo_node->Type() == TiXmlNode::ELEMENT)
      {
         // Check the state attribute.
         const char* p = reginfo_node->ToElement()->Attribute("state");
         if (p && strcmp(p, "full") == 0)
         {
            // If the state is "full", delete the current state.
            state_from_this_subscr->destroyAll();
            Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                          "ContactSet::notifyEventCallback clearing state");
         }

         // Find all the <registration> elements for this URI.
         for (TiXmlNode* registration_node = 0;
              (registration_node =
               reginfo_node->IterateChildren("registration",
                                             registration_node));
            )
         {
            // Do not test the aor attribute of <registration> elements
            // because the reg event server may be operating with the real
            // AOR for this URI, whereas we may have been told of an alias.

             // Get the state attribute of registration node
             TiXmlElement* registration_element = registration_node->ToElement();
             const char* registration_state = registration_element->Attribute("state");
             // The state will be "init" when no contacts are registered for this AOR
             if (strcmp(registration_state, "init") == 0)
             {
                 // Allow direct dialog subscriptions to the AOR when the AOR is not registered
                 // and no contact information can be retrieved from registrar through reg event subscriptions.
                 // This behavior is needed to monitor static extensions like park orbit extensions. These
                 // entities will never register to the registrar.
                 allowDirectUriSubscription = true;
                 Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                         "ContactSet::notifyEventCallback allowDirectUriSubscription true");

                 // Continue with the next <registration> element because the current one has no contacts to iterate.
                 continue;
             }

            // Find all the <contact> elements.
            for (TiXmlNode* contact_node = 0;
                 (contact_node =
                  registration_node->IterateChildren("contact",
                                                     contact_node));
               )
            {
               TiXmlElement* contact_element = contact_node->ToElement();

               // Get the state attribute.
               const char* state = contact_element->Attribute("state");
               // Get the id attribute
               const char* id = contact_element->Attribute("id");

               // Get the Contact URI for the phone. If a GRUU address is present we should
               // use that as the Contact URI. Otherwise, use the contact URI present in the
               // "uri" element, and append any path headers that are present to the ROUTE
               // header parameter in the URI. This will ensure proper routing in HA systems.
               // Please refer to XECS-1694 for more details.

               UtlString* uri_allocated = new UtlString;
               UtlBoolean check_uri = TRUE;
               TiXmlNode* pub_gruu_node = contact_element->FirstChild("gr:pub-gruu");
               if (pub_gruu_node)
               {
                  TiXmlElement* pub_gruu_element = pub_gruu_node->ToElement();
                  UtlString pub_gruu_uri(pub_gruu_element->Attribute("uri"));
                  if (!pub_gruu_uri.isNull())
                  {
                     // Check the URI Scheme. Only accept the GRUU address if it is of either
                     // a sip or sips scheme
                     Url tmp(pub_gruu_uri, TRUE);
                     Url::Scheme uriScheme = tmp.getScheme();
                     if(Url::SipUrlScheme == uriScheme ||
                        Url::SipsUrlScheme == uriScheme)
                     {
                        tmp.removeAngleBrackets();
                        tmp.getUri(*uri_allocated);
                        check_uri = FALSE;
                     }
                  }
               }

               // If we did not find a GRUU address, then use the address in the "uri" element as the
               // contact URI, and check for path headers.
               if (check_uri)
               {
                  TiXmlNode* u = contact_element->FirstChild("uri");
                  if (u)
                  {
                     textContentShallow(*uri_allocated, u);

                     // Iterate through all the path header elements. Path headers are stored in the
                     // "unknown-param" elements that have a "name" attribute value of "path".
                     for (TiXmlNode* unknown_param_node = 0;
                          (unknown_param_node = contact_node->IterateChildren("unknown-param",
                                                                               unknown_param_node));
                          )
                     {
                        TiXmlElement* unknown_param_element = unknown_param_node->ToElement();
                        UtlString path(unknown_param_element->Attribute("name"));
                        if(0 == path.compareTo("path"))
                        {
                           UtlString pathVector;
                           textContentShallow(pathVector, unknown_param_node);
                           if(!pathVector.isNull())
                           {
                              Url contact_uri(*uri_allocated, TRUE);

                              // there is already a Route header parameter in the contact; append it to the
                              // Route derived from the Path vector.
                              UtlString existingRouteValue;
                              if ( contact_uri.getHeaderParameter(SIP_ROUTE_FIELD, existingRouteValue))
                              {
                                 pathVector.append(SIP_MULTIFIELD_SEPARATOR);
                                 pathVector.append(existingRouteValue);
                              }
                              contact_uri.setHeaderParameter(SIP_ROUTE_FIELD, pathVector);
                              contact_uri.removeAngleBrackets();
                              contact_uri.getUri(*uri_allocated);
                           }
                        }
                     }
                  }
               }

               // Only process <contact> elements that have the needed values.
               if (state && state[0] != '\0' &&
                   id && id[0] != '\0' &&
                   !uri_allocated->isNull())
               {
                  UtlString* id_allocated = new UtlString(id);

                  if (strcmp(state, "active") == 0)
                  {
                     // Add the contact if it is not already present.
                     if (!state_from_this_subscr->find(id_allocated))
                     {
                        // Prepend the registration Call-Id and ';' to *uri_allocated..
                        uri_allocated->insert(0, ';');
                        const char* call_id = contact_element->Attribute("callid");
                        if (call_id)
                        {
                           uri_allocated->insert(0, call_id);
                        }

                        // Check that we don't have too many contacts.
                        if (state_from_this_subscr->entries() <
                            getResourceListServer()->getMaxContInRegSubsc())
                        {
                           // Insert the registration record.
                           if (state_from_this_subscr->insertKeyAndValue(id_allocated,
                                                                         uri_allocated))
                           {
                              Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                                            "ContactSet::notifyEventCallback adding id = '%s' Call-Id;URI = '%s'",
                                            id, uri_allocated->data());
                              id_allocated = NULL;
                              uri_allocated = NULL;
                           }
                           else
                           {
                              Os::Logger::instance().log(FAC_RLS, PRI_ERR,
                                            "ContactSet::notifyEventCallback adding id = '%s' Call-Id;URI = '%s' failed",
                                            id_allocated->data(), uri_allocated->data());
                              Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                                            "ContactSet::notifyEventCallback *state_from_this_subscr is:");
                              UtlHashMapIterator itor(*state_from_this_subscr);
                              UtlContainable* k;
                              while ((k = itor()))
                              {
                                 UtlContainable* v = itor.value();
                                 Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                                               "ContactSet::notifyEventCallback (*state_from_this_subscr)['%s'] = '%s'",
                                               (dynamic_cast <UtlString*> (k))->data(),
                                               (dynamic_cast <UtlString*> (v))->data());
                              }
                           }
                        }
                        else
                        {
                           Os::Logger::instance().log(FAC_RLS, PRI_ERR,
                                         "ContactSet::notifyEventCallback cannot add Call-Id;RUI '%s', already %zu in ContactSet '%s' subscription '%s'",
                                         uri_allocated->data(),
                                         state_from_this_subscr->entries(),
                                         mUri.data(), dialogHandle->data());
                        }
                     }
                  }
                  else if (strcmp(state, "terminated") == 0)
                  {
                     // Delete it from the contact state.
                     state_from_this_subscr->destroy(id_allocated);
                     Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                                   "ContactSet::notifyEventCallback deleting id = '%s'",
                                   id);
                  }

                  // Free id_allocated, if it is not pointed to by a data
                  // structure, which is indicated by setting it to NULL.
                  if (id_allocated)
                  {
                     delete id_allocated;
                  }
               }
               else
               {
                  Os::Logger::instance().log(FAC_RLS, PRI_ERR,
                                "ContactSet::notifyEventCallback <contact> element with id = '%s' is missing id, state, and/or URI",
                                id ? id : "(missing)");
               }
               // Free uri_allocated, if it is not pointed to by a data
               // structure, which is indicated by setting it to NULL.
               if (uri_allocated)
               {
                  delete uri_allocated;
               }
            }
         }
      }
      else
      {
         // Error parsing the contents.
         Os::Logger::instance().log(FAC_RLS, PRI_ERR,
                       "ContactSet::notifyEventCallback malformed reg event content for mUri = '%s'",
                       mUri.data());
      }

      // Update the subscriptions we maintain to agree with the new state.
      updateSubscriptions(allowDirectUriSubscription);
   }
}

// Update the subscriptions we maintain to agree with the current contact state
void ContactSet::updateSubscriptions(bool allowDirectUriSubscription)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ContactSet::updateSubscriptions mUri = '%s'",
                 mUri.data());

   // First, scan mSubscriptions to construct a list of all the
   // call-id/contact combinations.
   // (If a phone reboots and starts registering with a different Call-Id,
   // the call-id/contact combination will be different even if the contact
   // URI is unchanged.  So the new registration will appear to be different
   // to this machinery, and it will establish a new SubscriptionSet to
   // the contact URI.  This compensates for the fact that the previous
   // SubscriptionSet to the contact URI appears to the RLS to be
   // working but the phone no longer knows of the subscription.  The
   // reg events will eventually terminate the old combination and we
   // will delete its SubscriptionSet.)
   UtlHashBag callid_contacts;

   UtlHashMapIterator subs_itor(mSubscriptions);
   while (subs_itor())
   {
      if (Os::Logger::instance().willLog(FAC_RLS, PRI_DEBUG))
      {
         Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                       "ContactSet::updateSubscriptions subscription '%s'",
                       (dynamic_cast <UtlString*> (subs_itor.key()))->data());
      }
      UtlHashMap* contact_state =
         dynamic_cast <UtlHashMap*> (subs_itor.value());
      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ContactSet::updateSubscriptions contact_state = %p",
                    contact_state);
      UtlHashMapIterator contact_itor(*contact_state);
      while (contact_itor())
      {
         UtlString* contact =
            dynamic_cast <UtlString*> (contact_itor.value());
         if (Os::Logger::instance().willLog(FAC_RLS, PRI_DEBUG))
         {
            Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                          "ContactSet::updateSubscriptions contact id '%s', Call-Id/URI '%s'",
                          (dynamic_cast <UtlString*> (contact_itor.key()))->data(),
                          contact->data());
         }
         // Check if the contact is already in callid_contacts.
         if (!callid_contacts.find(contact))
         {
            // If not, add it.
            UtlString* c = new UtlString(*contact);
            callid_contacts.insert(c);
            Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                          "ContactSet::updateSubscriptions contact added");
         }
      }
   }

   // If the list of callid_contacts is empty, it means that no contacts have been obtained
   // so far. This could happen because the registrar is down or does not support reg events.
   // In such cases if it is permitted (allowDirectResourceSubscription is enabled) the
   // mUri is added as the default contactwith an empty registration Call-Id) so that
   // a direct subscription will be attempted to the mUri.
   //
   // WARNING: Beware that the mechanism below is not correctly implemented and if used
   // it will result in a dialog subscription being done before a reg subscription,
   // which will be deleted and redone when the notify for reg subscription is received.
   // The same will happen when the reg subscription is terminated: the dialog subscription
   // will also be terminated and anew dialog subscription will be generated.
   if (callid_contacts.isEmpty() && allowDirectUriSubscription)
   {
      Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                    "ContactSet::updateSubscriptions adding default contact mUri = '%s'",
                    mUri.data());
      UtlString* c = new UtlString(";");
      c->append(mUri);
      callid_contacts.insert(c);
   }

   // Now that we have a clean list of callid_contacts, update
   // SubscriptionSets to match it.

   // If we both terminate subscriptions and create subscriptions,
   // wait a short while to allow the terminations to complete.  This
   // should not be necessary, but it makes life easier on Polycom
   // phones which (at this time) cannot support two subscriptions at
   // a time, and if the termination of the old subscription arrives
   // after the initiation of the new subscription, the new
   // subscription will be lost.
   // This variable tracks whether such a wait is needed before a
   // subscription is started.
   bool wait_after_subscription_ended = false;

   // Iterate through the list of SubscriptionSets and remove any that aren't
   // in callid_contacts.
   {
      UtlHashMapIterator itor(mSubscriptionSets);
      UtlString* ss;
      while ((ss = dynamic_cast <UtlString*> (itor())))
      {
         if (!callid_contacts.find(ss))
         {
            Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                          "ContactSet::updateSubscriptions deleting subscription for '%s' in mUri = '%s'",
                          ss->data(), mUri.data());
            mSubscriptionSets.destroy(ss);
            wait_after_subscription_ended = true;
         }
      }
   }

    // Iterate through callid_contacts and add a SubscriptionSet for
    // any that aren't in mSubscriptionSets.
    // We don't limit the number of additions here, as the size of
    // callid_contacts is guarded by the tests in notifyEventCallback.
    {
        UtlHashBagIterator itor(callid_contacts);
        UtlString* callid_contact;
        long subscription_wait_msec = getResourceListServer()->getChangeDelay() / 2;
        int subscriptionSetCount = 0;

        while ((callid_contact = dynamic_cast <UtlString*> (itor())))
        {
            // If we both terminate subscriptions and create subscriptions,
            // wait a short while to allow the terminations to complete.
            // Note that this wait must be less than the bulk add/delete
            // change delay, as that is how fast ResourceListFileReader
            // generates requests to the ResourceListServer task, so a longer
            // wait would prevent the ResourceListServer task from servicing
            // requests as fast as it received them.
        	if (!mSubscriptionSets.find(callid_contact))
        	{
				if (wait_after_subscription_ended)
				{
					Os::Logger::instance().log(FAC_SAA, PRI_DEBUG,
							"ContactSet::updateSubscriptions waiting for %d msec",
							subscription_wait_msec);

					OsTime offset(subscription_wait_msec);
					bool ret = getResourceListSet()->addSubscriptionSetByTimer(*callid_contact, this, offset);
					if (ret)
					{
						subscriptionSetCount++;
						/* for each successful fire timer the wait is incremented */
						subscription_wait_msec += SUBSCRIPTION_WAIT_INCR_MSEC;
					}
				}
				else
				{
					subscriptionSetCount++;
					addSubscriptionSet(callid_contact);
				}
        	}
        }

        Os::Logger::instance().log(FAC_SAA, PRI_DEBUG,
                "ContactSet::updateSubscriptions added '%d' new SubscriptionSets",
                subscriptionSetCount);

    }

   // Free callid_contacts.
   callid_contacts.destroyAll();
}

// Add to the HttpBody the current state of the resource instances.
void ContactSet::generateBody(UtlString& rlmi,
                              HttpBody& body,
                              UtlBoolean consolidated,
                              const UtlString& displayName) const
{
   // Iterate through the list of SubscriptionSet's and call their
   // generateBody methods.
   UtlHashMapIterator itor(mSubscriptionSets);
   UtlString* ss;
   while ((ss = dynamic_cast <UtlString*> (itor())))
   {
      (dynamic_cast <SubscriptionSet*> (itor.value()))->
         generateBody(rlmi, body, consolidated, displayName);
   }
}

// Remove dialogs in terminated state and terminated resource instances.
void ContactSet::purgeTerminated()
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "ContactSet::purgeTerminated mUri = '%s'",
                 mUri.data());

   // Iterate through the list of SubscriptionSet's and call their
   // purgeTerminated methods.
   UtlHashMapIterator itor(mSubscriptionSets);
   UtlString* ss;
   while ((ss = dynamic_cast <UtlString*> (itor())))
   {
      (dynamic_cast <SubscriptionSet*> (itor.value()))->
         purgeTerminated();
   }
}

/* ============================ ACCESSORS ================================= */

/* //////////////////////////// PUBLIC //////////////////////////////////// */

/* ============================ INQUIRY =================================== */

// Dump the object's internal state.
void ContactSet::dumpState() const
{
   // indented 8, 10, and 12

   Os::Logger::instance().log(FAC_RLS, PRI_INFO,
                 "\t        ContactSet %p mUri = '%s', mSubscriptionEarlyDialogHandle = '%s'",
                 this, mUri.data(), mSubscriptionEarlyDialogHandle.data());

   UtlHashMapIterator itor(mSubscriptions);
   UtlString* handle;
   while ((handle = dynamic_cast <UtlString*> (itor())))
   {
      UtlHashMap* h = dynamic_cast <UtlHashMap*> (itor.value());
      Os::Logger::instance().log(FAC_RLS, PRI_INFO,
                    "\t          mSubscriptions{'%s'} %p",
                    handle->data(), h);
      UtlHashMapIterator itor2(*h);
      UtlString* id;
      while ((id = dynamic_cast <UtlString*> (itor2())))
      {
         UtlString* value = dynamic_cast <UtlString*> (itor2.value());
         int semi = value->index(';');
         if (semi == UTL_NOT_FOUND)
         {
            semi = value->length();
         }
         const char* v = value->data();
         Os::Logger::instance().log(FAC_RLS, PRI_INFO,
                       "\t            id = '%s', Call-Id = '%.*s', URI = '%s'",
                       id->data(), semi, v, v + semi + 1);
      }
   }

   UtlHashMapIterator itor3(mSubscriptionSets);
   while ((handle = dynamic_cast <UtlString*> (itor3())))
   {
      SubscriptionSet* ss = dynamic_cast <SubscriptionSet*> (itor3.value());
      ss->dumpState();
   }
}

/**
 * Get the ContainableType for a UtlContainable-derived class.
 */
UtlContainableType ContactSet::getContainableType() const
{
   return ContactSet::TYPE;
}

void ContactSet::addSubscriptionSet(const UtlString* callidContact)
{
    Os::Logger::instance().log(FAC_SAA, PRI_DEBUG,
            "ContactSetSet::addSubscriptionSet this = %p, mUri = '%s' callid;contact = '%s'",
             this, mUri.data(), callidContact->data());

     // Get the contact URI into a UtlString.
     UtlString uri(callidContact->data() +
                   callidContact->index(';') +
                   1);

    // Create the subscription set
	SubscriptionSet* ss = new SubscriptionSet(mResource,  uri);

	// Add the subscription to the set.
	mSubscriptionSets.insertKeyAndValue(new UtlString(*callidContact), ss);

	Os::Logger::instance().log(FAC_SAA, PRI_DEBUG,
			"ContactSet::addSubscriptionSet "
			"added SubscriptionSet for uri = '%s'",
			uri.data() );
}

/* //////////////////////////// PROTECTED ///////////////////////////////// */

/* //////////////////////////// PRIVATE /////////////////////////////////// */


/* ============================ FUNCTIONS ================================= */
