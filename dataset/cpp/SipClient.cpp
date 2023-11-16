//
// Copyright (C) 2007 Pingtel Corp., certain elements licensed under a Contributor Agreement.
// Contributors retain copyright to elements licensed under a Contributor Agreement.
// Licensed to the User under the LGPL license.
//
//
// $$
////////////////////////////////////////////////////////////////////////
//////

// SYSTEM INCLUDES
#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <unistd.h>

// APPLICATION INCLUDES
#include <net/SipMessage.h>
#include <net/SipClient.h>
#include <net/SipMessageEvent.h>
#include <net/SipProtocolServerBase.h>
#include <net/SipUserAgentBase.h>
#include <net/Instrumentation.h>

#include <os/OsDateTime.h>
#include <os/OsDatagramSocket.h>
#include <os/OsStatus.h>
#include <os/OsLogger.h>
#include <os/OsEvent.h>

#include <utl/XmlContent.h>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <sstream>
#include <vector>

#define SIP_DEFAULT_RTT 500
// The time in milliseconds that we allow poll() to wait.
// This must be short, as the run() loop must wake up periodically to check
// if the client's thread is being shut down.
// And the SipUserAgent, when garbage collecting idle clients, waits for
// the clients to finish shutting down.
#define POLL_TIMEOUT 100

#define LOG_TIME
//#define TEST_SOCKET

// EXTERNAL FUNCTIONS
// EXTERNAL VARIABLES
// CONSTANTS

// All requests must contain at least 72 characters:
/*
X Y SIP/2.0 \n\r
i: A\n\r
f: B\n\r
t: C\n\r
c: 1\n\r
v: SIP/2.0/UDP D\n\r
l: 0 \n\r
\n\r

*/
// However to be tolerant of malformed messages we allow smaller:
#define MINIMUM_SIP_MESSAGE_SIZE 30
#define MAX_UDP_PACKET_SIZE (1024 * 64)

// STATIC VARIABLE INITIALIZATIONS

const UtlContainableType SipClient::TYPE = "SipClient";

const UtlContainableType SipClientSendMsg::TYPE = "SipClientSendMsg";

SipTransportRateLimitStrategy SipClient::_rateLimit;
SipTransportRateLimitStrategy& SipClient::rateLimit()
{
    return SipClient::_rateLimit;
}

// Methods for SipClientSendMsg.

// Constructor
SipClientSendMsg::SipClientSendMsg(const unsigned char msgType,
                                   const unsigned char msgSubType,
                                   const SipMessage& message,
                                   const char* address, int port) :
   OsMsg(msgType, msgSubType),
   mpMessage(new SipMessage(message)),
   mAddress(strdup(address)),
   mPort(port)
{
}

//Constructor for Keep Alive with no actual message
SipClientSendMsg::SipClientSendMsg(const unsigned char msgType,
                                   const unsigned char msgSubType,
                                   const char* address, int port) :
   OsMsg(msgType, msgSubType),
   mpMessage(0),
   mAddress(strdup(address)),
   mPort(port)
{
}



//:Copy constructor
SipClientSendMsg::SipClientSendMsg(const SipClientSendMsg& rOsMsg) :
   OsMsg(rOsMsg),
   mpMessage(new SipMessage(*rOsMsg.mpMessage)),
   mAddress(strdup(rOsMsg.mAddress)),
   mPort(rOsMsg.mPort)
{
}

//:Destructor
SipClientSendMsg::~SipClientSendMsg()
{
   free(mAddress);

   // mpMessage may have been nulled by detachMessage, so we have to
   // test it before deleting it.
   if (mpMessage)
   {
      delete mpMessage;
   }
}

//:Create a copy of this msg object (which may be of a derived type)
OsMsg* SipClientSendMsg::createCopy(void) const
{
   return new SipClientSendMsg(*this);
}

//:Assignment operator
SipClientSendMsg& SipClientSendMsg::operator=(const SipClientSendMsg& rhs)
{
   if (this != &rhs)            // handle the assignment to self case
   {
      OsMsg::operator=(rhs);
      mpMessage = new SipMessage(*rhs.mpMessage);
      free(mAddress);
      mAddress = strdup(rhs.mAddress);
      mPort = rhs.mPort;
   }

   return *this;
}

/// Return the SipMessage component, and NULL the SipMessage component,
/// so the SipClientSendMsg no longer owns it.
SipMessage* SipClientSendMsg::detachMessage(void)
{
   SipMessage* ret = mpMessage;
   mpMessage = NULL;
   return ret;
}

// Component accessors.
const SipMessage* SipClientSendMsg::getMessage(void) const
{
   return mpMessage;
}

const char* SipClientSendMsg::getAddress(void) const
{
   return mAddress;
}

int SipClientSendMsg::getPort(void) const
{
   return mPort;
}

/* //////////////////////////// PUBLIC //////////////////////////////////// */

/* ============================ CREATORS ================================== */

// Constructor
SipClient::SipClient(OsSocket* socket,
                     SipProtocolServerBase* pSipServer,
                     SipUserAgentBase* sipUA,
                     const char* taskNameString,
                     UtlBoolean bIsSharedSocket ) :
   OsServerTaskWaitable(taskNameString),
   mClientSocket(socket),
   mSocketType(socket ? socket->getIpProtocol() : OsSocket::UNKNOWN),
   mpSipUserAgent(sipUA),
   mpSipServer(pSipServer),
   mRemoteViaPort(PORT_NONE),
   mRemoteReceivedPort(PORT_NONE),
   mSocketLock(OsBSem::Q_FIFO, OsBSem::FULL),
   mFirstResendTimeoutMs(SIP_DEFAULT_RTT * 4), // for first transcation time out
   mbSharedSocket(bIsSharedSocket),
   mWriteQueued(FALSE),
   mbTcpOnErrWaitForSend(TRUE)
{
   touch();

   if (mClientSocket)
   {
       mClientSocket->getRemoteHostName(&mRemoteHostName);
       mClientSocket->getRemoteHostIp(&mRemoteSocketAddress, &mRemoteHostPort);
       mClientSocket->getLocalHostIp(&mLocalHostAddress);
       mLocalHostPort = mClientSocket->getLocalHostPort();

       Os::Logger::instance().log(FAC_SIP, PRI_INFO,
                     "SipClient[%s]::_ created %s %s socket %d: host '%s' '%s' port %d, local IP '%s' port %d",
                     mName.data(),
                     OsSocket::ipProtocolString(mSocketType),
                     mbSharedSocket ? "shared" : "unshared",
                     socket->getSocketDescriptor(),
                     mRemoteHostName.data(), mRemoteSocketAddress.data(), mRemoteHostPort,
                     mLocalHostAddress.data(), mLocalHostPort
                     );
   }
}

// Destructor
SipClient::~SipClient()
{
    Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                  "SipClient[%s]::~ called",
                  mName.data());

    // Tell the associated thread to shut itself down.
    requestShutdown();

    // Do not delete the event listers, as they are not subordinate.

    // Free the socket
    if(mClientSocket)
    {
        // Close the socket to unblock the run method
        // in case it is blocked in a waitForReadyToRead or
        // a read on the mClientSocket.  This should also
        // cause the run method to exit.
        if (!mbSharedSocket)
        {
           Os::Logger::instance().log(FAC_SIP, PRI_DEBUG, "SipClient[%s]::~ %p socket %p closing %s socket",
                         mName.data(), this,
                         mClientSocket, OsSocket::ipProtocolString(mSocketType));
           mClientSocket->close();
        }

        // Wait for the task to exit so that it does not
        // reference the socket or other members after they
        // get deleted.
        if(isStarted() || isShuttingDown())
        {
            waitUntilShutDown();
        }

        if (!mbSharedSocket)
        {
            delete mClientSocket;
        }
        mClientSocket = NULL;
    }
    else if(isStarted() || isShuttingDown())
    {
        // It should not get here but just in case
        waitUntilShutDown();
    }
}

/* ============================ MANIPULATORS ============================== */

// Handles an incoming message (from the message queue).
UtlBoolean SipClient::handleMessage(OsMsg& eventMessage)
{
   UtlBoolean messageProcessed = FALSE;

   int msgType = eventMessage.getMsgType();
   int msgSubType = eventMessage.getMsgSubType();

   if(msgType == OsMsg::OS_EVENT &&
      msgSubType == SipClientSendMsg::SIP_CLIENT_SEND)
   {
      // Queued SIP message to send.

      // Call sendMessage method to send the SIP message (or to
      // store its contents to be sent).
      SipClientSendMsg* sendMsg =
         dynamic_cast <SipClientSendMsg*> (&eventMessage);
      sendMessage(*sendMsg->getMessage(), sendMsg->getAddress(),
                  sendMsg->getPort());
      messageProcessed = TRUE;
      // *eventMessage will be deleted by ::run(), as usual.
      // Its destructor will free any storage owned by it.
   }

   return (messageProcessed);
}

// Queue a message to be sent to the specified address and port.
UtlBoolean SipClient::sendTo(SipMessage& message,
                             const char* address,
                             int port)
{
   UtlBoolean sendOk;

   if (mClientSocket)
   {
      // If port == PORT_NONE, get the correct default port for this
      // transport method.
      int portToSendTo = ( port == PORT_NONE ? defaultPort() : port );

      // We are about to post a message that will cause the
      // SIP message to be sent.  Notify the user agent so
      // that it can offer the message to all its registered
      // output processors.

      ssize_t msgLength = 0;
      UtlString msgText;
      message.getBytes(&msgText, &msgLength, true);
      if (msgLength)
      {
        system_tap_sip_tx(
             mLocalHostAddress.data(), portIsValid(mLocalHostPort) ? mLocalHostPort : defaultPort(),
             address, portToSendTo,
             msgText.data(), msgLength);

        mpSipUserAgent->executeAllSipOutputProcessors( message, address, portToSendTo );
      }

      // Create message to queue.
      SipClientSendMsg sendMsg(OsMsg::OS_EVENT,
                               SipClientSendMsg::SIP_CLIENT_SEND,
                               message, address,
                               portToSendTo );

      // Post the message to the task's queue.
      
      if (mSocketType != OsSocket::UDP)
      {
        OsStatus status = postMessage(sendMsg, OsTime::NO_WAIT);
        sendOk = status == OS_SUCCESS;
      }
      else
      {
        sendOk = handleMessage(sendMsg);
      }
      
      if (!sendOk)
      {
         Os::Logger::instance().log(FAC_SIP, PRI_ERR,
                       "SipClient[%s]::sendTo attempt to post message failed",
                       mName.data());
      }
   }
   else
   {
      Os::Logger::instance().log(FAC_SIP, PRI_CRIT,
                    "SipClient[%s]::sendTo called for client without socket",
                    mName.data()
         );
      sendOk = FALSE;
   }

   return sendOk;
}

// Remove and report(if requested) all stored message content (because the socket
// is not usable).
// This is the default, do-nothing, implementation, to be overridden
// by classes that use this functionality.
void SipClient::emptyBuffer(bool reportError)
{
   assert(FALSE);
}

// Continue sending stored message content (because the socket
// is now writable).
// This is the default, do-nothing, implementation, to be overridden
// by classes that use this functionality.
void SipClient::writeMore(void)
{
   assert(FALSE);
}

UtlBoolean SipClient::isSharedSocket( void ) const
{
    return mbSharedSocket;
}

void SipClient::touch()
{
   OsTime time;
   OsDateTime::getCurTimeSinceBoot(time);
   touchedTime = time.seconds();
   //Os::Logger::instance().log(FAC_SIP, PRI_DEBUG, "SipClient[%s]::touch client: %p time: %d\n",
   //             mName.data(), this, touchedTime);
}

/* ============================ ACCESSORS ================================= */

void SipClient::getClientNames(UtlString& clientNames) const
{
    char portString[40];

    // host DNS name
    sprintf(portString, "%d", mRemoteHostPort);
    clientNames = " remote host: ";
    clientNames.append(mRemoteHostName);
    clientNames.append(":");
    clientNames.append(portString);

    // host IP address
    clientNames.append(" remote IP: ");
    clientNames.append(mRemoteSocketAddress);
    clientNames.append(":");
    clientNames.append(portString);

    // via address
    clientNames.append(" remote Via address: ");
    clientNames.append(mRemoteViaAddress);
    clientNames.append(":");
    clientNames.appendNumber(mRemoteViaPort);

    // received address
    clientNames.append(" received address: ");
    clientNames.append(mReceivedAddress);
    clientNames.append(":");
    clientNames.appendNumber(mRemoteReceivedPort);
}

long SipClient::getLastTouchedTime() const
{
   return (touchedTime);
}

/* ============================ INQUIRY =================================== */

UtlBoolean SipClient::isOk()
{
  
    
   return OsServerTaskWaitable::isOk() && mClientSocket->isOk() && isNotShut();
}

bool SipClient::isWritable()
{
  static const char * KEEP_ALIVE = "\r\n\r\n";
  static const int KEEP_ALIVE_SIZE = strlen(KEEP_ALIVE);
  static const int KEEP_ALIVE_WAIT_TIME = 10; // milliseconds
  
  bool writable = false;
  
  if (isOk())
  {
    //
    // Send CRLF/CRLF with maximum wait time of 10 milliseconds
    //
    if (OsSocket::isFramed(mClientSocket->getIpProtocol()))
    {
      //
      // UDP sockets are shared sockets.  This is only applicable for TCP/TLS clients
      //
      writable = true;
    }
    else
    {
      //
      // TCP and TLS.  Make sure sockets are writable
      //
      writable = mClientSocket->write(KEEP_ALIVE, KEEP_ALIVE_SIZE, KEEP_ALIVE_WAIT_TIME) > 0;
      
      if (!writable)
      {
        //
        // Close this socket.  The other end closed the connection.
        // This also assures that garbage collector collects this client 
        // in the next iteration because isOk() would now return false.
        //
        mClientSocket->close();
      }
    }
  }
  
  return writable;
}

UtlBoolean SipClient::isAcceptableForDestination( const UtlString& hostName, int hostPort, const UtlString& localIp )
{
   UtlBoolean isAcceptable = FALSE;

   // Only accept it if the local IP is correct.
   if (0 == strcmp(getLocalIp(), localIp))
   {
      if( isSharedSocket() )
      {
         // A shared socket implies that it is not connected to any specific far-end host and
         // therefore can be used to send to any destination.
         isAcceptable = TRUE;
      }
      else
      {
         int tempHostPort = portIsValid(hostPort) ? hostPort : defaultPort();

#ifdef TEST_SOCKET
         Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                       "SipClient[%s]::isAcceptableForDestination hostName = '%s', tempHostPort = %d, mRemoteHostName = '%s', mRemoteHostPort = %d, mRemoteSocketAddress = '%s', mReceivedAddress = '%s', mRemoteViaAddress = '%s'",
                       mName.data(),
                       hostName.data(), tempHostPort, mRemoteHostName.data(), mRemoteHostPort, mRemoteSocketAddress.data(), mReceivedAddress.data(), mRemoteViaAddress.data());
#endif

         // If the ports match and the host is the same as either the
         // original name that the socket was constructed with or the
         // name it was resolved to (usually an IP address).
         if (   mRemoteHostPort == tempHostPort
             && (   hostName.compareTo(mRemoteHostName, UtlString::ignoreCase) == 0
                 || hostName.compareTo(mRemoteSocketAddress, UtlString::ignoreCase) == 0))
         {
             isAcceptable = TRUE;
         }
         else if (   mRemoteReceivedPort == tempHostPort
                  && hostName.compareTo(mReceivedAddress, UtlString::ignoreCase) == 0)
         {
             isAcceptable = TRUE;
         }
         else if (mRemoteViaPort == tempHostPort && (   hostName.compareTo(mRemoteHostName, UtlString::ignoreCase) == 0
                 || hostName.compareTo(mRemoteSocketAddress, UtlString::ignoreCase) == 0))
         {
           isAcceptable = TRUE;
         }
         else if (   mRemoteViaPort == tempHostPort
                  && hostName.compareTo(mRemoteViaAddress, UtlString::ignoreCase) == 0)
         {
             // Cannot trust what the other side said was their IP address
             // as this is a bad spoofing/denial of service hole
             Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                           "SipClient[%s]::isAcceptableForDestination matches %s:%d but is not trusted",
                           mName.data(),
                           mRemoteViaAddress.data(), mRemoteViaPort);
         }
      }
   }

   // Make sure client is okay before declaring it acceptable
   if( isAcceptable && !isWritable() )
   {
      Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                    "SipClient[%s]::isAcceptableForDestination('%s', %d, '%s')"
                    " Client matches host/port but is not OK",
                    mName.data(), hostName.data(), hostPort, localIp.data());
      isAcceptable = FALSE;
   }
   return(isAcceptable);
}

const UtlString& SipClient::getLocalIp()
{
    return mClientSocket->getLocalIp();
}

/* //////////////////////////// PROTECTED ///////////////////////////////// */

// Thread execution code.
int SipClient::run(void* runArg)
{
  static const int SIZE_OF_PING = strlen("\r\n\r\n");
  
   OsMsg*    pMsg = NULL;
   OsStatus  res;
   // Buffer to hold data read from the socket but not yet parsed
   // into incoming SIP messages.
   UtlString readBuffer;
   bool      waitingToReportErr  = FALSE;    // controls whether to read-select on socket
   bool      tcpOnErrWaitForSend = TRUE;
   int       repeatedEOFs = 0;

   Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                 "SipClient[%s]::run start  "
                 "tcpOnErrWaitForSend-%d waitingToReportErr-%d mbTcpOnErrWaitForSend-%d repeatedEOFs-%d",
                 mName.data(), tcpOnErrWaitForSend, waitingToReportErr,
                 mbTcpOnErrWaitForSend, repeatedEOFs);

   // Wait structure:
   struct pollfd fds[2];
   // Incoming message on the message queue (to be sent on the socket).
   fds[0].fd = mPipeReadingFd;
   // Socket ready to write (to continue sending message).
   // Socket ready to read (message to be received).

   do
   {
      assert(repeatedEOFs < 20);
      // The file descriptor for the socket may changemsg->getSendAddress(&fromIpAddress, &fromPort);, as OsSocket's
      // can be re-opened.
      fds[1].fd = mClientSocket->getSocketDescriptor();

      // Initialize the revents members.
      // This may not be necessary (the man page is not clear), but
      // Valgrind flags 'fds[*].revents' as undefined if they aren't
      // initialized.
      fds[0].revents = 0;
      fds[1].revents = 0;

      fds[0].events = POLLIN;   // only read-select on pipe

      // For non-blocking connect failures, don't read-select on socket if
      // the initial read showed an error but we have to wait to report it.
      if (!waitingToReportErr)
      {
          // This is the normal path.
          // Read the socket only if the socket is not shared.
          // If it is shared, the ancestral SipClient will read it.
          // If multiple threads attempt to read the socket, poll() may
          // succeed but another may read the data, leaving us to block on
          // read.
          fds[1].events = mbSharedSocket ? 0 : POLLIN;

          // Set wait for writing the socket if there is queued messages to
          // send.
          if (mWriteQueued)
          {
             // Wait for output on the socket to not block.
             fds[1].events |= POLLOUT;
          }

      }
      else
      {
          // just waiting to report error, ignore the socket
          fds[1].fd =-1;
          fds[1].events = 0;
      }

      // If there is residual data in the read buffer,
      // pretend the socket is ready to read.
      if (!readBuffer.isNull())
      {
         fds[1].revents = POLLIN;
      }
      else
      {
         // Otherwise, call poll() to wait.
         int resPoll = poll(&fds[0], sizeof (fds) / sizeof (fds[0]),
                        POLL_TIMEOUT);
         assert(resPoll >= 0 || (resPoll == -1 && errno == EINTR));
         if (resPoll != 0)
         {
             Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                           "SipClient[%s]::run "
                           "resPoll= %d revents: fd[0]= %x fd[1]= %x",
                           mName.data(),
                           resPoll, fds[0].revents, fds[1].revents );
         }
      }

      if ((fds[1].revents & (POLLERR | POLLHUP)) != 0)
      {
          Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                        "SipClient[%s]::run "
                        "SipMessage::poll error(%d) ",
                        mName.data(), errno);

          if (OsSocket::isFramed(mClientSocket->getIpProtocol()))
          {
              Os::Logger::instance().log(FAC_SIP, PRI_ERR,
                            "SipClient[%s]::run "
                            "SipMessage::poll error(%d) got POLLERR | POLLHUP on UDP socket",
                            mName.data(), errno);

          }
          else	// eg. tcp socket
          // This client's socket is a connection-oriented protocol and the
          // connection has been terminated (probably by the remote end).
          // We must terminate the SipClient.
          // We drop the queued messages, but we do not report them to
          // SipUserAgent as failed sends.  This will cause SipUserAgent to
          // retry the send using the same transport (rather than continuing
          // to the next transport), which should cause a new connection to
          // be made to the remote end.
          {
              // On non-blocking connect failures, we need to get the first send message
              // in order to successfully trigger the protocol fallback mechanism
              if (!tcpOnErrWaitForSend)
              {
                 // Return all buffered messages with a transport error indication.
                 emptyBuffer(TRUE);
                 clientStopSelf();
              }
              else
              {
                 fds[1].revents &= ~(POLLERR | POLLHUP);  // clear error bit if waiting
                 waitingToReportErr = TRUE;
              }
          }
      }

      // Check for message queue messages (fds[0]) before checking the socket(fds[1]),
      // to make sure that we process shutdown messages promptly, even
      // if we would be spinning trying to service the socket.
      else if ((fds[0].revents & POLLIN) != 0)
      {
         // Poll finished because the pipe is ready to read.
         // (One byte in pipe means message available in queue.)
         // Only a SipClient with a derived SipClientWriteBuffer
         // uses the pipe in the Sip message send process

         // Check to see how many messages are in the queue.
         int numberMsgs = (getMessageQueue())->numMsgs();
         Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                       "SipClient[%s]::run got pipe-select  "
                       "Number of Messages waiting: %d",
                       mName.data(),
                       numberMsgs);
         int i;
         char buffer[1];
         for (i = 0; i < numberMsgs; i++)
         {
            // Receive the messages.
            res = receiveMessage((OsMsg*&) pMsg, OsTime::NO_WAIT);
            if (res != OS_SUCCESS)
              break;

            // Normally, this is a SIP message for the write buffer.  Once we have gotten
            // here, we are able to report any initial non-blocking connect error.
            mbTcpOnErrWaitForSend = FALSE;
            tcpOnErrWaitForSend = FALSE;
            Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                          "SipClient[%s]::run got pipe-select  "
                          "mbTcpOnErrWaitForSend-%d waitingToReportErr-%d mbTcpOnErrWaitForSend-%d repeatedEOFs-%d",
                          mName.data(), mbTcpOnErrWaitForSend, waitingToReportErr,
                          mbTcpOnErrWaitForSend, repeatedEOFs);

            // Read 1 byte from the pipe to clear it for this message.  One byte is
            // inserted into the pipe for each message.
            assert(read(mPipeReadingFd, &buffer, 1) == 1);

            if (!handleMessage(*pMsg))            // process the message (from queue)
            {
               OsServerTask::handleMessage(*pMsg);
            }

            if (!pMsg->getSentFromISR())
            {
               pMsg->releaseMsg();                         // free the message
            }

            // In order to report an unframed(eg TCP) socket error to SipUserAgent dispatcher,
            // the error must be carried in a sip message from the client's message queue.
            // The message holds all the identifying information.
            if (waitingToReportErr)
            {
                // Return all buffered messages with a transport error indication.
                emptyBuffer(TRUE);
                clientStopSelf();
            }
         }
      } // end reading msg-available-for-output-queue pipe

      else if ((fds[1].revents & POLLOUT) != 0)
      {
         // Poll finished because socket is ready to write.

         // Call method to continue writing data.
         writeMore();
      }
      else if ((fds[1].revents & POLLIN) != 0)
      {
         // Poll finished because socket is ready to read.

         // Read message.
         // Must allocate a new message because SipUserAgent::dispatch will
         // take ownership of it.

         SipMessage* msg = new SipMessage;
         int res = msg->read(mClientSocket,
                             HTTP_DEFAULT_SOCKET_BUFFER_SIZE,
                             &readBuffer);

         if (msg->ignoreLastRead())
         {
           //
           // Read operation thinks this is NOT a valid SIP/HTTP packet.  Eg. STUN requests
           //
           delete msg;
           readBuffer.remove(0);
           continue;
         }
         // Use readBuffer to hold any unparsed data after the message
         // we read.
         // Note that if a message was successfully parsed, readBuffer
         // still contains as its prefix the characters of that message.
         // We save them for logging purposes below and will delete them later.

         UtlString remoteHostAddress;
         int remoteHostPort;
         msg->getSendAddress(&remoteHostAddress, &remoteHostPort);
         if (!mClientSocket->isSameHost(remoteHostAddress.data(), mLocalHostAddress.data()))
         {
           try
           {
             if (!remoteHostAddress.isNull())
             {
               boost::asio::ip::address remoteIp = boost::asio::ip::address::from_string(remoteHostAddress.data());

               if (rateLimit().isBannedAddress(remoteIp))
               {
                  delete msg;
                  readBuffer.remove(0);
                  continue;
               }

               rateLimit().logPacket(remoteIp, 0);
             }
           }
           catch(const std::exception& e)
           {
             Os::Logger::instance().log(FAC_SIP_INCOMING, PRI_CRIT, 
               "SipClient[%s]::run rate limit exception: %s",  mName.data(), e.what());
           }
         }


         // Note that input was processed at this time.
         touch();

         //
         // Count the CR/LF to see if this is a keep-alive
         //
         int crlfCount = 0;
         for (int i = 0; i < res; i++)
         {
           if (readBuffer(i) == '\r' || readBuffer(i) == '\n')
           {
             crlfCount++;
           } else
           {
             break;
           }
         }

         if (res > 0 && res == crlfCount)
         {
             repeatedEOFs = 0;
             // The 'message' was a keepalive (CR-LF or CR-LF-CR-LF).
             UtlString fromIpAddress;
             int fromPort;
             

             // Get the send address for response.
             msg->getSendAddress(&fromIpAddress, &fromPort);
             if ( !portIsValid(fromPort))
             {
                 fromPort = defaultPort();
             }

            // Log the message at DEBUG level.
            // Only bother processing if the logs are enabled
            if (   mpSipUserAgent->isMessageLoggingEnabled()
                   || Os::Logger::instance().willLog(FAC_SIP_INCOMING, PRI_DEBUG)
               )
            {
               UtlString logMessage;
               logMessage.append("Read keepalive message:\n");
               logMessage.append("----Local Host:");
               logMessage.append(mLocalHostAddress);
               logMessage.append("---- Port: ");
               logMessage.appendNumber(
                  portIsValid(mLocalHostPort) ? mLocalHostPort : defaultPort());
               logMessage.append("----\n");
               logMessage.append("----Remote Host:");
               logMessage.append(fromIpAddress);
               logMessage.append("---- Port: ");
               logMessage.appendNumber(
                  portIsValid(fromPort) ? fromPort : defaultPort());
               logMessage.append("----\n");

               logMessage.append(readBuffer.data(), res);
               UtlString messageString;
               logMessage.append(messageString);
               logMessage.append("====================END====================\n");

               // Don't bother to send the message to the SipUserAgent for its internal log.

               // Write the message to the syslog.
               Os::Logger::instance().log(FAC_SIP_INCOMING, PRI_DEBUG, "%s", logMessage.data());
            }

            //
            // Only send a PONG (CRLF) for PING (CRLF/CRLF)
            // 
            if (crlfCount == SIZE_OF_PING)
            {
              UtlString buffer;
              int bufferLen;

              // send one (PONG) CRLF set in the reply
              buffer.append("\r\n");
              bufferLen = buffer.length();
             
              // send the CR-LF response message
              switch (mSocketType)
              {
              case OsSocket::TCP:
              {
                 Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                               "SipClient[%s]::run send TCP keep-alive CR-LF response, ",
                               mName.data());
                 SipClientSendMsg sendMsg(OsMsg::OS_EVENT,
                                          SipClientSendMsg::SIP_CLIENT_SEND_KEEP_ALIVE,
                                          fromIpAddress,
                                          fromPort);
                  handleMessage(sendMsg);     // add newly created keep-alive to write buffer
              }
                 break;
              case OsSocket::UDP:
              {
                  Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                                "SipClient[%s]::run send UDP keep-alive CR-LF response, ",
                                mName.data());
                 (dynamic_cast <OsDatagramSocket*> (mClientSocket))->write(buffer.data(),
                                                                           bufferLen,
                                                                           fromIpAddress,
                                                                           fromPort);
              }
                 break;
              default:
                 break;
              }
            }

            // Delete the SipMessage allocated above, which is no longer needed.
            delete msg;

            // Now that logging is done, remove the parsed bytes and
            // remember any unparsed input for later use.
            readBuffer.remove(0, res);
         }  // end keep-alive msg

         else if (res > 0)      // got message, but not keep-alive
         {
            // Message successfully read.
            repeatedEOFs = 0;

            // Do preliminary processing of message to log it,
            // clean up its data, and extract any needed source address.
            if (preprocessMessage(*msg, readBuffer, res))
            {
              // Dispatch the message.
              // dispatch() takes ownership of *msg.
              mpSipUserAgent->dispatch(msg);
            }
            else
            {
              //
              // Drop this message silently but log it on debug
              //
              OS_LOG_DEBUG( FAC_SIP, "SipClient::preprocessMessage -  Dropping " << res << " bytes of malformed packet." );
              delete msg;
            }

            // Now that logging is done, remove the parsed bytes and
            // remember any unparsed input for later use.
            readBuffer.remove(0, res);
         }  // end process read of >0 bytes
         else
         {
            // Something went wrong while reading the message.
            // (Possibly EOF on a connection-oriented socket.)
            repeatedEOFs++;

            // Delete the SipMessage allocated above, which is no longer needed.
            delete msg;
            Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                          "SipClient[%s]::run SipMessage::read returns %d (error(%d) or EOF), "
                          "readBuffer = '%.1000s'",
                          mName.data(), res, errno, readBuffer.data());

            Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                          "SipClient[%s]::run error wait status  "
                          "tcpOnErrWaitForSend-%d waitingToReportErr-%d "
                          "mbTcpOnErrWaitForSend-%d repeatedEOFs-%d "
                          "protocol %d framed %d",
                          mName.data(),
                          tcpOnErrWaitForSend, waitingToReportErr,
                          mbTcpOnErrWaitForSend, repeatedEOFs,
                          mClientSocket->getIpProtocol(),
                          OsSocket::isFramed(mClientSocket->getIpProtocol()));

            // If the socket is not framed (is connection-oriented),
            // we need to abort the connection and post a message
            // :TODO: This doesn't work right for framed connection-oriented
            // protocols (like SCTP), but OsSocket doesn't have an EOF-query
            // method -- we need to close all connection-oriented
            // sockets as well in case it was an EOF.
            // Define a virtual function that returns the correct bit.
            if (!OsSocket::isFramed(mClientSocket->getIpProtocol()))
            {
                // On non-blocking connect failures, we need to get the first send message
                // in order to successfully trigger the protocol fallback mechanism
                if (!tcpOnErrWaitForSend)
                {
                   // Return all buffered messages with a transport error indication.
                   emptyBuffer(TRUE);
                   clientStopSelf();
                }
                else
                {
                   fds[1].revents &= ~(POLLERR | POLLHUP);  // clear error bit if waiting
                   waitingToReportErr = TRUE;
                }
            }
            // Delete the data read so far, which will not have been
            // deleted by HttpMessage::read.
            readBuffer.remove(0);
         }
      } // end POLLIN reading socket
   }
   while (isStarted());

   return 0;        // and then exit
}

static std::vector<std::string> string_tokenize(const std::string& str, const char* tok)
{
  std::vector<std::string> tokens;
  boost::split(tokens, str, boost::is_any_of(tok), boost::token_compress_on);
  return tokens;
}

static bool convert_tel_to_sip_uri(const std::string& telUri, const std::string& domain, std::string& sipUri)
{
  sipUri.reserve(telUri.size() + domain.size() + 11);
  sipUri = "sip:";
  std::string scheme;
  scheme.reserve(5);
  bool foundScheme = false;
  bool foundParams = false;
  for (std::string::const_iterator iter = telUri.begin(); iter != telUri.end(); iter++)
  {
    if (!foundScheme)
    {
      if (*iter == ':')
      {
        foundScheme = true;
        if (scheme != "tel")
        {
          return false;
        }
      }
      else
      {
        scheme.push_back(*iter);
      }
    }
    else if (!foundParams)
    {
      if (*iter == ';')
      {
        foundParams = true;
        sipUri += "@";
        sipUri += domain;
        sipUri += ";user=phone;";
      }
      else
      {
        sipUri.push_back(*iter);
      }
    }
    else
    {
      sipUri.push_back(*iter);
    }
  }
  
  if (!foundParams)
  {
    sipUri += "@";
    sipUri += domain;
    sipUri += ";user=phone";
  }
  
  return true;
}

static bool convert_start_line_tel_uri(const std::string& startLine, const std::string& domain, std::string& requestLine)
{
  //
  // Tokenize the start line 
  //
  std::vector<std::string> startLineTokens = string_tokenize(startLine, " ");
  if (startLineTokens.size() != 3)
  {
    OS_LOG_WARNING(FAC_SIP, "SipClient::convert_start_line_tel_uri - unable to get valid token count from " << startLine);
    return false;
  }
  
  std::string& method = startLineTokens[0];
  std::string& telUri = startLineTokens[1];
  std::string& version = startLineTokens[2];
  std::string sipUri;
  
  convert_tel_to_sip_uri(telUri, domain, sipUri);
  
    
  std::ostringstream strm;
  strm << method << " " << sipUri << " " << version;
  requestLine = strm.str();
  
  return true;
}

bool SipClient::preprocessRequestLine(SipMessage& msg)
{
  if (msg.isResponse())
    return true;

  //
  // Check if we have a telephone URI and convert it to sip uri format
  //
  std::string firstHeaderLine = msg.getFirstHeaderLine();
  if (firstHeaderLine.find("tel:") != std::string::npos)
  {
    if (mpSipUserAgent->getDomain().empty())
    {
      //
      // If domain is not set, we will drop the request because we do not support the
      // tel-uri format.  This has been reported to crash the proxy
      //
      OS_LOG_WARNING(FAC_SIP, "SipClient::preprocessRequestLine - Unable to process " <<  firstHeaderLine << " because SipUserAgent::getDomain() is not set.");
      return false;
    }

    std::string requestLine;
    if (!convert_start_line_tel_uri(firstHeaderLine, mpSipUserAgent->getDomain(), requestLine))
    {
      OS_LOG_WARNING(FAC_SIP, "SipClient::preprocessRequestLine - Unable to convert " <<  firstHeaderLine << " to a SIP URI.");
      return false;
    }

    OS_LOG_INFO(FAC_SIP, "SipClient::preprocessRequestLine - Rewrote tel request-line: (was) " <<  firstHeaderLine << " (now) " << requestLine);

    msg.setFirstHeaderLine(requestLine.c_str());
  }

  return true;
}

static bool convert_uri_header_tel_uri(SipMessage& msg, const std::string& headerName, const std::string& header, const std::string& domain, std::string& newHeader)
{
  //
  // Check if we have enclosing brackets
  //
  bool hasBrackets = header.find("<tel:") != std::string::npos;
  std::string displayName;
  std::string headerParams;
  std::string telUri;
  std::string sipUri;
  
  if (!hasBrackets)
  {
    telUri = header;
  }
  else
  {
    displayName.reserve(header.size());
    telUri.reserve(header.size());
    headerParams.reserve(header.size());
    
    bool foundLT = false;
    bool foundGT = false;
    for (std::string::const_iterator iter = header.begin(); iter != header.end(); iter++)
    {
      if (!foundLT)
      {
        if (*iter == '<')
        {
          foundLT = true;
        }
        else
        {
          displayName.push_back(*iter);
        }
      }
      else if (!foundGT)
      {
        if (*iter == '>')
        {
          foundGT = true;
        }
        else
        {
          telUri.push_back(*iter);
        }
      }
      else
      {
        headerParams.push_back(*iter);
      }
    }
  }
  
  if (!convert_tel_to_sip_uri(telUri, domain, sipUri))
    return false;
  
  std::ostringstream prop;
  prop << headerName << "-TEL-URI";
  msg.setProperty(prop.str(), telUri);
  
  std::ostringstream strm;
  if (!displayName.empty())
  {
    strm << displayName;
  }
  
  strm << "<" << sipUri << ">" << headerParams;
  newHeader = strm.str();
  return true;
}

bool SipClient::preprocessUriHeader(SipMessage& msg, const char* headerName)
{
  const char* hdr = msg.getHeaderValue(0, headerName);
  if (!hdr)
  {
    return false;
  }
 
  std::string header = hdr;
  if (header.find("tel:") != std::string::npos)
  {
    if (mpSipUserAgent->getDomain().empty())
    {
      //
      // If domain is not set, we will drop the request because we do not support the
      // tel-uri format.  This has been reported to crash the proxy
      //
      OS_LOG_WARNING(FAC_SIP, "SipClient::preprocessUriHeader - Unable to process " <<  header << " because SipUserAgent::getDomain() is not set.");
      return false;
    }

    std::string newHeader;
    if (!convert_uri_header_tel_uri(msg, headerName, header, mpSipUserAgent->getDomain(), newHeader))
    {
      OS_LOG_WARNING(FAC_SIP, "SipClient::preprocessUriHeader - Unable to convert " <<  header << " to a SIP URI.");
      return false;
    }

    OS_LOG_INFO(FAC_SIP, "SipClient::preprocessUriHeader - Rewrote tel-uri " << headerName << " : (was) " <<  header << " (now) " << newHeader);

    msg.setHeaderValue(headerName, newHeader.c_str(), 0);
  }
  
  return true;
}

// Do preliminary processing of message to log it,
// clean up its data, and extract any needed source address.
bool SipClient::preprocessMessage(SipMessage& msg,
                                  const UtlString& msgText,
                                  int msgLength)
{
  
  msg.setProperty("transport-queue-size", boost::lexical_cast<std::string>(mIncomingQ.numMsgs()));
  msg.setProperty("transport-queue-max-size", boost::lexical_cast<std::string>(mIncomingQ.maxMsgs()));

   // Canonicalize short field names.
   msg.replaceShortFieldNames();

   //
   // Make sure that the message is valid.  We will check for presence of Call-Id, CSeq, From, To and Via Fields
   //
   if (
     !msg.getHeaderValue(0, SIP_CSEQ_FIELD) ||
     !msg.getHeaderValue(0, SIP_CALLID_FIELD) ||
     !msg.getHeaderValue(0, SIP_FROM_FIELD) ||
     !msg.getHeaderValue(0, SIP_TO_FIELD) ||
     !msg.getHeaderValue(0, SIP_VIA_FIELD))
   {
     return false;
   }
   
   if (!preprocessRequestLine(msg))
   {
     return false;
   }
   
   if (!preprocessUriHeader(msg, SIP_FROM_FIELD))
   {
     return false;
   }
   
   if (!preprocessUriHeader(msg, SIP_TO_FIELD))
   {
     return false;
   }
   
   // Get the send address.
   UtlString fromIpAddress;
   int fromPort;
   msg.getSendAddress(&fromIpAddress, &fromPort);

   // Log the message.
   // Only bother processing if the logs are enabled
   if (   mpSipUserAgent->isMessageLoggingEnabled()
          || Os::Logger::instance().willLog(FAC_SIP_INCOMING, PRI_INFO)
      )
   {
      UtlString logMessage;
      logMessage.append("Read SIP message:\n");
      logMessage.append("----Local Host:");
      logMessage.append(mLocalHostAddress);
      logMessage.append("---- Port: ");
      logMessage.appendNumber(
                 portIsValid(mLocalHostPort) ? mLocalHostPort : defaultPort());
      logMessage.append("----\n");
      logMessage.append("----Remote Host:");
      logMessage.append(fromIpAddress);
      logMessage.append("---- Port: ");
      logMessage.appendNumber(
                 portIsValid(fromPort) ? fromPort : defaultPort());
      logMessage.append("----\n");

      logMessage.append(msgText.data(), msgLength);
      UtlString messageString;
      logMessage.append(messageString);
      logMessage.append("====================END====================\n");

      // Send the message to the SipUserAgent for its internal log.
      mpSipUserAgent->logMessage(logMessage.data(), logMessage.length());
      // Write the message to the syslog.
      Os::Logger::instance().log(FAC_SIP_INCOMING, PRI_INFO, "%s", logMessage.data());
   }

   // Set the date field if not present
   long epochDate;
   if (!msg.getDateField(&epochDate))
   {
      msg.setDateField();
   }

   // Set the protocol and time.
   msg.setSendProtocol(mSocketType);
   msg.setTransportTime(touchedTime);

   // Keep track of where this message came from
   msg.setSendAddress(fromIpAddress.data(), fromPort);

   // Keep track of the interface on which this message was
   // received.
   msg.setInterfaceIpPort(mClientSocket->getLocalIp(), mClientSocket->getLocalHostPort());

   if (mReceivedAddress.isNull())
   {
      mReceivedAddress = fromIpAddress;
      mRemoteReceivedPort = fromPort;
   }

   // If this is a request...
   if (!msg.isResponse())
   {
      UtlString lastAddress;
      int lastPort;
      UtlString lastProtocol;
      int receivedPort;
      UtlBoolean receivedSet;
      UtlBoolean maddrSet;
      UtlBoolean receivedPortSet;

      // Fill in 'received' and 'rport' in top Via if needed.
      msg.setReceivedViaParams(fromIpAddress, fromPort);

      // Derive information about the other end of the connection
      // from the Via that the other end provided.

      // Get the addresses from the topmost Via.
      msg.getTopVia(&lastAddress, &lastPort, &lastProtocol,
                    &receivedPort, &receivedSet, &maddrSet,
                    &receivedPortSet);

      // :QUERY: Should this test be mClientSocket->isConnected()?
      if (   (   mSocketType == OsSocket::TCP
                 || mSocketType == OsSocket::SSL_SOCKET
                )
             && !receivedPortSet
         )
      {
         // we can use this socket as if it were
         // connected to the port specified in the
         // via field
         mRemoteReceivedPort = lastPort;
      }

      // Keep track of the address the other
      // side said they sent from.  Note, this cannot
      // be trusted unless this transaction is
      // authenticated
      if (mRemoteViaAddress.isNull())
      {
         mRemoteViaAddress = lastAddress;
         mRemoteViaPort =
            portIsValid(lastPort) ? lastPort : defaultPort();
      }
   }

   //
   // Call all sip input processors
   //
   system_tap_sip_rx(fromIpAddress.data(), portIsValid(fromPort) ? fromPort : defaultPort(),
           mLocalHostAddress.data(), portIsValid(mLocalHostPort) ? mLocalHostPort : defaultPort(),
           msgText.data(), msgLength);

   mpSipUserAgent->executeAllSipInputProcessors(msg, fromIpAddress.data(), portIsValid(fromPort) ? fromPort : defaultPort());

   return true;
}

// Test whether the socket is ready to read. (Does not block.)
UtlBoolean SipClient::isReadyToRead()
{
   return mClientSocket->isReadyToRead(0);
}

// Wait until the socket is ready to read (or has an error).
UtlBoolean SipClient::waitForReadyToRead()
{
   return mClientSocket->isReadyToRead(-1);
}

// Called by the thread to shut the SipClient down and signal its
// owning server that it has done so.
void SipClient::clientStopSelf()
{
   Os::Logger::instance().log(FAC_SIP, PRI_DEBUG,
                 "SipClient[%s]::clientStopSelf called",
                 mName.data());

   // Stop the run loop.
   OsTask::requestShutdown();
}

/* //////////////////////////// PRIVATE /////////////////////////////////// */

/* ============================ FUNCTIONS ================================= */
