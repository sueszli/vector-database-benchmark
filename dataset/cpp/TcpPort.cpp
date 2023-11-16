/*! @file TCPPort.cpp
    @brief Implementation of UdpPort class.

    @author Aaron Wong, Jason Kulk
 
 Copyright (c) 2009 Aaron Wong, Jason Kulk
 
    This file is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with NUbot.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "TcpPort.h"
#include "Infrastructure/NUImage/NUImage.h"
#include "Infrastructure/NUSensorsData/NUSensorsData.h"
#include "debug.h"
#include "debugverbositynetwork.h"
#include <string.h>
#include <errno.h>
#if defined(USE_LOCALISATION)
    #include "Localisation/Localisation.h"
#endif
#include "Infrastructure/FieldObjects/FieldObjects.h"

/*! @brief Constructs a tcp port on the specified port
 

    @param portnumber the port number the data will be sent and received on
 */
TcpPort::TcpPort(int portnumber): Thread(string("Tcp Thread"), 0)
{
#ifdef WIN32
    WSADATA wsa_Data;
    int wsa_ReturnCode = WSAStartup(0x101,&wsa_Data);
    if (wsa_ReturnCode != 0)
    {
        debug <<  "WSA ERROR CODE: "<< wsa_ReturnCode << endl;
    }
#endif
#if DEBUG_NETWORK_VERBOSITY > 4
    debug << "TcpPort::TcpPort(" << portnumber << ")" << endl;
#endif
    m_port_number = portnumber;
    if ((m_sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) 
        errorlog << "TcpPort::TcpPort(" << m_port_number << "). Failed to create socket file descriptor." << endl;

    // Set the reuse address flag
    #ifdef WIN32
        char reuseflag = 1;
    #else
        int reuseflag = 1;
    #endif
    if (setsockopt(m_sockfd, SOL_SOCKET, SO_REUSEADDR, &reuseflag, sizeof(reuseflag)) == -1)
        errorlog << "TcpPort::TcpPort(). Failed to set reuseaddr socket options, errno: " << errno << endl;
        
    m_address.sin_family = AF_INET;                             // host byte order
    m_address.sin_port = htons(m_port_number);                  // short, network byte order
    m_address.sin_addr.s_addr = htonl(INADDR_ANY);                     // automatically fill with my IP
    memset(m_address.sin_zero, '\0', sizeof m_address.sin_zero);
    
    m_broadcast_address.sin_family = AF_INET;                   // host byte order
    m_broadcast_address.sin_port = htons(m_port_number);        // short, network byte order 
    m_broadcast_address.sin_addr.s_addr = htonl(INADDR_BROADCAST);     // automatically fill with my local Broadcast IP
    memset(m_broadcast_address.sin_zero, '\0', sizeof m_broadcast_address.sin_zero);
    
#if DEBUG_NETWORK_VERBOSITY > 4
    debug << "TcpPort::TcpPort(). Binding socket." << endl;
#endif
    if (bind(m_sockfd, (struct sockaddr *)&m_address, sizeof m_address) == -1)
        errorlog << "TcpPort::TcpPort(" << portnumber << "). Failed to bind socket." << endl;
    
    m_time_last_receive = 0;           //!< @todo TODO: change the initial value for this to something -3000ms!
    
    m_message_size = 0;
    m_has_data = false;
    m_clientSockfd = -1;
    pthread_mutex_init(&m_socket_mutex, NULL);
    
    start();
}

/*! @brief Closes the udp port
 */
TcpPort::~TcpPort()
{
#ifdef WIN32
    closesocket(m_sockfd);
    WSACleanup();
#endif
#ifndef WIN32
    close(m_sockfd);
#endif
    pthread_mutex_destroy(&m_socket_mutex);
}

/*! @brief Run the TCP port's main loop
 
    Waits until data is received on the appropriate port.
    Then copies it to m_data, setting m_message_size, m_has_data and m_time_last_receive in the process
 */
void TcpPort::run()
{
#if DEBUG_NETWORK_VERBOSITY > 4
    debug << "TcpPort::run(). Starting udpport:" << m_port_number << "'s mainloop" << endl;
#endif
    struct sockaddr_in local_their_addr; // connector's address information
    socklen_t local_addr_len = sizeof(local_their_addr);
    char localdata[10*1024];
    int localnumBytes = 0;
    listen(m_sockfd,5);     //Start Listening for Clients

    while(1)
    {
        m_clientSockfd = accept(m_sockfd, (struct sockaddr *)&local_their_addr, &local_addr_len);
        #ifdef WIN32
            localnumBytes = recv(m_clientSockfd, localdata, sizeof(localdata),0);
        #endif
        #ifndef WIN32
            localnumBytes = read(m_clientSockfd, localdata,sizeof(localdata));
        #endif
        if ( localnumBytes != -1 && local_their_addr.sin_addr.s_addr != m_address.sin_addr.s_addr && local_their_addr.sin_addr.s_addr != m_broadcast_address.sin_addr.s_addr)
        {   //!< @todo TODO: This doesn't work. You need to discard packets that you have sent yourself
            #if DEBUG_NETWORK_VERBOSITY > 3
                debug << "TcpPort::run()." << m_port_number <<" Received " << localnumBytes << " bytes from " << inet_ntoa(local_their_addr.sin_addr) << endl;
            #endif
            #if DEBUG_NETWORK_VERBOSITY > 4
                debug << "TcpPort::run(). Received ";
                for (int i=0; i<localnumBytes; i++)
                    debug << localdata[i];
                debug << endl;
            #endif

            pthread_mutex_lock(&m_socket_mutex);
            memcpy(m_data, localdata, 10*1024);
            m_message_size = localnumBytes; 
            m_has_data = true;
            pthread_mutex_unlock(&m_socket_mutex);
        }
        
    }
    return;
}

/*! @brief Wraps the received data (m_data) into a network_data_t and returns it
        
    @return the data received is returned. If there is not data then the returned object will have a size of -1
 */
network_data_t TcpPort::receiveData()
{
    network_data_t netdata;
    netdata.size = -1;
    netdata.data = NULL; 
    
    pthread_mutex_lock(&m_socket_mutex);
    if (m_has_data == true)
    {
        netdata.size = m_message_size;
        netdata.data = new char[m_message_size];
        memcpy(netdata.data, m_data, m_message_size);
        m_has_data = false;
    }
    pthread_mutex_unlock(&m_socket_mutex);
    #if DEBUG_NETWORK_VERBOSITY > 4
    debug << "TCP Recieved: " << netdata.size;
    #endif
    return netdata;
}

/*! @brief Sends the network data (netdata) to all local hosts provided I think someone is listening
 */
void TcpPort::sendData(network_data_t netdata)
{
    pthread_mutex_lock(&m_socket_mutex);
    if (m_clientSockfd == -1)
    {   
        #if DEBUG_NETWORK_VERBOSITY > 4
        debug << "TcpPort::sendData(). No connected client "<< endl;
        #endif
        return;
    }
    #if DEBUG_NETWORK_VERBOSITY > 4
        debug << "TcpPort::sendData(). Sending " << netdata.size << " bytes to Requested"  << endl;

        //debug << "DATA 1st 4 bytes: "<< (int)netdata.data[0] << ","<<(int)netdata.data[1] << "," << (int)netdata.data[2] << "," << (int)netdata.data[3];
    #endif
    int localnumBytes(0);
    #ifdef WIN32
        localnumBytes = send(m_clientSockfd, netdata.data, netdata.size,0);
    #else
        localnumBytes = write(m_clientSockfd, netdata.data, netdata.size);
    #endif
    #if DEBUG_NETWORK_VERBOSITY > 4
        if(localnumBytes < 0)
            debug << "TcpPort::sendData(). Sending Error "<< endl;
    #endif
    pthread_mutex_unlock(&m_socket_mutex);
    return;
}

void TcpPort::sendData(const NUImage& p_image, const NUSensorsData &p_sensors)
{
    network_data_t netdata;
    stringstream buffer;
    network_data_t sensordata;
    stringstream sensorsbuffer;
    sensorsbuffer << p_sensors;
    string sensorsString = sensorsbuffer.str();
    sensordata.data = (char*) sensorsString.c_str();
    sensordata.size = sensorsString.size();
    
    int sensorsSize = sensordata.size;
    NUImage::Header image_header = NUImage::currentVersionHeader();
    int imagewidth = p_image.getWidth();
    int imageheight = p_image.getHeight();
    double timeStamp = p_image.GetTimestamp();
    bool flipped = p_image.flipped;
    buffer.write(reinterpret_cast<char*>(&sensorsSize), sizeof(sensorsSize));
    buffer.write(reinterpret_cast<char*>(&image_header), sizeof(image_header));
    buffer.write(reinterpret_cast<char*>(&imagewidth), sizeof(imagewidth));
    buffer.write(reinterpret_cast<char*>(&imageheight), sizeof(imageheight));
    buffer.write(reinterpret_cast<char*>(&timeStamp), sizeof(timeStamp));
    buffer.write(reinterpret_cast<char*>(&flipped), sizeof(flipped));
    
    string s = buffer.str();
    netdata.data = (char*) s.c_str();
    netdata.size = s.size();
    
    sendData(netdata);
    
    for(int y = 0; y < imageheight; y++)
    {
        const Pixel& line_start = p_image.at(0, y);    // We want to transmit the raw data, so use the at() access function or lines may get out of order.
        network_data_t linedata;
        linedata.data = (char*) &line_start;
        linedata.size = sizeof(line_start)*imagewidth;
        sendData(linedata);
    }
    sendData(sensordata);
    
}

#if defined(USE_LOCALISATION)
    void TcpPort::sendData(const Localisation& p_locwm, const FieldObjects& p_objects)
    {
        #if DEBUG_NETWORK_VERBOSITY > 4
            debug << "Sending worldmodel packet" << endl;
        #endif
        network_data_t netdata;
        network_data_t sizedata;
        stringstream buffer;
        buffer << p_locwm;
        buffer << p_objects;
        netdata.data = (char*) buffer.str().c_str();
        netdata.size = buffer.str().size();

        int totalsize = netdata.size;
        sizedata.data = reinterpret_cast<char*>(&totalsize);
        sizedata.size = sizeof(totalsize);

        sendData(sizedata);
        sendData(netdata);
    }
#endif
