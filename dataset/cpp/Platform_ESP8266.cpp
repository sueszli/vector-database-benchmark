/*
 * Platform_ESP8266.cpp
 * Copyright (C) 2020-2023 Linar Yusupov
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#if defined(ESP8266)

#include "SoCHelper.h"
#include "EEPROMHelper.h"
#include "WiFiHelper.h"

ESP8266WebServer server ( 80 );

static void ESP8266_setup()
{
  hw_info.model = SOFTRF_MODEL_WEBTOP_SERIAL;
  hw_info.revision = HW_REV_DEVKIT;

  Serial.begin(SERIAL_OUT_BR, SERIAL_OUT_BITS);
}

static void ESP8266_post_init()
{

}

static void ESP8266_loop()
{

}

static void ESP8266_fini()
{

}

static void ESP8266_reset()
{
  ESP.restart();
}

static void ESP8266_sleep_ms(int ms)
{
  /* TODO */
}

static uint32_t ESP8266_getChipId()
{
  return ESP.getChipId();
}

static uint32_t ESP8266_getFreeHeap()
{
  return ESP.getFreeHeap();
}

static bool ESP8266_EEPROM_begin(size_t size)
{
  EEPROM.begin(size);
  return true;
}

static void ESP8266_WiFi_set_param(int ndx, int value)
{
  switch (ndx)
  {
  case WIFI_PARAM_TX_POWER:
    WiFi.setOutputPower(value);
    break;
  case WIFI_PARAM_DHCP_LEASE_TIME:
    if (WiFi.getMode() == WIFI_AP) {
      wifi_softap_set_dhcps_lease_time((uint32) (value * 60)); /* in minutes */
    }
    break;
  default:
    break;
  }
}

static bool ESP8266_WiFi_hostname(String aHostname)
{
  return WiFi.hostname(aHostname);
}

static void ESP8266_WiFiUDP_stopAll()
{
  WiFiUDP::stopAll();
}

static IPAddress ESP8266_WiFi_get_broadcast()
{
  struct ip_info ipinfo;
  IPAddress broadcastIp;

  if (WiFi.getMode() == WIFI_STA) {
    wifi_get_ip_info(STATION_IF, &ipinfo);
  } else {
    wifi_get_ip_info(SOFTAP_IF, &ipinfo);
  }
  broadcastIp = ~ipinfo.netmask.addr | ipinfo.ip.addr;

  return broadcastIp;
}

static void ESP8266_WiFi_transmit_UDP(int port, byte *buf, size_t size)
{
  IPAddress ClientIP;
  struct station_info *stat_info;
  WiFiMode_t mode = WiFi.getMode();

  switch (mode)
  {
  case WIFI_STA:
    ClientIP = ESP8266_WiFi_get_broadcast();

    SoC->swSer_enableRx(false);

    Uni_Udp.beginPacket(ClientIP, port);
    Uni_Udp.write(buf, size);
    Uni_Udp.endPacket();

    SoC->swSer_enableRx(true);

    break;
  case WIFI_AP:
    stat_info = wifi_softap_get_station_info();

    while (stat_info != NULL) {
      ClientIP = stat_info->ip.addr;

      SoC->swSer_enableRx(false);

      Uni_Udp.beginPacket(ClientIP, port);
      Uni_Udp.write(buf, size);
      Uni_Udp.endPacket();

      SoC->swSer_enableRx(true);

      stat_info = STAILQ_NEXT(stat_info, next);
    }
    wifi_softap_free_station_info();
    break;
  case WIFI_OFF:
  default:
    break;
  }
}

static size_t ESP8266_WiFi_Receive_UDP(uint8_t *buf, size_t max_size)
{
  return 0; // WiFi_Receive_UDP(buf, max_size);
}

static int ESP8266_WiFi_clients_count()
{
  struct station_info *stat_info;
  int clients = 0;
  WiFiMode_t mode = WiFi.getMode();

  switch (mode)
  {
  case WIFI_AP:
    stat_info = wifi_softap_get_station_info();

    while (stat_info != NULL) {
      clients++;

      stat_info = STAILQ_NEXT(stat_info, next);
    }
    wifi_softap_free_station_info();

    return clients;
  case WIFI_STA:
  default:
    return -1; /* error */
  }
}

static void ESP8266_swSer_begin(unsigned long baud)
{
  SerialInput.begin(baud, SERIAL_8N1);

  if (settings->m.connection == CON_SERIAL_AUX) {
    SerialInput.swap();
  }
}

static void ESP8266_swSer_enableRx(boolean arg)
{

}

static uint32_t ESP8266_maxSketchSpace()
{
  return (ESP.getFreeSketchSpace() - 0x1000) & 0xFFFFF000;
}

static void ESP8266_Battery_setup()
{

}

static float ESP8266_Battery_voltage()
{
  return analogRead (SOC_GPIO_PIN_BATTERY) / SOC_A0_VOLTAGE_DIVIDER ;
}

static bool ESP8266_DB_init()
{
  return false;
}

static bool ESP8266_DB_query(uint8_t type, uint32_t id, char *buf, size_t size)
{
  return false;
}

static void ESP8266_DB_fini()
{

}

static void ESP8266_TTS(char *message)
{

}

static void ESP8266_Button_setup()
{

}

static void ESP8266_Button_loop()
{

}

static void ESP8266_Button_fini()
{

}

static bool ESP8266_Baro_setup()
{
  return false;
}

static void ESP8266_WDT_setup()
{

}

static void ESP8266_WDT_fini()
{

}

static void ESP8266_Service_Mode(boolean arg)
{

}

const SoC_ops_t ESP8266_ops = {
  SOC_ESP8266,
  "ESP8266",
  ESP8266_setup,
  ESP8266_post_init,
  ESP8266_loop,
  ESP8266_fini,
  ESP8266_reset,
  ESP8266_sleep_ms,
  ESP8266_getChipId,
  ESP8266_getFreeHeap,
  ESP8266_EEPROM_begin,
  ESP8266_WiFi_set_param,
  ESP8266_WiFi_hostname,
  ESP8266_WiFiUDP_stopAll,
  ESP8266_WiFi_transmit_UDP,
  ESP8266_WiFi_Receive_UDP,
  ESP8266_WiFi_clients_count,
  ESP8266_swSer_begin,
  ESP8266_swSer_enableRx,
  ESP8266_maxSketchSpace,
  ESP8266_Battery_setup,
  ESP8266_Battery_voltage,
  ESP8266_DB_init,
  ESP8266_DB_query,
  ESP8266_DB_fini,
  ESP8266_TTS,
  ESP8266_Button_setup,
  ESP8266_Button_loop,
  ESP8266_Button_fini,
  ESP8266_Baro_setup,
  ESP8266_WDT_setup,
  ESP8266_WDT_fini,
  ESP8266_Service_Mode,
  NULL,
  NULL
};

#endif /* ESP8266 */
