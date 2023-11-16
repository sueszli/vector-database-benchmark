/*
 * BluetoothHelper.cpp
 * Copyright (C) 2018-2023 Linar Yusupov
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

#if defined(ESP32)
#include "sdkconfig.h"
#endif

#if defined(ESP32) && !defined(CONFIG_IDF_TARGET_ESP32S2)

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled!
#endif

/*
    BLE code is based on Neil Kolban example for IDF:
      https://github.com/nkolban/esp32-snippets/blob/master/cpp_utils/tests/BLE%20Tests/SampleNotify.cpp
    Ported to Arduino ESP32 by Evandro Copercini    
    HM-10 emulation and adaptation for SoftRF is done by Linar Yusupov.
*/
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#include "esp_gap_bt_api.h"

#include "../system/SoC.h"
#include "EEPROM.h"
#include "Bluetooth.h"
#include "WiFi.h"
#include "Battery.h"

#include <core_version.h>

BLEServer* pServer = NULL;
BLECharacteristic* pUARTCharacteristic = NULL;
BLECharacteristic* pBATCharacteristic  = NULL;

BLECharacteristic* pModelCharacteristic         = NULL;
BLECharacteristic* pSerialCharacteristic        = NULL;
BLECharacteristic* pFirmwareCharacteristic      = NULL;
BLECharacteristic* pHardwareCharacteristic      = NULL;
BLECharacteristic* pSoftwareCharacteristic      = NULL;
BLECharacteristic* pManufacturerCharacteristic  = NULL;

bool deviceConnected    = false;
bool oldDeviceConnected = false;

#if defined(USE_BLE_MIDI)
BLECharacteristic* pMIDICharacteristic = NULL;
#endif /* USE_BLE_MIDI */

cbuf *BLE_FIFO_RX, *BLE_FIFO_TX;

#if defined(CONFIG_IDF_TARGET_ESP32)
#include <BluetoothSerial.h>
BluetoothSerial SerialBT;
#endif /* CONFIG_IDF_TARGET_ESP32 */

String BT_name = HOSTNAME;

static unsigned long BLE_Notify_TimeMarker = 0;
static unsigned long BLE_Advertising_TimeMarker = 0;

BLEDescriptor UserDescriptor(BLEUUID((uint16_t)0x2901));

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      BLE_Advertising_TimeMarker = millis();
    }
};

class UARTCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pUARTCharacteristic) {
#if defined(ESP_IDF_VERSION_MAJOR) && ESP_IDF_VERSION_MAJOR>=5
      String rxValue = pUARTCharacteristic->getValue();
#else
      std::string rxValue = pUARTCharacteristic->getValue();
#endif /* ESP_IDF_VERSION_MAJOR */

      if (rxValue.length() > 0) {
        BLE_FIFO_RX->write(rxValue.c_str(),
                      (BLE_FIFO_RX->room() > rxValue.length() ?
                      rxValue.length() : BLE_FIFO_RX->room()));
      }
    }
};

static void ESP32_Bluetooth_setup()
{
  BT_name += "-";
  BT_name += String(SoC->getChipId() & 0x00FFFFFFU, HEX);

  switch (settings->bluetooth)
  {
#if defined(CONFIG_IDF_TARGET_ESP32)
  case BLUETOOTH_SPP:
    {
      esp_bt_controller_mem_release(ESP_BT_MODE_BLE);

      SerialBT.begin(BT_name.c_str());
    }
    break;
#endif /* CONFIG_IDF_TARGET_ESP32 */
  case BLUETOOTH_LE_HM10_SERIAL:
    {
      BLE_FIFO_RX = new cbuf(BLE_FIFO_RX_SIZE);
      BLE_FIFO_TX = new cbuf(BLE_FIFO_TX_SIZE);

#if defined(CONFIG_IDF_TARGET_ESP32)
      esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT);
#endif /* CONFIG_IDF_TARGET_ESP32 */

      // Create the BLE Device
      BLEDevice::init((BT_name+"-LE").c_str());

      /*
       * Set the MTU of the packets sent,
       * maximum is 500, Apple needs 23 apparently.
       */
      // BLEDevice::setMTU(23);

      // Create the BLE Server
      pServer = BLEDevice::createServer();
      pServer->setCallbacks(new MyServerCallbacks());

      // Create the BLE Service
      BLEService *pService = pServer->createService(BLEUUID(UART_SERVICE_UUID16));

      // Create a BLE Characteristic
      pUARTCharacteristic = pService->createCharacteristic(
                              BLEUUID(UART_CHARACTERISTIC_UUID16),
                              BLECharacteristic::PROPERTY_READ   |
                              BLECharacteristic::PROPERTY_NOTIFY |
                              BLECharacteristic::PROPERTY_WRITE_NR
                            );

      UserDescriptor.setValue("HMSoft");
      pUARTCharacteristic->addDescriptor(&UserDescriptor);
      pUARTCharacteristic->addDescriptor(new BLE2902());

      pUARTCharacteristic->setCallbacks(new UARTCallbacks());

      // Start the service
      pService->start();

      // Create the BLE Service
      pService = pServer->createService(BLEUUID(UUID16_SVC_BATTERY));

      // Create a BLE Characteristic
      pBATCharacteristic = pService->createCharacteristic(
                              BLEUUID(UUID16_CHR_BATTERY_LEVEL),
                              BLECharacteristic::PROPERTY_READ   |
                              BLECharacteristic::PROPERTY_NOTIFY
                            );
      pBATCharacteristic->addDescriptor(new BLE2902());

      // Start the service
      pService->start();

      // Create the BLE Service
      pService = pServer->createService(BLEUUID(UUID16_SVC_DEVICE_INFORMATION));

      // Create BLE Characteristics
      pModelCharacteristic = pService->createCharacteristic(
                              BLEUUID(UUID16_CHR_MODEL_NUMBER_STRING),
                              BLECharacteristic::PROPERTY_READ
                            );
      pSerialCharacteristic = pService->createCharacteristic(
                              BLEUUID(UUID16_CHR_SERIAL_NUMBER_STRING),
                              BLECharacteristic::PROPERTY_READ
                            );
      pFirmwareCharacteristic = pService->createCharacteristic(
                              BLEUUID(UUID16_CHR_FIRMWARE_REVISION_STRING),
                              BLECharacteristic::PROPERTY_READ
                            );
      pHardwareCharacteristic = pService->createCharacteristic(
                              BLEUUID(UUID16_CHR_HARDWARE_REVISION_STRING),
                              BLECharacteristic::PROPERTY_READ
                            );
      pSoftwareCharacteristic = pService->createCharacteristic(
                              BLEUUID(UUID16_CHR_SOFTWARE_REVISION_STRING),
                              BLECharacteristic::PROPERTY_READ
                            );
      pManufacturerCharacteristic = pService->createCharacteristic(
                              BLEUUID(UUID16_CHR_MANUFACTURER_NAME_STRING),
                              BLECharacteristic::PROPERTY_READ
                            );

      const char *Model         = hw_info.model == SOFTRF_MODEL_STANDALONE ? "Standalone Edition" :
                                  hw_info.model == SOFTRF_MODEL_PRIME_MK2  ? "Prime Mark II"      :
                                  hw_info.model == SOFTRF_MODEL_PRIME_MK3  ? "Prime Mark III"     :
                                  hw_info.model == SOFTRF_MODEL_HAM        ? "Ham Edition"        :
                                  hw_info.model == SOFTRF_MODEL_MIDI       ? "Midi Edition"       :
                                  "Unknown";
      char SerialNum[9];
      snprintf(SerialNum, sizeof(SerialNum), "%08X", SoC->getChipId());

      const char *Firmware      = "Arduino ESP32 " ARDUINO_ESP32_RELEASE;

      char Hardware[9];
      snprintf(Hardware, sizeof(Hardware), "%08X", hw_info.revision);

      const char *Manufacturer  = SOFTRF_IDENT;
      const char *Software      = SOFTRF_FIRMWARE_VERSION;

      pModelCharacteristic->       setValue((uint8_t *) Model,        strlen(Model));
      pSerialCharacteristic->      setValue((uint8_t *) SerialNum,    strlen(SerialNum));
      pFirmwareCharacteristic->    setValue((uint8_t *) Firmware,     strlen(Firmware));
      pHardwareCharacteristic->    setValue((uint8_t *) Hardware,     strlen(Hardware));
      pSoftwareCharacteristic->    setValue((uint8_t *) Software,     strlen(Software));
      pManufacturerCharacteristic->setValue((uint8_t *) Manufacturer, strlen(Manufacturer));

      // Start the service
      pService->start();

#if defined(USE_BLE_MIDI)
      // Create the BLE Service
      pService = pServer->createService(BLEUUID(MIDI_SERVICE_UUID));

      // Create a BLE Characteristic
      pMIDICharacteristic = pService->createCharacteristic(
                              BLEUUID(MIDI_CHARACTERISTIC_UUID),
                              BLECharacteristic::PROPERTY_READ   |
                              BLECharacteristic::PROPERTY_WRITE  |
                              BLECharacteristic::PROPERTY_NOTIFY |
                              BLECharacteristic::PROPERTY_WRITE_NR
                            );

      // Create a BLE Descriptor
      pMIDICharacteristic->addDescriptor(new BLE2902());

      // Start the service
      pService->start();
#endif /* USE_BLE_MIDI */

      // Start advertising
      BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
#if 0
      pAdvertising->addServiceUUID(BLEUUID(UART_SERVICE_UUID16));
      pAdvertising->addServiceUUID(BLEUUID(UUID16_SVC_BATTERY));
#if defined(USE_BLE_MIDI)
      pAdvertising->addServiceUUID(BLEUUID(MIDI_SERVICE_UUID));
#endif /* USE_BLE_MIDI */
#else
      /* work around https://github.com/espressif/arduino-esp32/issues/6750 */
      BLEAdvertisementData BLEAdvData;
      BLEAdvData.setFlags(0x06);
      BLEAdvData.setCompleteServices(BLEUUID(UART_SERVICE_UUID16));
      BLEAdvData.setCompleteServices(BLEUUID(UUID16_SVC_BATTERY));
#if defined(USE_BLE_MIDI)
      BLEAdvData.setCompleteServices(BLEUUID(MIDI_SERVICE_UUID));
#endif /* USE_BLE_MIDI */
      pAdvertising->setAdvertisementData(BLEAdvData);
#endif
      pAdvertising->setScanResponse(true);
      pAdvertising->setMinPreferred(0x06);  // functions that help with iPhone connections issue
      pAdvertising->setMaxPreferred(0x12);
      BLEDevice::startAdvertising();

      BLE_Advertising_TimeMarker = millis();
    }
    break;
  case BLUETOOTH_A2DP_SOURCE:
#if defined(ENABLE_BT_VOICE)
    void bt_app_main(void);

    bt_app_main();
#endif
    break;
  case BLUETOOTH_NONE:
  default:
    break;
  }
}

static void ESP32_Bluetooth_loop()
{
  switch (settings->bluetooth)
  {
  case BLUETOOTH_LE_HM10_SERIAL:
    {
      // notify changed value
      // bluetooth stack will go into congestion, if too many packets are sent
      if (deviceConnected && (millis() - BLE_Notify_TimeMarker > 10)) { /* < 18000 baud */

          uint8_t chunk[BLE_MAX_WRITE_CHUNK_SIZE];
          size_t size = BLE_FIFO_TX->available();
          size = size < BLE_MAX_WRITE_CHUNK_SIZE ? size : BLE_MAX_WRITE_CHUNK_SIZE;

          if (size > 0) {
            BLE_FIFO_TX->read((char *) chunk, size);

            pUARTCharacteristic->setValue(chunk, size);
            pUARTCharacteristic->notify();
          }

          BLE_Notify_TimeMarker = millis();
      }
      // disconnecting
      if (!deviceConnected && oldDeviceConnected && (millis() - BLE_Advertising_TimeMarker > 500) ) {
          // give the bluetooth stack the chance to get things ready
          pServer->startAdvertising(); // restart advertising
          oldDeviceConnected = deviceConnected;
          BLE_Advertising_TimeMarker = millis();
      }
      // connecting
      if (deviceConnected && !oldDeviceConnected) {
          // do stuff here on connecting
          oldDeviceConnected = deviceConnected;
      }
      if (deviceConnected && isTimeToBattery()) {
        uint8_t battery_level = Battery_charge();

        pBATCharacteristic->setValue(&battery_level, 1);
        pBATCharacteristic->notify();
      }
    }
    break;
  case BLUETOOTH_NONE:
  case BLUETOOTH_SPP:
  case BLUETOOTH_A2DP_SOURCE:
  default:
    break;
  }
}

static void ESP32_Bluetooth_fini()
{
  /* TBD */
}

static int ESP32_Bluetooth_available()
{
  int rval = 0;

  switch (settings->bluetooth)
  {
#if defined(CONFIG_IDF_TARGET_ESP32)
  case BLUETOOTH_SPP:
    rval = SerialBT.available();
    break;
#endif /* CONFIG_IDF_TARGET_ESP32 */
  case BLUETOOTH_LE_HM10_SERIAL:
    rval = BLE_FIFO_RX->available();
    break;
  case BLUETOOTH_NONE:
  case BLUETOOTH_A2DP_SOURCE:
  default:
    break;
  }

  return rval;
}

static int ESP32_Bluetooth_read()
{
  int rval = -1;

  switch (settings->bluetooth)
  {
#if defined(CONFIG_IDF_TARGET_ESP32)
  case BLUETOOTH_SPP:
    rval = SerialBT.read();
    break;
#endif /* CONFIG_IDF_TARGET_ESP32 */
  case BLUETOOTH_LE_HM10_SERIAL:
    rval = BLE_FIFO_RX->read();
    break;
  case BLUETOOTH_NONE:
  case BLUETOOTH_A2DP_SOURCE:
  default:
    break;
  }

  return rval;
}

static size_t ESP32_Bluetooth_write(const uint8_t *buffer, size_t size)
{
  size_t rval = size;

  switch (settings->bluetooth)
  {
#if defined(CONFIG_IDF_TARGET_ESP32)
  case BLUETOOTH_SPP:
    rval = SerialBT.write(buffer, size);
    break;
#endif /* CONFIG_IDF_TARGET_ESP32 */
  case BLUETOOTH_LE_HM10_SERIAL:
    rval = BLE_FIFO_TX->write((char *) buffer,
                        (BLE_FIFO_TX->room() > size ? size : BLE_FIFO_TX->room()));
    break;
  case BLUETOOTH_NONE:
  case BLUETOOTH_A2DP_SOURCE:
  default:
    break;
  }

  return rval;
}

IODev_ops_t ESP32_Bluetooth_ops = {
  "ESP32 Bluetooth",
  ESP32_Bluetooth_setup,
  ESP32_Bluetooth_loop,
  ESP32_Bluetooth_fini,
  ESP32_Bluetooth_available,
  ESP32_Bluetooth_read,
  ESP32_Bluetooth_write
};

#if defined(ENABLE_BT_VOICE)

/*
   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "freertos/xtensa_api.h"
#include "freertos/FreeRTOSConfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "esp_log.h"

static void bt_app_task_handler(void *arg);
static bool bt_app_send_msg(bt_app_msg_t *msg);
static void bt_app_work_dispatched(bt_app_msg_t *msg);

static xQueueHandle bt_app_task_queue = NULL;
static xTaskHandle bt_app_task_handle = NULL;

bool bt_app_work_dispatch(bt_app_cb_t p_cback, uint16_t event, void *p_params, int param_len, bt_app_copy_cb_t p_copy_cback)
{
    ESP_LOGD(BT_APP_CORE_TAG, "%s event 0x%x, param len %d", __func__, event, param_len);

    bt_app_msg_t msg;
    memset(&msg, 0, sizeof(bt_app_msg_t));

    msg.sig = BT_APP_SIG_WORK_DISPATCH;
    msg.event = event;
    msg.cb = p_cback;

    if (param_len == 0) {
        return bt_app_send_msg(&msg);
    } else if (p_params && param_len > 0) {
        if ((msg.param = malloc(param_len)) != NULL) {
            memcpy(msg.param, p_params, param_len);
            /* check if caller has provided a copy callback to do the deep copy */
            if (p_copy_cback) {
                p_copy_cback(&msg, msg.param, p_params);
            }
            return bt_app_send_msg(&msg);
        }
    }

    return false;
}

static bool bt_app_send_msg(bt_app_msg_t *msg)
{
    if (msg == NULL) {
        return false;
    }

    if (xQueueSend(bt_app_task_queue, msg, 10 / portTICK_RATE_MS) != pdTRUE) {
        ESP_LOGE(BT_APP_CORE_TAG, "%s xQueue send failed", __func__);
        return false;
    }
    return true;
}

static void bt_app_work_dispatched(bt_app_msg_t *msg)
{
    if (msg->cb) {
        msg->cb(msg->event, msg->param);
    }
}

static void bt_app_task_handler(void *arg)
{
    bt_app_msg_t msg;
    for (;;) {
        if (pdTRUE == xQueueReceive(bt_app_task_queue, &msg, (portTickType)portMAX_DELAY)) {
            ESP_LOGD(BT_APP_CORE_TAG, "%s, sig 0x%x, 0x%x", __func__, msg.sig, msg.event);
            switch (msg.sig) {
            case BT_APP_SIG_WORK_DISPATCH:
                bt_app_work_dispatched(&msg);
                break;
            default:
                ESP_LOGW(BT_APP_CORE_TAG, "%s, unhandled sig: %d", __func__, msg.sig);
                break;
            } // switch (msg.sig)

            if (msg.param) {
                free(msg.param);
            }
        }
    }
}

void bt_app_task_start_up(void)
{
    bt_app_task_queue = xQueueCreate(10, sizeof(bt_app_msg_t));
    xTaskCreate(bt_app_task_handler, "BtAppT", 2048, NULL, configMAX_PRIORITIES - 3, &bt_app_task_handle);
    return;
}

void bt_app_task_shut_down(void)
{
    if (bt_app_task_handle) {
        vTaskDelete(bt_app_task_handle);
        bt_app_task_handle = NULL;
    }
    if (bt_app_task_queue) {
        vQueueDelete(bt_app_task_queue);
        bt_app_task_queue = NULL;
    }
}

/*
   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/unistd.h>
#include <sys/stat.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "esp_system.h"
#include "esp_log.h"

#include "esp_bt.h"
#include "esp_bt_main.h"
#include "esp_bt_device.h"
#include "esp_gap_bt_api.h"
#include "esp_a2dp_api.h"
#include "esp_avrc_api.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "driver/sdspi_host.h"
#include "sdmmc_cmd.h"

#define BT_AV_TAG               "BT_AV"

/* event for handler "bt_av_hdl_stack_up */
enum {
    BT_APP_EVT_STACK_UP = 0,
};

/* A2DP global state */
enum {
    APP_AV_STATE_IDLE,
    APP_AV_STATE_DISCOVERING,
    APP_AV_STATE_DISCOVERED,
    APP_AV_STATE_UNCONNECTED,
    APP_AV_STATE_CONNECTING,
    APP_AV_STATE_CONNECTED,
    APP_AV_STATE_DISCONNECTING,
};

/* sub states of APP_AV_STATE_CONNECTED */
enum {
    APP_AV_MEDIA_STATE_IDLE,
    APP_AV_MEDIA_STATE_STARTING,
    APP_AV_MEDIA_STATE_STARTED,
    APP_AV_MEDIA_STATE_STOPPING,
};

#define BT_APP_HEART_BEAT_EVT                (0xff00)

/// handler for bluetooth stack enabled events
static void bt_av_hdl_stack_evt(uint16_t event, void *p_param);

/// callback function for A2DP source
static void bt_app_a2d_cb(esp_a2d_cb_event_t event, esp_a2d_cb_param_t *param);

/// callback function for A2DP source audio data stream
static int32_t bt_app_a2d_data_cb(uint8_t *data, int32_t len);

static void a2d_app_heart_beat(void *arg);

/// A2DP application state machine
static void bt_app_av_sm_hdlr(uint16_t event, void *param);

/* A2DP application state machine handler for each state */
static void bt_app_av_state_unconnected(uint16_t event, void *param);
static void bt_app_av_state_connecting(uint16_t event, void *param);
static void bt_app_av_state_connected(uint16_t event, void *param);
static void bt_app_av_state_disconnecting(uint16_t event, void *param);

static esp_bd_addr_t peer_bda = {0};
static uint8_t peer_bdname[ESP_BT_GAP_MAX_BDNAME_LEN + 1];
static int m_a2d_state = APP_AV_STATE_IDLE;
static int m_media_state = APP_AV_MEDIA_STATE_IDLE;
static int m_intv_cnt = 0;
static int m_connecting_intv = 0;
static uint32_t m_pkt_cnt = 0;

TimerHandle_t tmr;

int m_sample = 0;

static char *bda2str(esp_bd_addr_t bda, char *str, size_t size)
{
    if (bda == NULL || str == NULL || size < 18) {
        return NULL;
    }

    uint8_t *p = bda;
    sprintf(str, "%02x:%02x:%02x:%02x:%02x:%02x",
            p[0], p[1], p[2], p[3], p[4], p[5]);
    return str;
}

void bt_app_main()
{
    // Initialize NVS.
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK( ret );

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();

    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();

    gpio_set_pull_mode(GPIO_NUM_15, GPIO_PULLUP_ONLY);   // CMD, needed in 4- and 1- line modes
    gpio_set_pull_mode(GPIO_NUM_2, GPIO_PULLUP_ONLY);    // D0, needed in 4- and 1-line modes
    gpio_set_pull_mode(GPIO_NUM_4, GPIO_PULLUP_ONLY);    // D1, needed in 4-line mode only
    gpio_set_pull_mode(GPIO_NUM_12, GPIO_PULLUP_ONLY);   // D2, needed in 4-line mode only
    gpio_set_pull_mode(GPIO_NUM_13, GPIO_PULLUP_ONLY);   // D3, needed in 4- and 1-line modes
    gpio_pullup_en(GPIO_NUM_15);
    gpio_pullup_en(GPIO_NUM_2);
    gpio_pullup_en(GPIO_NUM_4);
    gpio_pullup_en(GPIO_NUM_12);
    gpio_pullup_en(GPIO_NUM_13);

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024
    };

    sdmmc_card_t* card;
    ret = esp_vfs_fat_sdmmc_mount("/sdcard", &host, &slot_config, &mount_config, &card);

    if (ret != ESP_OK) {
        return;
    }

    sdmmc_card_print_info(stdout, card);
    m_sample = open("/sdcard/voice.wav", O_RDONLY);

    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();

    if (esp_bt_controller_init(&bt_cfg) != ESP_OK) {
        ESP_LOGE(BT_AV_TAG, "%s initialize controller failed\n", __func__);
        return;
    }

    if (esp_bt_controller_enable(ESP_BT_MODE_BTDM) != ESP_OK) {
        ESP_LOGE(BT_AV_TAG, "%s enable controller failed\n", __func__);
        return;
    }

    if (esp_bluedroid_init() != ESP_OK) {
        ESP_LOGE(BT_AV_TAG, "%s initialize bluedroid failed\n", __func__);
        return;
    }

    if (esp_bluedroid_enable() != ESP_OK) {
        ESP_LOGE(BT_AV_TAG, "%s enable bluedroid failed\n", __func__);
        return;
    }

    /* create application task */
    bt_app_task_start_up();

    /* Bluetooth device name, connection mode and profile set up */
    bt_app_work_dispatch(bt_av_hdl_stack_evt, BT_APP_EVT_STACK_UP, NULL, 0, NULL);
}

static bool get_name_from_eir(uint8_t *eir, uint8_t *bdname, uint8_t *bdname_len)
{
    uint8_t *rmt_bdname = NULL;
    uint8_t rmt_bdname_len = 0;

    if (!eir) {
        return false;
    }

    rmt_bdname = esp_bt_gap_resolve_eir_data(eir, ESP_BT_EIR_TYPE_CMPL_LOCAL_NAME, &rmt_bdname_len);
    if (!rmt_bdname) {
        rmt_bdname = esp_bt_gap_resolve_eir_data(eir, ESP_BT_EIR_TYPE_SHORT_LOCAL_NAME, &rmt_bdname_len);
    }

    if (rmt_bdname) {
        if (rmt_bdname_len > ESP_BT_GAP_MAX_BDNAME_LEN) {
            rmt_bdname_len = ESP_BT_GAP_MAX_BDNAME_LEN;
        }

        if (bdname) {
            memcpy(bdname, rmt_bdname, rmt_bdname_len);
            bdname[rmt_bdname_len] = '\0';
        }
        if (bdname_len) {
            *bdname_len = rmt_bdname_len;
        }
        return true;
    }

    return false;
}

static void filter_inquiry_scan_result(esp_bt_gap_cb_param_t *param)
{
    char bda_str[18];
    uint32_t cod = 0;
    int32_t rssi = -129; /* invalid value */
    uint8_t *eir = NULL;
    esp_bt_gap_dev_prop_t *p;

    ESP_LOGI(BT_AV_TAG, "Scanned device: %s", bda2str(param->disc_res.bda, bda_str, 18));
    for (int i = 0; i < param->disc_res.num_prop; i++) {
        p = param->disc_res.prop + i;
        switch (p->type) {
        case ESP_BT_GAP_DEV_PROP_COD:
            cod = *(uint32_t *)(p->val);
            ESP_LOGI(BT_AV_TAG, "--Class of Device: 0x%x", cod);
            break;
        case ESP_BT_GAP_DEV_PROP_RSSI:
            rssi = *(int8_t *)(p->val);
            ESP_LOGI(BT_AV_TAG, "--RSSI: %d", rssi);
            break;
        case ESP_BT_GAP_DEV_PROP_EIR:
            eir = (uint8_t *)(p->val);
            break;
        case ESP_BT_GAP_DEV_PROP_BDNAME:
        default:
            break;
        }
    }

    /* search for device with MAJOR service class as "rendering" in COD */
    if (!esp_bt_gap_is_valid_cod(cod) ||
            !(esp_bt_gap_get_cod_srvc(cod) & ESP_BT_COD_SRVC_RENDERING)) {
        return;
    }

    /* search for device named "BT SPEAKER" in its extended inqury response */
    if (eir) {
        get_name_from_eir(eir, peer_bdname, NULL);
        if (strcmp((char *)peer_bdname, "raspberrypi") != 0) {
            return;
        }

        ESP_LOGI(BT_AV_TAG, "Found a target device, address %s, name %s", bda_str, peer_bdname);
        m_a2d_state = APP_AV_STATE_DISCOVERED;
        memcpy(peer_bda, param->disc_res.bda, ESP_BD_ADDR_LEN);
        ESP_LOGI(BT_AV_TAG, "Cancel device discovery ...");
        esp_bt_gap_cancel_discovery();
    }
}


void bt_app_gap_cb(esp_bt_gap_cb_event_t event, esp_bt_gap_cb_param_t *param)
{
    switch (event) {
    case ESP_BT_GAP_DISC_RES_EVT: {
        filter_inquiry_scan_result(param);
        break;
    }
    case ESP_BT_GAP_DISC_STATE_CHANGED_EVT: {
        if (param->disc_st_chg.state == ESP_BT_GAP_DISCOVERY_STOPPED) {
            if (m_a2d_state == APP_AV_STATE_DISCOVERED) {
                m_a2d_state = APP_AV_STATE_CONNECTING;
                ESP_LOGI(BT_AV_TAG, "Device discovery stopped.");
                ESP_LOGI(BT_AV_TAG, "a2dp connecting to peer: %s", peer_bdname);
                esp_a2d_source_connect(peer_bda);
            } else {
                // not discovered, continue to discover
                ESP_LOGI(BT_AV_TAG, "Device discovery failed, continue to discover...");
                esp_bt_gap_start_discovery(ESP_BT_INQ_MODE_GENERAL_INQUIRY, 10, 0);
            }
        } else if (param->disc_st_chg.state == ESP_BT_GAP_DISCOVERY_STARTED) {
            ESP_LOGI(BT_AV_TAG, "Discovery started.");
        }
        break;
    }
    case ESP_BT_GAP_RMT_SRVCS_EVT:
    case ESP_BT_GAP_RMT_SRVC_REC_EVT:
    default: {
        ESP_LOGI(BT_AV_TAG, "event: %d", event);
        break;
    }
    }
    return;
}

static void bt_av_hdl_stack_evt(uint16_t event, void *p_param)
{
    ESP_LOGD(BT_AV_TAG, "%s evt %d", __func__, event);
    switch (event) {
    case BT_APP_EVT_STACK_UP: {
        /* set up device name */
        char *dev_name = "ESP_A2DP_SRC";
        esp_bt_dev_set_device_name(dev_name);

        /* register GAP callback function */
        esp_bt_gap_register_callback(bt_app_gap_cb);

        /* initialize A2DP source */
        esp_a2d_register_callback(&bt_app_a2d_cb);
        esp_a2d_source_register_data_callback(bt_app_a2d_data_cb);
        esp_a2d_source_init();

        /* set discoverable and connectable mode */
        esp_bt_gap_set_scan_mode(ESP_BT_SCAN_MODE_CONNECTABLE_DISCOVERABLE);

        /* start device discovery */
        ESP_LOGI(BT_AV_TAG, "Starting device discovery...");
        m_a2d_state = APP_AV_STATE_DISCOVERING;
        esp_bt_gap_start_discovery(ESP_BT_INQ_MODE_GENERAL_INQUIRY, 10, 0);

        /* create and start heart beat timer */
        do {
            int tmr_id = 0;
            tmr = xTimerCreate("connTmr", (10000 / portTICK_RATE_MS),
                               pdTRUE, (void *)tmr_id, a2d_app_heart_beat);
            xTimerStart(tmr, portMAX_DELAY);
        } while (0);
        break;
    }
    default:
        ESP_LOGE(BT_AV_TAG, "%s unhandled evt %d", __func__, event);
        break;
    }
}

static void bt_app_a2d_cb(esp_a2d_cb_event_t event, esp_a2d_cb_param_t *param)
{
    bt_app_work_dispatch(bt_app_av_sm_hdlr, event, param, sizeof(esp_a2d_cb_param_t), NULL);
}

static int32_t bt_app_a2d_data_cb(uint8_t *data, int32_t len)
{
    if (len < 0 || data == NULL) {
        return 0;
    }

    int l = read(m_sample, data, len);
    if (l < len) {
        lseek(m_sample, 0, SEEK_SET);
    }
    return len;
}

static void a2d_app_heart_beat(void *arg)
{
    bt_app_work_dispatch(bt_app_av_sm_hdlr, BT_APP_HEART_BEAT_EVT, NULL, 0, NULL);
}

static void bt_app_av_sm_hdlr(uint16_t event, void *param)
{
    ESP_LOGI(BT_AV_TAG, "%s state %d, evt 0x%x", __func__, m_a2d_state, event);
    switch (m_a2d_state) {
    case APP_AV_STATE_DISCOVERING:
    case APP_AV_STATE_DISCOVERED:
        break;
    case APP_AV_STATE_UNCONNECTED:
        bt_app_av_state_unconnected(event, param);
        break;
    case APP_AV_STATE_CONNECTING:
        bt_app_av_state_connecting(event, param);
        break;
    case APP_AV_STATE_CONNECTED:
        bt_app_av_state_connected(event, param);
        break;
    case APP_AV_STATE_DISCONNECTING:
        bt_app_av_state_disconnecting(event, param);
        break;
    default:
        ESP_LOGE(BT_AV_TAG, "%s invalid state %d", __func__, m_a2d_state);
        break;
    }
}

static void bt_app_av_state_unconnected(uint16_t event, void *param)
{
    switch (event) {
    case ESP_A2D_CONNECTION_STATE_EVT:
    case ESP_A2D_AUDIO_STATE_EVT:
    case ESP_A2D_AUDIO_CFG_EVT:
    case ESP_A2D_MEDIA_CTRL_ACK_EVT:
        break;
    case BT_APP_HEART_BEAT_EVT: {
        uint8_t *p = peer_bda;
        ESP_LOGI(BT_AV_TAG, "a2dp connecting to peer: %02x:%02x:%02x:%02x:%02x:%02x",
                 p[0], p[1], p[2], p[3], p[4], p[5]);
        esp_a2d_source_connect(peer_bda);
        m_a2d_state = APP_AV_STATE_CONNECTING;
        m_connecting_intv = 0;
        break;
    }
    default:
        ESP_LOGE(BT_AV_TAG, "%s unhandled evt %d", __func__, event);
        break;
    }
}

static void bt_app_av_state_connecting(uint16_t event, void *param)
{
    esp_a2d_cb_param_t *a2d = NULL;
    switch (event) {
    case ESP_A2D_CONNECTION_STATE_EVT: {
        a2d = (esp_a2d_cb_param_t *)(param);
        if (a2d->conn_stat.state == ESP_A2D_CONNECTION_STATE_CONNECTED) {
            ESP_LOGI(BT_AV_TAG, "a2dp connected");
            m_a2d_state =  APP_AV_STATE_CONNECTED;
            m_media_state = APP_AV_MEDIA_STATE_IDLE;

        } else if (a2d->conn_stat.state == ESP_A2D_CONNECTION_STATE_DISCONNECTED) {
            m_a2d_state =  APP_AV_STATE_UNCONNECTED;
        }
        break;
    }
    case ESP_A2D_AUDIO_STATE_EVT:
    case ESP_A2D_AUDIO_CFG_EVT:
    case ESP_A2D_MEDIA_CTRL_ACK_EVT:
        break;
    case BT_APP_HEART_BEAT_EVT:
        if (++m_connecting_intv >= 2) {
            m_a2d_state = APP_AV_STATE_UNCONNECTED;
            m_connecting_intv = 0;
        }
        break;
    default:
        ESP_LOGE(BT_AV_TAG, "%s unhandled evt %d", __func__, event);
        break;
    }
}

static void bt_app_av_media_proc(uint16_t event, void *param)
{
    esp_a2d_cb_param_t *a2d = NULL;
    switch (m_media_state) {
    case APP_AV_MEDIA_STATE_IDLE: {
        if (event == BT_APP_HEART_BEAT_EVT) {
            ESP_LOGI(BT_AV_TAG, "a2dp media ready checking ...");
            esp_a2d_media_ctrl(ESP_A2D_MEDIA_CTRL_CHECK_SRC_RDY);
        } else if (event == ESP_A2D_MEDIA_CTRL_ACK_EVT) {
            a2d = (esp_a2d_cb_param_t *)(param);
            if (a2d->media_ctrl_stat.cmd == ESP_A2D_MEDIA_CTRL_CHECK_SRC_RDY &&
                    a2d->media_ctrl_stat.status == ESP_A2D_MEDIA_CTRL_ACK_SUCCESS) {
                ESP_LOGI(BT_AV_TAG, "a2dp media ready, starting ...");
                esp_a2d_media_ctrl(ESP_A2D_MEDIA_CTRL_START);
                m_media_state = APP_AV_MEDIA_STATE_STARTING;
            }
        }
        break;
    }
    case APP_AV_MEDIA_STATE_STARTING: {
        if (event == ESP_A2D_MEDIA_CTRL_ACK_EVT) {
            a2d = (esp_a2d_cb_param_t *)(param);
            if (a2d->media_ctrl_stat.cmd == ESP_A2D_MEDIA_CTRL_START &&
                    a2d->media_ctrl_stat.status == ESP_A2D_MEDIA_CTRL_ACK_SUCCESS) {
                ESP_LOGI(BT_AV_TAG, "a2dp media start successfully.");
                m_intv_cnt = 0;
                m_media_state = APP_AV_MEDIA_STATE_STARTED;
            } else {
                // not started succesfully, transfer to idle state
                ESP_LOGI(BT_AV_TAG, "a2dp media start failed.");
                m_media_state = APP_AV_MEDIA_STATE_IDLE;
            }
        }
        break;
    }
    case APP_AV_MEDIA_STATE_STARTED: {
        if (event == BT_APP_HEART_BEAT_EVT) {
            if (++m_intv_cnt >= 10) {
                ESP_LOGI(BT_AV_TAG, "a2dp media stopping...");
                esp_a2d_media_ctrl(ESP_A2D_MEDIA_CTRL_STOP);
                m_media_state = APP_AV_MEDIA_STATE_STOPPING;
                m_intv_cnt = 0;
            }
        }
        break;
    }
    case APP_AV_MEDIA_STATE_STOPPING: {
        if (event == ESP_A2D_MEDIA_CTRL_ACK_EVT) {
            a2d = (esp_a2d_cb_param_t *)(param);
            if (a2d->media_ctrl_stat.cmd == ESP_A2D_MEDIA_CTRL_STOP &&
                    a2d->media_ctrl_stat.status == ESP_A2D_MEDIA_CTRL_ACK_SUCCESS) {
                ESP_LOGI(BT_AV_TAG, "a2dp media stopped successfully, disconnecting...");
                m_media_state = APP_AV_MEDIA_STATE_IDLE;
                esp_a2d_source_disconnect(peer_bda);
                m_a2d_state = APP_AV_STATE_DISCONNECTING;
            } else {
                ESP_LOGI(BT_AV_TAG, "a2dp media stopping...");
                esp_a2d_media_ctrl(ESP_A2D_MEDIA_CTRL_STOP);
            }
        }
        break;
    }
    }
}

static void bt_app_av_state_connected(uint16_t event, void *param)
{
    esp_a2d_cb_param_t *a2d = NULL;
    switch (event) {
    case ESP_A2D_CONNECTION_STATE_EVT: {
        a2d = (esp_a2d_cb_param_t *)(param);
        if (a2d->conn_stat.state == ESP_A2D_CONNECTION_STATE_DISCONNECTED) {
            ESP_LOGI(BT_AV_TAG, "a2dp disconnected");
            m_a2d_state = APP_AV_STATE_UNCONNECTED;
        }
        break;
    }
    case ESP_A2D_AUDIO_STATE_EVT: {
        a2d = (esp_a2d_cb_param_t *)(param);
        if (ESP_A2D_AUDIO_STATE_STARTED == a2d->audio_stat.state) {
            m_pkt_cnt = 0;
        }
        break;
    }
    case ESP_A2D_AUDIO_CFG_EVT:
        // not suppposed to occur for A2DP source
        break;
    case ESP_A2D_MEDIA_CTRL_ACK_EVT:
    case BT_APP_HEART_BEAT_EVT: {
        bt_app_av_media_proc(event, param);
        break;
    }
    default:
        ESP_LOGE(BT_AV_TAG, "%s unhandled evt %d", __func__, event);
        break;
    }
}

static void bt_app_av_state_disconnecting(uint16_t event, void *param)
{
    esp_a2d_cb_param_t *a2d = NULL;
    switch (event) {
    case ESP_A2D_CONNECTION_STATE_EVT: {
        a2d = (esp_a2d_cb_param_t *)(param);
        if (a2d->conn_stat.state == ESP_A2D_CONNECTION_STATE_DISCONNECTED) {
            ESP_LOGI(BT_AV_TAG, "a2dp disconnected");
            m_a2d_state =  APP_AV_STATE_UNCONNECTED;
        }
        break;
    }
    case ESP_A2D_AUDIO_STATE_EVT:
    case ESP_A2D_AUDIO_CFG_EVT:
    case ESP_A2D_MEDIA_CTRL_ACK_EVT:
    case BT_APP_HEART_BEAT_EVT:
        break;
    default:
        ESP_LOGE(BT_AV_TAG, "%s unhandled evt %d", __func__, event);
        break;
    }
}

#endif /* ENABLE_BT_VOICE */

#elif defined(ARDUINO_ARCH_NRF52)

#include "../system/SoC.h"
#include "Bluetooth.h"

#include <bluefruit.h>
#include <Adafruit_LittleFS.h>
#include <InternalFileSystem.h>
#include <BLEUart_HM10.h>
#include <TinyGPS++.h>
#if defined(USE_BLE_MIDI)
#include <MIDI.h>
#endif /* USE_BLE_MIDI */

#include "WiFi.h"
#include "Battery.h"
#include "GNSS.h"
#include "RF.h"
#include "../protocol/radio/Legacy.h"
#include "Baro.h"
#include "EEPROM.h"
#include "Sound.h"
#include "../protocol/data/NMEA.h"
#include "../protocol/data/GDL90.h"
#include "../protocol/data/D1090.h"

/*
 * SensorBox Serivce: aba27100-143b-4b81-a444-edcd0000f020
 * Navigation       : aba27100-143b-4b81-a444-edcd0000f022
 * Movement         : aba27100-143b-4b81-a444-edcd0000f023
 * GPS2             : aba27100-143b-4b81-a444-edcd0000f024
 * System           : aba27100-143b-4b81-a444-edcd0000f025
 */

const uint8_t SENSBOX_UUID_SERVICE[] =
{
    0x20, 0xF0, 0x00, 0x00, 0xCD, 0xED, 0x44, 0xA4,
    0x81, 0x4B, 0x3B, 0x14, 0x00, 0x71, 0xA2, 0xAB,
};

const uint8_t SENSBOX_UUID_NAVIGATION[] =
{
    0x22, 0xF0, 0x00, 0x00, 0xCD, 0xED, 0x44, 0xA4,
    0x81, 0x4B, 0x3B, 0x14, 0x00, 0x71, 0xA2, 0xAB,
};

const uint8_t SENSBOX_UUID_MOVEMENT[] =
{
    0x23, 0xF0, 0x00, 0x00, 0xCD, 0xED, 0x44, 0xA4,
    0x81, 0x4B, 0x3B, 0x14, 0x00, 0x71, 0xA2, 0xAB,
};

const uint8_t SENSBOX_UUID_GPS2[] =
{
    0x24, 0xF0, 0x00, 0x00, 0xCD, 0xED, 0x44, 0xA4,
    0x81, 0x4B, 0x3B, 0x14, 0x00, 0x71, 0xA2, 0xAB,
};

const uint8_t SENSBOX_UUID_SYSTEM[] =
{
    0x25, 0xF0, 0x00, 0x00, 0xCD, 0xED, 0x44, 0xA4,
    0x81, 0x4B, 0x3B, 0x14, 0x00, 0x71, 0xA2, 0xAB,
};

BLESensBox::BLESensBox(void) :
  BLEService   (SENSBOX_UUID_SERVICE),
  _sensbox_nav (SENSBOX_UUID_NAVIGATION),
  _sensbox_move(SENSBOX_UUID_MOVEMENT),
  _sensbox_gps2(SENSBOX_UUID_GPS2),
  _sensbox_sys (SENSBOX_UUID_SYSTEM)
{

}

err_t BLESensBox::begin(void)
{
  VERIFY_STATUS( BLEService::begin() );

  _sensbox_nav.setProperties(CHR_PROPS_NOTIFY);
  _sensbox_nav.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  _sensbox_nav.setFixedLen(sizeof(sensbox_navigation_t));
  _sensbox_move.setProperties(CHR_PROPS_NOTIFY);
  _sensbox_move.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  _sensbox_move.setFixedLen(sizeof(sensbox_movement_t));
  _sensbox_gps2.setProperties(CHR_PROPS_NOTIFY);
  _sensbox_gps2.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  _sensbox_gps2.setFixedLen(sizeof(sensbox_gps2_t));
  _sensbox_sys.setProperties(CHR_PROPS_NOTIFY);
  _sensbox_sys.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  _sensbox_sys.setFixedLen(sizeof(sensbox_system_t));
  VERIFY_STATUS( _sensbox_nav.begin()  );
  VERIFY_STATUS( _sensbox_move.begin() );
  VERIFY_STATUS( _sensbox_gps2.begin() );
  VERIFY_STATUS( _sensbox_sys.begin()  );

  return ERROR_NONE;
}

bool BLESensBox::notify_nav(uint8_t status)
{
  if (!_sensbox_nav.notifyEnabled(Bluefruit.connHandle()))
    return false;

  sensbox_navigation_t data = {0};

  data.timestamp = ThisAircraft.timestamp;
  data.lat       = (int32_t) (ThisAircraft.latitude  * 10000000);
  data.lon       = (int32_t) (ThisAircraft.longitude * 10000000);
  data.gnss_alt  = (int16_t) ThisAircraft.altitude;
  data.pres_alt  = (int16_t) ThisAircraft.pressure_altitude;
  data.status    = status;

  return _sensbox_nav.notify(&data, sizeof(sensbox_navigation_t)) > 0;
}

bool BLESensBox::notify_move(uint8_t status)
{
  if (!_sensbox_move.notifyEnabled(Bluefruit.connHandle()))
    return false;

  sensbox_movement_t data = {0};

  data.pres_alt  = (int32_t) (ThisAircraft.pressure_altitude * 100);
  data.vario     = (int16_t) ((ThisAircraft.vs * 10) / (_GPS_FEET_PER_METER * 6));
  data.gs        = (int16_t) (ThisAircraft.speed  * _GPS_MPS_PER_KNOT * 10);
  data.cog       = (int16_t) (ThisAircraft.course * 10);
  data.status    = status;

  return _sensbox_move.notify(&data, sizeof(sensbox_movement_t)) > 0;
}

bool BLESensBox::notify_gps2(uint8_t status)
{
  if (!_sensbox_gps2.notifyEnabled(Bluefruit.connHandle()))
    return false;

  sensbox_gps2_t data = {0};

  data.sats      = (uint8_t) gnss.satellites.value();
  data.status    = status;

  return _sensbox_gps2.notify(&data, sizeof(sensbox_gps2_t)) > 0;
}

bool BLESensBox::notify_sys(uint8_t status)
{
  if (!_sensbox_sys.notifyEnabled(Bluefruit.connHandle()))
    return false;

  sensbox_system_t data = {0};

  data.battery   = (uint8_t) Battery_charge();
  data.temp      = (int16_t) (Baro_temperature() * 10);
  data.status    = status;

  return _sensbox_sys.notify(&data, sizeof(sensbox_system_t)) > 0;
}

static unsigned long BLE_Notify_TimeMarker  = 0;
static unsigned long BLE_SensBox_TimeMarker = 0;

/*********************************************************************
 This is an example for our nRF52 based Bluefruit LE modules

 Pick one up today in the adafruit shop!

 Adafruit invests time and resources providing this open source code,
 please support Adafruit and open-source hardware by purchasing
 products from Adafruit!

 MIT license, check LICENSE for more information
 All text above, and the splash screen below must be included in
 any redistribution
*********************************************************************/

// BLE Service
BLEDfu        bledfu;       // OTA DFU service
BLEDis        bledis;       // device information
BLEUart_HM10  bleuart_HM10; // TI UART over BLE
#if !defined(EXCLUDE_NUS)
BLEUart       bleuart_NUS;  // Nordic UART over BLE
#endif /* EXCLUDE_NUS */
BLEBas        blebas;       // battery
BLESensBox    blesens;      // SensBox

#if defined(USE_BLE_MIDI)
BLEMidi       blemidi;

MIDI_CREATE_INSTANCE(BLEMidi, blemidi, MIDI_BLE);
#endif /* USE_BLE_MIDI */

#if defined(ENABLE_REMOTE_ID)
#include "../protocol/radio/RemoteID.h"

#define UUID16_COMPANY_ID_ASTM 0xFFFA

static unsigned long RID_Time_Marker = 0;

BLEService    BLE_ODID_service;

static const uint8_t ODID_Uuid[] = {0x00, 0x00, 0xff, 0xfa, 0x00, 0x00, 0x10, 0x00,
                                    0x80, 0x00, 0x00, 0x80, 0x5f, 0x9b, 0x34, 0xfb};
#endif /* ENABLE_REMOTE_ID */

String BT_name = HOSTNAME;

#define UUID16_COMPANY_ID_NORDIC 0x0059

static uint8_t BeaconUuid[16] =
{
 /* https://openuuid.net: becf4a85-29b8-476e-928f-fce11f303344 */
  0xbe, 0xcf, 0x4a, 0x85, 0x29, 0xb8, 0x47, 0x6e,
  0x92, 0x8f, 0xfc, 0xe1, 0x1f, 0x30, 0x33, 0x44
};

// UUID, Major, Minor, RSSI @ 1M
BLEBeacon iBeacon(BeaconUuid, 0x0102, 0x0304, -64);

void startAdv(void)
{
  bool no_data = (settings->nmea_out != NMEA_BLUETOOTH  &&
                  settings->gdl90    != GDL90_BLUETOOTH &&
                  settings->d1090    != D1090_BLUETOOTH);

  // Advertising packet

#if defined(USE_IBEACON)
  if (no_data && settings->volume == BUZZER_OFF) {
    uint32_t id = SoC->getChipId();
    uint16_t major = (id >> 16) & 0x0000FFFF;
    uint16_t minor = (id      ) & 0x0000FFFF;

    // Manufacturer ID is required for Manufacturer Specific Data
    iBeacon.setManufacturer(UUID16_COMPANY_ID_NORDIC);
    iBeacon.setMajorMinor(major, minor);

    // Set the beacon payload using the BLEBeacon class
    Bluefruit.Advertising.setBeacon(iBeacon);
  } else
#endif /* USE_IBEACON */
  {
    Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
    Bluefruit.Advertising.addTxPower();

#if defined(ENABLE_REMOTE_ID)
    if (rid_enabled()) {
      Bluefruit.Advertising.addService(BLE_ODID_service);
      Bluefruit.Advertising.addName();
    } else
#endif /* ENABLE_REMOTE_ID */
#if defined(USE_BLE_MIDI)
    if (settings->volume != BUZZER_OFF) {
      Bluefruit.Advertising.addService(blemidi, bleuart_HM10);
    } else
#endif /* USE_BLE_MIDI */
    {
      Bluefruit.Advertising.addService(
#if !defined(EXCLUDE_NUS)
                                       bleuart_NUS,
#endif /* EXCLUDE_NUS */
                                       bleuart_HM10);
    }
  }

  // Secondary Scan Response packet (optional)
  // Since there is no room for 'Name' in Advertising packet
#if defined(ENABLE_REMOTE_ID)
  if (!rid_enabled()) 
#endif /* ENABLE_REMOTE_ID */
  {
    Bluefruit.ScanResponse.addName();
  }

  /* Start Advertising
   * - Enable auto advertising if disconnected
   * - Interval:  fast mode = 20 ms, slow mode = 152.5 ms
   * - Timeout for fast mode is 30 seconds
   * - Start(timeout) with timeout = 0 will advertise forever (until connected)
   *
   * For recommended advertising interval
   * https://developer.apple.com/library/content/qa/qa1931/_index.html
   */
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);    // in unit of 0.625 ms
  Bluefruit.Advertising.setFastTimeout(30);      // number of seconds in fast mode
  Bluefruit.Advertising.start(0);                // 0 = Don't stop advertising after n seconds
}

// callback invoked when central connects
void connect_callback(uint16_t conn_handle)
{
#if DEBUG_BLE
  // Get the reference to current connection
  BLEConnection* connection = Bluefruit.Connection(conn_handle);

  char central_name[32] = { 0 };
  connection->getPeerName(central_name, sizeof(central_name));

  Serial.print("Connected to ");
  Serial.println(central_name);
#endif
}

/**
 * Callback invoked when a connection is dropped
 * @param conn_handle connection where this event happens
 * @param reason is a BLE_HCI_STATUS_CODE which can be found in ble_hci.h
 */
void disconnect_callback(uint16_t conn_handle, uint8_t reason)
{
#if DEBUG_BLE
  (void) conn_handle;
  (void) reason;

  Serial.println();
  Serial.print("Disconnected, reason = 0x"); Serial.println(reason, HEX);
#endif
}

void nRF52_Bluetooth_setup()
{
  BT_name += "-";
  BT_name += String(SoC->getChipId() & 0x00FFFFFFU, HEX);

#if defined(ENABLE_REMOTE_ID)
  rid_init();
  RID_Time_Marker = millis();
#endif /* ENABLE_REMOTE_ID */

  // Setup the BLE LED to be enabled on CONNECT
  // Note: This is actually the default behavior, but provided
  // here in case you want to control this LED manually via PIN 19
  Bluefruit.autoConnLed(LED_BLUE == SOC_GPIO_LED_BLE ? true : false);

  // Config the peripheral connection with maximum bandwidth
  // more SRAM required by SoftDevice
  // Note: All config***() function must be called before begin()
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);

  Bluefruit.begin();
  Bluefruit.setTxPower(4);    // Check bluefruit.h for supported values
  Bluefruit.setName((BT_name+"-LE").c_str());
  Bluefruit.Periph.setConnectCallback(connect_callback);
  Bluefruit.Periph.setDisconnectCallback(disconnect_callback);

  // To be consistent OTA DFU should be added first if it exists
  bledfu.begin();

  // Configure and Start Device Information Service
  bledis.setManufacturer(nRF52_Device_Manufacturer);
  bledis.setModel(nRF52_Device_Model);
  bledis.setHardwareRev(hw_info.revision > 2 ?
                        Hardware_Rev[3] : Hardware_Rev[hw_info.revision]);
  bledis.setSoftwareRev(SOFTRF_FIRMWARE_VERSION);
  bledis.begin();

  // Configure and Start BLE Uart Service
  bleuart_HM10.begin();
#if !defined(EXCLUDE_NUS)
  bleuart_NUS.begin();
  bleuart_NUS.bufferTXD(true);
#endif /* EXCLUDE_NUS */

  // Start BLE Battery Service
  blebas.begin();
  blebas.write(100);

#if defined(ENABLE_REMOTE_ID)
  if (rid_enabled()) {
    BLE_ODID_service.setUuid(BLEUuid(ODID_Uuid /* UUID16_COMPANY_ID_ASTM */));
    BLE_ODID_service.begin();
  }
#endif /* ENABLE_REMOTE_ID */

  // Start SensBox Service
  blesens.begin();

#if defined(USE_BLE_MIDI)
  // Initialize MIDI with no any input channels
  // This will also call blemidi service's begin()
  MIDI_BLE.begin(MIDI_CHANNEL_OFF);
#endif /* USE_BLE_MIDI */

  // Set up and start advertising
  startAdv();

#if DEBUG_BLE
  Serial.println("Please use Adafruit's Bluefruit LE app to connect in UART mode");
  Serial.println("Once connected, enter character(s) that you wish to send");
#endif

  BLE_Notify_TimeMarker  = millis();
  BLE_SensBox_TimeMarker = millis();
}

/*********************************************************************
 End of Adafruit licensed text
*********************************************************************/

static void nRF52_Bluetooth_loop()
{
  // notify changed value
  // bluetooth stack will go into congestion, if too many packets are sent
  if ( Bluefruit.connected()              &&
       bleuart_HM10.notifyEnabled()       &&
       (millis() - BLE_Notify_TimeMarker > 10)) { /* < 18000 baud */
    bleuart_HM10.flushTXD();

    BLE_Notify_TimeMarker = millis();
  }

  if (isTimeToBattery()) {
    blebas.write(Battery_charge());
  }

  if (Bluefruit.connected() && isTimeToSensBox()) {
    uint8_t sens_status = isValidFix() ? GNSS_STATUS_3D_MOVING : GNSS_STATUS_NONE;
    blesens.notify_nav (sens_status);
    blesens.notify_move(sens_status);
    blesens.notify_gps2(sens_status);
    blesens.notify_sys (sens_status);
    BLE_SensBox_TimeMarker = millis();
  }

#if defined(ENABLE_REMOTE_ID)
  if (rid_enabled() && isValidFix()) {
    if ((millis() - RID_Time_Marker) > 74) {
      rid_encode((void *) &utm_data, &ThisAircraft);
      squitter.transmit(&utm_data);

      RID_Time_Marker = millis();
    }
  }
#endif /* ENABLE_REMOTE_ID */
}

static void nRF52_Bluetooth_fini()
{
  uint8_t sd_en = 0;
  (void) sd_softdevice_is_enabled(&sd_en);

  if ( Bluefruit.connected() ) {
    if ( bleuart_HM10.notifyEnabled() ) {
      // flush TXD since we use bufferTXD()
      bleuart_HM10.flushTXD();
    }

#if !defined(EXCLUDE_NUS)
    if ( bleuart_NUS.notifyEnabled() ) {
      // flush TXD since we use bufferTXD()
      bleuart_NUS.flushTXD();
    }
#endif /* EXCLUDE_NUS */
  }

  if (Bluefruit.Advertising.isRunning()) {
    Bluefruit.Advertising.stop();
  }

  if (sd_en) sd_softdevice_disable();
}

static int nRF52_Bluetooth_available()
{
  int rval = 0;

  if ( !Bluefruit.connected() ) {
    return rval;
  }

  /* Give priority to HM-10 input */
  if ( bleuart_HM10.notifyEnabled() ) {
    return bleuart_HM10.available();
  }

#if !defined(EXCLUDE_NUS)
  if ( bleuart_NUS.notifyEnabled() ) {
    rval = bleuart_NUS.available();
  }
#endif /* EXCLUDE_NUS */

  return rval;
}

static int nRF52_Bluetooth_read()
{
  int rval = -1;

  if ( !Bluefruit.connected() ) {
    return rval;
  }

  /* Give priority to HM-10 input */
  if ( bleuart_HM10.notifyEnabled() ) {
    return bleuart_HM10.read();
  }

#if !defined(EXCLUDE_NUS)
  if ( bleuart_NUS.notifyEnabled() ) {
    rval = bleuart_NUS.read();
  }
#endif /* EXCLUDE_NUS */

  return rval;
}

static size_t nRF52_Bluetooth_write(const uint8_t *buffer, size_t size)
{
  size_t rval = size;

  if ( !Bluefruit.connected() ) {
    return rval;
  }

  /* Give priority to HM-10 output */
  if ( bleuart_HM10.notifyEnabled() && size > 0) {
    return bleuart_HM10.write(buffer, size);
  }

#if !defined(EXCLUDE_NUS)
  if ( bleuart_NUS.notifyEnabled() && size > 0) {
    rval = bleuart_NUS.write(buffer, size);
  }
#endif /* EXCLUDE_NUS */

  return rval;
}

IODev_ops_t nRF52_Bluetooth_ops = {
  "nRF52 Bluetooth",
  nRF52_Bluetooth_setup,
  nRF52_Bluetooth_loop,
  nRF52_Bluetooth_fini,
  nRF52_Bluetooth_available,
  nRF52_Bluetooth_read,
  nRF52_Bluetooth_write
};

#elif defined(ARDUINO_ARCH_RP2040)
#include "../system/SoC.h"
#if !defined(EXCLUDE_BLUETOOTH)

#include <queue>
#include <pico/cyw43_arch.h>
#include <CoreMutex.h>
#include <btstack.h>

#include <api/RingBuffer.h>

#include "EEPROM.h"
#include "WiFi.h"
#include "Bluetooth.h"
#include "Battery.h"

static bool _running = false;
static mutex_t _mutex;
static bool _overflow = false;
static volatile bool _connected = false;

static uint32_t _writer;
static uint32_t _reader;
static size_t   _fifoSize = 32;
static uint8_t *_queue = NULL;

static const int RFCOMM_SERVER_CHANNEL = 1;

static uint16_t _channelID;
static uint8_t  _spp_service_buffer[150];
static btstack_packet_callback_registration_t _hci_event_callback_registration;

static volatile int _writeLen = 0;
static const void *_writeBuff;

RingBufferN<BLE_FIFO_TX_SIZE> BLE_FIFO_TX = RingBufferN<BLE_FIFO_TX_SIZE>();
RingBufferN<BLE_FIFO_RX_SIZE> BLE_FIFO_RX = RingBufferN<BLE_FIFO_RX_SIZE>();

String BT_name = HOSTNAME;

/* ------- SPP BEGIN ------ */

static void hci_spp_packet_handler(uint8_t type, uint16_t channel, uint8_t *packet, uint16_t size) {
    UNUSED(channel);
    bd_addr_t event_addr;
    //uint8_t   rfcomm_channel_nr;
    //uint16_t  mtu;
    int i;

    switch (type) {
    case HCI_EVENT_PACKET:
        switch (hci_event_packet_get_type(packet)) {
        case HCI_EVENT_PIN_CODE_REQUEST:
            //Serial.printf("Pin code request - using '0000'\n");
            hci_event_pin_code_request_get_bd_addr(packet, event_addr);
            gap_pin_code_response(event_addr, "0000");
            break;

        case HCI_EVENT_USER_CONFIRMATION_REQUEST:
            // ssp: inform about user confirmation request
            //Serial.printf("SSP User Confirmation Request with numeric value '%06" PRIu32 "'\n", little_endian_read_32(packet, 8));
            //Serial.printf("SSP User Confirmation Auto accept\n");
            break;

        case RFCOMM_EVENT_INCOMING_CONNECTION:
            rfcomm_event_incoming_connection_get_bd_addr(packet, event_addr);
            //rfcomm_channel_nr = rfcomm_event_incoming_connection_get_server_channel(packet);
            _channelID = rfcomm_event_incoming_connection_get_rfcomm_cid(packet);
            //Serial.printf("RFCOMM channel %u requested for %s\n", rfcomm_channel_nr, bd_addr_to_str(event_addr));
            rfcomm_accept_connection(_channelID);
            break;

        case RFCOMM_EVENT_CHANNEL_OPENED:
            if (rfcomm_event_channel_opened_get_status(packet)) {
                //Serial.printf("RFCOMM channel open failed, status 0x%02x\n", rfcomm_event_channel_opened_get_status(packet));
            } else {
                _channelID = rfcomm_event_channel_opened_get_rfcomm_cid(packet);
                //mtu = rfcomm_event_channel_opened_get_max_frame_size(packet);
                //Serial.printf("RFCOMM channel open succeeded. New RFCOMM Channel ID %u, max frame size %u\n", rfcomm_channel_id, mtu);
                _connected = true;
            }
            break;
        case RFCOMM_EVENT_CAN_SEND_NOW:
            rfcomm_send(_channelID, (uint8_t *)_writeBuff, _writeLen);
            _writeLen = 0;
            break;
        case RFCOMM_EVENT_CHANNEL_CLOSED:
            //Serial.printf("RFCOMM channel closed\n");
            _channelID = 0;
            _connected = false;
            break;

        default:
            break;
        }
        break;

    case RFCOMM_DATA_PACKET:
        for (i = 0; i < size; i++) {
            auto next_writer = _writer + 1;
            if (next_writer == _fifoSize) {
                next_writer = 0;
            }
            if (next_writer != _reader) {
                _queue[_writer] = packet[i];
                asm volatile("" ::: "memory"); // Ensure the queue is written before the written count advances
                // Avoid using division or mod because the HW divider could be in use
                _writer = next_writer;
            } else {
                _overflow = true;
            }
        }
        break;

    default:
        break;
    }
}

/* ------- SPP END ------ */

/* ------- BLE BEGIN------ */

#define REPORT_INTERVAL_MS 3000
#define MAX_NR_CONNECTIONS 3
#define APP_AD_FLAGS       0x06

#define DEBUG_BLE          0

uint8_t *_advData   = nullptr;
uint8_t _advDataLen = 0;
uint8_t *_attdb     = nullptr;
size_t _attdbLen    = 0;

void _buildAdvData(const char *completeLocalName) {
    free(_advData);
    _advDataLen = 9 + strlen(completeLocalName);
    _advData = (uint8_t*) malloc(_advDataLen);
    int i = 0;
    // Flags general discoverable, BR/EDR not supported
    // 0x02, BLUETOOTH_DATA_TYPE_FLAGS, 0x06,
    _advData[i++] = 0x02;
    _advData[i++] = BLUETOOTH_DATA_TYPE_FLAGS;
    _advData[i++] = 0x06;
    // Name
    // 0x0d, BLUETOOTH_DATA_TYPE_COMPLETE_LOCAL_NAME,
    _advData[i++] = 1 + strlen(completeLocalName);
    _advData[i++] = BLUETOOTH_DATA_TYPE_COMPLETE_LOCAL_NAME;
    memcpy(_advData + i, completeLocalName, strlen(completeLocalName));
    i += strlen(completeLocalName);
    // 16-bit Service UUIDs
    _advData[i++] = 0x03;
    _advData[i++] = BLUETOOTH_DATA_TYPE_COMPLETE_LIST_OF_16_BIT_SERVICE_CLASS_UUIDS;
    _advData[i++] = 0xe0;
    _advData[i++] = 0xff;
}

static constexpr const uint8_t _attdb_head[] = {
    // ATT DB Version
    1,

    // 0x0001 PRIMARY_SERVICE-GAP_SERVICE
    0x0a, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x28, 0x00, 0x18,
    // 0x0002 CHARACTERISTIC-GAP_DEVICE_NAME - READ
    0x0d, 0x00, 0x02, 0x00, 0x02, 0x00, 0x03, 0x28, 0x02, 0x03, 0x00, 0x00, 0x2a,
};

static constexpr const uint8_t _attdb_tail[] =  {
    // Specification Type org.bluetooth.service.battery_service
    // https://www.bluetooth.com/api/gatt/xmlfile?xmlFileName=org.bluetooth.service.battery_service.xml
    // Battery Service 180F
    // 0x0004 PRIMARY_SERVICE-ORG_BLUETOOTH_SERVICE_BATTERY_SERVICE
    0x0a, 0x00, 0x02, 0x00, 0x04, 0x00, 0x00, 0x28, 0x0f, 0x18,
    // 0x0005 CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_BATTERY_LEVEL - DYNAMIC | READ | NOTIFY
    0x0d, 0x00, 0x02, 0x00, 0x05, 0x00, 0x03, 0x28, 0x12, 0x06, 0x00, 0x19, 0x2a,
    // 0x0006 VALUE CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_BATTERY_LEVEL - DYNAMIC | READ | NOTIFY
    // READ_ANYBODY
    0x08, 0x00, 0x02, 0x01, 0x06, 0x00, 0x19, 0x2a,
    // 0x0007 CLIENT_CHARACTERISTIC_CONFIGURATION
    // READ_ANYBODY, WRITE_ANYBODY
    0x0a, 0x00, 0x0e, 0x01, 0x07, 0x00, 0x02, 0x29, 0x00, 0x00,

    // Specification Type org.bluetooth.service.device_information
    // https://www.bluetooth.com/api/gatt/xmlfile?xmlFileName=org.bluetooth.service.device_information.xml
    // Device Information 180A
    // 0x0008 PRIMARY_SERVICE-ORG_BLUETOOTH_SERVICE_DEVICE_INFORMATION
    0x0a, 0x00, 0x02, 0x00, 0x08, 0x00, 0x00, 0x28, 0x0a, 0x18,
    // 0x0009 CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_MANUFACTURER_NAME_STRING - DYNAMIC | READ
    0x0d, 0x00, 0x02, 0x00, 0x09, 0x00, 0x03, 0x28, 0x02, 0x0a, 0x00, 0x29, 0x2a,
    // 0x000a VALUE CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_MANUFACTURER_NAME_STRING - DYNAMIC | READ
    // READ_ANYBODY
    0x08, 0x00, 0x02, 0x01, 0x0a, 0x00, 0x29, 0x2a,
    // 0x000b CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_MODEL_NUMBER_STRING - DYNAMIC | READ
    0x0d, 0x00, 0x02, 0x00, 0x0b, 0x00, 0x03, 0x28, 0x02, 0x0c, 0x00, 0x24, 0x2a,
    // 0x000c VALUE CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_MODEL_NUMBER_STRING - DYNAMIC | READ
    // READ_ANYBODY
    0x08, 0x00, 0x02, 0x01, 0x0c, 0x00, 0x24, 0x2a,
    // 0x000d CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_SERIAL_NUMBER_STRING - DYNAMIC | READ
    0x0d, 0x00, 0x02, 0x00, 0x0d, 0x00, 0x03, 0x28, 0x02, 0x0e, 0x00, 0x25, 0x2a,
    // 0x000e VALUE CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_SERIAL_NUMBER_STRING - DYNAMIC | READ
    // READ_ANYBODY
    0x08, 0x00, 0x02, 0x01, 0x0e, 0x00, 0x25, 0x2a,
    // 0x000f CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_HARDWARE_REVISION_STRING - DYNAMIC | READ
    0x0d, 0x00, 0x02, 0x00, 0x0f, 0x00, 0x03, 0x28, 0x02, 0x10, 0x00, 0x27, 0x2a,
    // 0x0010 VALUE CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_HARDWARE_REVISION_STRING - DYNAMIC | READ
    // READ_ANYBODY
    0x08, 0x00, 0x02, 0x01, 0x10, 0x00, 0x27, 0x2a,
    // 0x0011 CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_FIRMWARE_REVISION_STRING - DYNAMIC | READ
    0x0d, 0x00, 0x02, 0x00, 0x11, 0x00, 0x03, 0x28, 0x02, 0x12, 0x00, 0x26, 0x2a,
    // 0x0012 VALUE CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_FIRMWARE_REVISION_STRING - DYNAMIC | READ
    // READ_ANYBODY
    0x08, 0x00, 0x02, 0x01, 0x12, 0x00, 0x26, 0x2a,
    // 0x0013 CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_SOFTWARE_REVISION_STRING - DYNAMIC | READ
    0x0d, 0x00, 0x02, 0x00, 0x13, 0x00, 0x03, 0x28, 0x02, 0x14, 0x00, 0x28, 0x2a,
    // 0x0014 VALUE CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_SOFTWARE_REVISION_STRING - DYNAMIC | READ
    // READ_ANYBODY
    0x08, 0x00, 0x02, 0x01, 0x14, 0x00, 0x28, 0x2a,
    // 0x0015 CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_SYSTEM_ID - DYNAMIC | READ
    0x0d, 0x00, 0x02, 0x00, 0x15, 0x00, 0x03, 0x28, 0x02, 0x16, 0x00, 0x23, 0x2a,
    // 0x0016 VALUE CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_SYSTEM_ID - DYNAMIC | READ
    // READ_ANYBODY
    0x08, 0x00, 0x02, 0x01, 0x16, 0x00, 0x23, 0x2a,
    // 0x0017 CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_IEEE_11073_20601_REGULATORY_CERTIFICATION_DATA_LIST - DYNAMIC | READ
    0x0d, 0x00, 0x02, 0x00, 0x17, 0x00, 0x03, 0x28, 0x02, 0x18, 0x00, 0x2a, 0x2a,
    // 0x0018 VALUE CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_IEEE_11073_20601_REGULATORY_CERTIFICATION_DATA_LIST - DYNAMIC | READ
    // READ_ANYBODY
    0x08, 0x00, 0x02, 0x01, 0x18, 0x00, 0x2a, 0x2a,
    // 0x0019 CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_PNP_ID - DYNAMIC | READ
    0x0d, 0x00, 0x02, 0x00, 0x19, 0x00, 0x03, 0x28, 0x02, 0x1a, 0x00, 0x50, 0x2a,
    // 0x001a VALUE CHARACTERISTIC-ORG_BLUETOOTH_CHARACTERISTIC_PNP_ID - DYNAMIC | READ
    // READ_ANYBODY
    0x08, 0x00, 0x02, 0x01, 0x1a, 0x00, 0x50, 0x2a,

    // 0x001b PRIMARY_SERVICE-FFE0
    0x0a, 0x00, 0x02, 0x00, 0x1b, 0x00, 0x00, 0x28, 0xe0, 0xff,
    // 0x001c CHARACTERISTIC-FFE1 - READ | WRITE_WITHOUT_RESPONSE | NOTIFY | DYNAMIC
    0x0d, 0x00, 0x02, 0x00, 0x1c, 0x00, 0x03, 0x28, 0x16, 0x1d, 0x00, 0xe1, 0xff,
    // 0x001d VALUE CHARACTERISTIC-FFE1 - READ | WRITE_WITHOUT_RESPONSE | NOTIFY | DYNAMIC
    0x08, 0x00, 0x06, 0x01, 0x1d, 0x00, 0xe1, 0xff,
    // 0x001e CLIENT_CHARACTERISTIC_CONFIGURATION
    // READ_ANYBODY, WRITE_ANYBODY
    0x0a, 0x00, 0x0e, 0x01, 0x1e, 0x00, 0x02, 0x29, 0x00, 0x00,
    // 0x001f USER_DESCRIPTION-READ-HMSoft
    // READ_ANYBODY, WRITE_ANYBODY
    0x08, 0x00, 0x0a, 0x01, 0x1f, 0x00, 0x01, 0x29,
    // END
    0x00, 0x00,
};

void _buildAttdb(const char *Name) {
    free(_attdb);
    _attdbLen = sizeof(_attdb_head) + 8 + strlen(Name) + sizeof(_attdb_tail);
    _attdb = (uint8_t *) malloc(_attdbLen);
    memcpy(_attdb, _attdb_head, sizeof(_attdb_head));
    // 0x0003 VALUE CHARACTERISTIC-GAP_DEVICE_NAME - READ
    // READ_ANYBODY
    // 0x11, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x2a, 0x48, 0x49, 0x44, 0x20, 0x4d, 0x6f, 0x75, 0x73, 0x65,
    int i = sizeof(_attdb_head);
    _attdb[i++] = 8 + strlen(Name);
    _attdb[i++] = 0x00;
    _attdb[i++] = 0x02;
    _attdb[i++] = 0x00;
    _attdb[i++] = 0x03;
    _attdb[i++] = 0x00;
    _attdb[i++] = 0x00;
    _attdb[i++] = 0x2a;
    memcpy(_attdb + i, Name, strlen(Name));
    i += strlen(Name);
    memcpy(_attdb + i, _attdb_tail, sizeof(_attdb_tail));
}

// support for multiple clients
typedef struct {
    char name;
    int le_notification_enabled;
    uint16_t value_handle;
    hci_con_handle_t connection_handle;
    int  counter;
    char test_data[200];
    int  test_data_len;
    uint32_t test_data_sent;
    uint32_t test_data_start;
} le_streamer_connection_t;

static le_streamer_connection_t le_streamer_connections[MAX_NR_CONNECTIONS];

// round robin sending
static int connection_index;

static void init_connections(void){
    // track connections
    int i;
    for (i=0;i<MAX_NR_CONNECTIONS;i++){
        le_streamer_connections[i].connection_handle = HCI_CON_HANDLE_INVALID;
        le_streamer_connections[i].name = 'A' + i;
    }
}

static le_streamer_connection_t * connection_for_conn_handle(hci_con_handle_t conn_handle){
    int i;
    for (i=0;i<MAX_NR_CONNECTIONS;i++){
        if (le_streamer_connections[i].connection_handle == conn_handle) return &le_streamer_connections[i];
    }
    return NULL;
}

static void next_connection_index(void){
    connection_index++;
    if (connection_index == MAX_NR_CONNECTIONS){
        connection_index = 0;
    }
}

static void test_reset(le_streamer_connection_t * context){
    context->test_data_start = btstack_run_loop_get_time_ms();
    context->test_data_sent = 0;
}

static void test_track_sent(le_streamer_connection_t * context, int bytes_sent){
    context->test_data_sent += bytes_sent;
    // evaluate
    uint32_t now = btstack_run_loop_get_time_ms();
    uint32_t time_passed = now - context->test_data_start;
    if (time_passed < REPORT_INTERVAL_MS) return;
    // print speed
    int bytes_per_second = context->test_data_sent * 1000 / time_passed;
#if DEBUG_BLE
    Serial.printf("%c: %"PRIu32" bytes sent-> %u.%03u kB/s\r\n", context->name, context->test_data_sent, bytes_per_second / 1000, bytes_per_second % 1000);
#endif /* DEBUG_BLE */

    // restart
    context->test_data_start = now;
    context->test_data_sent  = 0;
}

static void streamer(void){

    // find next active streaming connection
    int old_connection_index = connection_index;
    while (true) {
        // active found?
        if ((le_streamer_connections[connection_index].connection_handle != HCI_CON_HANDLE_INVALID) &&
            (le_streamer_connections[connection_index].le_notification_enabled)) break;

        // check next
        next_connection_index();

        // none found
        if (connection_index == old_connection_index) return;
    }

    le_streamer_connection_t * context = &le_streamer_connections[connection_index];

    size_t size = BLE_FIFO_TX.available();
    size = size < context->test_data_len ? size : context->test_data_len;
    size = size < BLE_MAX_WRITE_CHUNK_SIZE ? size : BLE_MAX_WRITE_CHUNK_SIZE;

    if (size > 0) {
      for (int i=0; i < size; i++) {
        context->test_data[i] = BLE_FIFO_TX.read_char();
      }

      // send
      att_server_notify(context->connection_handle, context->value_handle, (uint8_t*) context->test_data, size);

      // request next send event
      att_server_request_can_send_now_event(context->connection_handle);
    }

    // track
    test_track_sent(context, size);

    // check next
    next_connection_index();
}

static void hci_le_packet_handler (uint8_t packet_type, uint16_t channel, uint8_t *packet, uint16_t size){
    UNUSED(channel);
    UNUSED(size);

    if (packet_type != HCI_EVENT_PACKET) return;

    uint16_t conn_interval;
    hci_con_handle_t con_handle;
    static const char * const phy_names[] = {
        "1 M", "2 M", "Codec"
    };

    switch (hci_event_packet_get_type(packet)) {
        case BTSTACK_EVENT_STATE:
            // BTstack activated, get started
            if (btstack_event_state_get_state(packet) == HCI_STATE_WORKING) {
                //Serial.printf("To start the streaming, please run the le_streamer_client example on other device, or use some GATT Explorer, e.g. LightBlue, BLExplr.\r\n");
            }
            break;
        case HCI_EVENT_DISCONNECTION_COMPLETE:
            con_handle = hci_event_disconnection_complete_get_connection_handle(packet);
#if DEBUG_BLE
            Serial.printf("- LE Connection 0x%04x: disconnect, reason %02x\r\n", con_handle, hci_event_disconnection_complete_get_reason(packet));
#endif /* DEBUG_BLE */
            break;
        case HCI_EVENT_LE_META:
            switch (hci_event_le_meta_get_subevent_code(packet)) {
                case HCI_SUBEVENT_LE_CONNECTION_COMPLETE:
                    // print connection parameters (without using float operations)
                    con_handle    = hci_subevent_le_connection_complete_get_connection_handle(packet);
                    conn_interval = hci_subevent_le_connection_complete_get_conn_interval(packet);
#if DEBUG_BLE
                    Serial.printf("- LE Connection 0x%04x: connected - connection interval %u.%02u ms, latency %u\r\n", con_handle, conn_interval * 125 / 100,
                        25 * (conn_interval & 3), hci_subevent_le_connection_complete_get_conn_latency(packet));

                    // request min con interval 15 ms for iOS 11+
                    Serial.printf("- LE Connection 0x%04x: request 15 ms connection interval\r\n", con_handle);
#endif /* DEBUG_BLE */
                    gap_request_connection_parameter_update(con_handle, 12, 12, 0, 0x0048);
                    break;
                case HCI_SUBEVENT_LE_CONNECTION_UPDATE_COMPLETE:
                    // print connection parameters (without using float operations)
                    con_handle    = hci_subevent_le_connection_update_complete_get_connection_handle(packet);
                    conn_interval = hci_subevent_le_connection_update_complete_get_conn_interval(packet);
#if DEBUG_BLE
                    Serial.printf("- LE Connection 0x%04x: connection update - connection interval %u.%02u ms, latency %u\r\n", con_handle, conn_interval * 125 / 100,
                        25 * (conn_interval & 3), hci_subevent_le_connection_update_complete_get_conn_latency(packet));
#endif /* DEBUG_BLE */
                    break;
                case HCI_SUBEVENT_LE_DATA_LENGTH_CHANGE:
                    con_handle = hci_subevent_le_data_length_change_get_connection_handle(packet);
#if DEBUG_BLE
                    Serial.printf("- LE Connection 0x%04x: data length change - max %u bytes per packet\r\n", con_handle,
                                  hci_subevent_le_data_length_change_get_max_tx_octets(packet));
#endif /* DEBUG_BLE */
                    break;
                case HCI_SUBEVENT_LE_PHY_UPDATE_COMPLETE:
                    con_handle = hci_subevent_le_phy_update_complete_get_connection_handle(packet);
                    Serial.printf("- LE Connection 0x%04x: PHY update - using LE %s PHY now\r\n", con_handle,
                                  phy_names[hci_subevent_le_phy_update_complete_get_tx_phy(packet)]);
                    break;
                default:
                    break;
            }
            break;

        default:
            break;
    }
}

static void att_packet_handler (uint8_t packet_type, uint16_t channel, uint8_t *packet, uint16_t size){
    UNUSED(channel);
    UNUSED(size);

    int mtu;
    le_streamer_connection_t * context;
    switch (packet_type) {
        case HCI_EVENT_PACKET:
            switch (hci_event_packet_get_type(packet)) {
                case ATT_EVENT_CONNECTED:
                    // setup new
                    context = connection_for_conn_handle(HCI_CON_HANDLE_INVALID);
                    if (!context) break;
                    context->counter = 'A';
                    context->connection_handle = att_event_connected_get_handle(packet);
                    context->test_data_len = btstack_min(att_server_get_mtu(context->connection_handle) - 3, sizeof(context->test_data));
#if DEBUG_BLE
                    Serial.printf("%c: ATT connected, handle %04x, test data len %u\r\n", context->name, context->connection_handle, context->test_data_len);
#endif /* DEBUG_BLE */
                    break;
                case ATT_EVENT_MTU_EXCHANGE_COMPLETE:
                    mtu = att_event_mtu_exchange_complete_get_MTU(packet) - 3;
                    context = connection_for_conn_handle(att_event_mtu_exchange_complete_get_handle(packet));
                    if (!context) break;
                    context->test_data_len = btstack_min(mtu - 3, sizeof(context->test_data));
#if DEBUG_BLE
                    Serial.printf("%c: ATT MTU = %u => use test data of len %u\r\n", context->name, mtu, context->test_data_len);
#endif /* DEBUG_BLE */
                    break;
                case ATT_EVENT_CAN_SEND_NOW:
                    streamer();
                    break;
                case ATT_EVENT_DISCONNECTED:
                    context = connection_for_conn_handle(att_event_disconnected_get_handle(packet));
                    if (!context) break;
                    // free connection
#if DEBUG_BLE
                    Serial.printf("%c: ATT disconnected, handle %04x\r\n", context->name, context->connection_handle);
#endif /* DEBUG_BLE */
                    context->le_notification_enabled = 0;
                    context->connection_handle = HCI_CON_HANDLE_INVALID;
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }
}

static int att_write_callback(hci_con_handle_t con_handle, uint16_t att_handle, uint16_t transaction_mode, uint16_t offset, uint8_t *buffer, uint16_t buffer_size){
    UNUSED(offset);

#if DEBUG_BLE
    Serial.printf("att_write_callback att_handle %04x, transaction mode %u size %u offset %u\r\n", att_handle, transaction_mode, buffer_size, offset);
#endif /* DEBUG_BLE */
    if (transaction_mode != ATT_TRANSACTION_MODE_NONE) return 0;
    le_streamer_connection_t * context = connection_for_conn_handle(con_handle);
    switch(att_handle){
        case ATT_CHARACTERISTIC_FFE1_01_CLIENT_CONFIGURATION_HANDLE:
            context->le_notification_enabled = little_endian_read_16(buffer, 0) == GATT_CLIENT_CHARACTERISTICS_CONFIGURATION_NOTIFICATION;
            //Serial.printf("%c: Notifications enabled %u\r\n", context->name, context->le_notification_enabled);
            if (context->le_notification_enabled){
                switch (att_handle){
                    case ATT_CHARACTERISTIC_FFE1_01_CLIENT_CONFIGURATION_HANDLE:
                        context->value_handle = ATT_CHARACTERISTIC_FFE1_01_VALUE_HANDLE;
                        break;
                    default:
                        break;
                }
                att_server_request_can_send_now_event(context->connection_handle);
            }
            test_reset(context);
            break;
        case ATT_CHARACTERISTIC_FFE1_01_VALUE_HANDLE:
#if DEBUG_BLE
            Serial.printf("Write to 0x%04x, len %u offset %u\r\n", att_handle, buffer_size, offset);
#endif /* DEBUG_BLE */
            if (buffer_size > 0 && offset == 0) {
              size_t size = BLE_FIFO_RX.availableForStore();
              size = (size < buffer_size) ? size : buffer_size;
              for (size_t i = 0; i < size; i++) {
                BLE_FIFO_RX.store_char(buffer[i]);
              }
            }
            break;
        default:
            Serial.printf("Write to 0x%04x, len %u\r\n", att_handle, buffer_size);
            break;
    }
    return 0;
}

#define ATT_VALUE_MAX_LEN  50
#define ATT_NUM_ATTRIBUTES 10

typedef struct {
    uint16_t handle;
    uint16_t len;
    uint8_t  value[ATT_VALUE_MAX_LEN];
} attribute_t;

static attribute_t att_attributes[ATT_NUM_ATTRIBUTES];

// handle == 0 finds free attribute
static int att_attribute_for_handle(uint16_t aHandle){
    int i;
    for (i=0;i<ATT_NUM_ATTRIBUTES;i++){
        if (att_attributes[i].handle == aHandle) {
            return i;
        }
    }
    return -1;
}

static void att_setup_attribute(uint16_t attribute_handle, const uint8_t * value, uint16_t len){
    int index = att_attribute_for_handle(attribute_handle);
    if (index < 0){
        index = att_attribute_for_handle(0);
    }
#if DEBUG_BLE
    Serial.printf("Setup Attribute %04x, len %u, value: %s\r\n", attribute_handle, len, value);
#endif /* DEBUG_BLE */
    att_attributes[index].handle = attribute_handle;
    att_attributes[index].len    = len;
    memcpy(att_attributes[index].value, value, len);
}

static void att_attributes_init(void){
    int i;
    for (i=0;i<ATT_NUM_ATTRIBUTES;i++){
        att_attributes[i].handle = 0;
    }

    // preset some attributes
    att_setup_attribute(ATT_CHARACTERISTIC_FFE1_01_USER_DESCRIPTION_HANDLE,
                        (const uint8_t *) "HMSoft", strlen("HMSoft"));
}

static unsigned long BLE_Aux_Tx_TimeMarker = 0;

/* ------- BLE END ------ */

static void lockBluetooth() {
    async_context_acquire_lock_blocking(cyw43_arch_async_context());
}

static void unlockBluetooth() {
    async_context_release_lock(cyw43_arch_async_context());
}

static void CYW43_Bluetooth_setup()
{
  if (_running) return;

  BT_name += "-";
  BT_name += String(SoC->getChipId() & 0x00FFFFFFU, HEX);

  switch (settings->bluetooth)
  {
  case BLUETOOTH_SPP:
    {
      mutex_init(&_mutex);
      _overflow = false;

      _queue = new uint8_t[_fifoSize];
      _writer = 0;
      _reader = 0;

      // register for HCI events
      _hci_event_callback_registration.callback = &hci_spp_packet_handler;
      hci_add_event_handler(&_hci_event_callback_registration);

      l2cap_init();

#ifdef ENABLE_BLE
      // Initialize LE Security Manager. Needed for cross-transport key derivation
      sm_init();
#endif

      rfcomm_init();
      rfcomm_register_service(hci_spp_packet_handler, RFCOMM_SERVER_CHANNEL, 0xffff);  // reserved channel, mtu limited by l2cap

      // init SDP, create record for SPP and register with SDP
      sdp_init();
      bzero(_spp_service_buffer, sizeof(_spp_service_buffer));
      spp_create_sdp_record(_spp_service_buffer, 0x10001, RFCOMM_SERVER_CHANNEL, "CYW43_SPP");
      sdp_register_service(_spp_service_buffer);

      gap_discoverable_control(1);
      gap_ssp_set_io_capability(SSP_IO_CAPABILITY_DISPLAY_YES_NO);
      gap_set_local_name(BT_name.c_str());

      hci_power_control(HCI_POWER_ON);

      _running = true;
    }
    break;
  case BLUETOOTH_LE_HM10_SERIAL:
    {
      mutex_init(&_mutex);

      BT_name += "-LE";
      _buildAdvData(BT_name.c_str());
      _buildAttdb(BT_name.c_str());

      l2cap_init();
//      l2cap_set_max_le_mtu(26);

      // setup SM: Display only
      sm_init();

      // setup ATT server
      att_server_init(_attdb, NULL, att_write_callback);

      // Setup battery service
      battery_service_server_init((uint8_t) Battery_charge());

      static char SerialNum[9];
      snprintf(SerialNum, sizeof(SerialNum), "%08X", SoC->getChipId());
      static char Hardware[9];
      snprintf(Hardware,  sizeof(Hardware),  "%08X", hw_info.revision);

      const char *Firmware = "Arduino RP2040 " ARDUINO_PICO_VERSION_STR;

      // Setup device information service
      device_information_service_server_init();
      device_information_service_server_set_manufacturer_name(RP2040_Device_Manufacturer);
      device_information_service_server_set_model_number(RP2040_Device_Model);
      device_information_service_server_set_serial_number(SerialNum);
      device_information_service_server_set_hardware_revision(Hardware);
      device_information_service_server_set_firmware_revision(Firmware);
      device_information_service_server_set_software_revision(SOFTRF_FIRMWARE_VERSION);

      // register for HCI events
      _hci_event_callback_registration.callback = &hci_le_packet_handler;
      hci_add_event_handler(&_hci_event_callback_registration);

      att_attributes_init();

      // register for ATT events
      att_server_register_packet_handler(att_packet_handler);

      // setup advertisements
      uint16_t adv_int_min = 0x0030;
      uint16_t adv_int_max = 0x0030;
      uint8_t adv_type = 0;
      bd_addr_t null_addr;
      memset(null_addr, 0, 6);
      gap_advertisements_set_params(adv_int_min, adv_int_max, adv_type, 0, null_addr, 0x07, 0x00);
      gap_advertisements_set_data(_advDataLen, _advData);
      gap_advertisements_enable(1);

      // init client state
      init_connections();

      hci_power_control(HCI_POWER_ON);

       BLE_Aux_Tx_TimeMarker = millis();
      _running = true;
    }
    break;
  case BLUETOOTH_A2DP_SOURCE:
  case BLUETOOTH_NONE:
  default:
    break;
  }
}

static void CYW43_Bluetooth_loop()
{
  switch (settings->bluetooth)
  {
  case BLUETOOTH_LE_HM10_SERIAL:
    if (_running && (millis() - BLE_Aux_Tx_TimeMarker > 100)) {
      if (BLE_FIFO_TX.available() > 0) {
        streamer();
      }
      BLE_Aux_Tx_TimeMarker = millis();
    }

    if (isTimeToBattery()) {
      battery_service_server_set_battery_value((uint8_t) Battery_charge());
    }
    break;
  case BLUETOOTH_NONE:
  case BLUETOOTH_SPP:
  case BLUETOOTH_A2DP_SOURCE:
  default:
    break;
  }
}

static void CYW43_Bluetooth_fini()
{
  switch (settings->bluetooth)
  {
  case BLUETOOTH_SPP:
  case BLUETOOTH_LE_HM10_SERIAL:
    {
      if (!_running) {
          return;
      }
      _running = false;

      hci_power_control(HCI_POWER_OFF);
      lockBluetooth();
      if (_queue != NULL) delete[] _queue;
      unlockBluetooth();
    }
    break;
  case BLUETOOTH_A2DP_SOURCE:
  case BLUETOOTH_NONE:
  default:
    break;
  }
}

static int CYW43_Bluetooth_available()
{
  int rval = 0;

  switch (settings->bluetooth)
  {
  case BLUETOOTH_SPP:
    {
      CoreMutex m(&_mutex);
      if (_running && m) {
        rval = (_fifoSize + _writer - _reader) % _fifoSize;
      }
    }
    break;
  case BLUETOOTH_LE_HM10_SERIAL:
    rval = BLE_FIFO_RX.available();
    break;
  case BLUETOOTH_NONE:
  case BLUETOOTH_A2DP_SOURCE:
  default:
    break;
  }

  return rval;
}

static int CYW43_Bluetooth_read()
{
  int rval = -1;

  switch (settings->bluetooth)
  {
  case BLUETOOTH_SPP:
    {
      CoreMutex m(&_mutex);
      if (_running && m && _writer != _reader) {
          auto ret = _queue[_reader];
          asm volatile("" ::: "memory"); // Ensure the value is read before advancing
          auto next_reader = (_reader + 1) % _fifoSize;
          asm volatile("" ::: "memory"); // Ensure the reader value is only written once, correctly
          _reader = next_reader;
          rval = ret;
      }
    }
    break;
  case BLUETOOTH_LE_HM10_SERIAL:
    rval = BLE_FIFO_RX.read_char();
    break;
  case BLUETOOTH_NONE:
  case BLUETOOTH_A2DP_SOURCE:
  default:
    break;
  }

  return rval;
}

static size_t CYW43_Bluetooth_write(const uint8_t *buffer, size_t size)
{
  size_t rval = size;

  switch (settings->bluetooth)
  {
  case BLUETOOTH_SPP:
    {
      CoreMutex m(&_mutex);
      if (!_running || !m || !_connected || !size)  {
          return 0;
      }
      _writeBuff = buffer;
      _writeLen = size;
      lockBluetooth();
      rfcomm_request_can_send_now_event(_channelID);
      unlockBluetooth();
      while (_connected && _writeLen) {
          /* noop busy wait */
      }
    }
    break;
  case BLUETOOTH_LE_HM10_SERIAL:
    {
      size_t avail = BLE_FIFO_TX.availableForStore();
      if (size > avail) {
        rval = avail;
      }
      for (size_t i = 0; i < rval; i++) {
        BLE_FIFO_TX.store_char(buffer[i]);
      }
    }
    break;
  case BLUETOOTH_NONE:
  case BLUETOOTH_A2DP_SOURCE:
  default:
    break;
  }

  return rval;
}

IODev_ops_t CYW43_Bluetooth_ops = {
  "CYW43 Bluetooth",
  CYW43_Bluetooth_setup,
  CYW43_Bluetooth_loop,
  CYW43_Bluetooth_fini,
  CYW43_Bluetooth_available,
  CYW43_Bluetooth_read,
  CYW43_Bluetooth_write
};
#endif /* EXCLUDE_BLUETOOTH */
#endif /* ESP32 or ARDUINO_ARCH_NRF52 or ARDUINO_ARCH_RP2040 */
