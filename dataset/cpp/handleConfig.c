/*
 * BitMeterOS
 * http://codebox.org.uk/bitmeterOS
 *
 * Copyright (c) 2011 Rob Dawson
 *
 * Licensed under the GNU General Public License
 * http://www.gnu.org/licenses/gpl.txt
 *
 * This file is part of BitMeterOS.
 *
 * BitMeterOS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BitMeterOS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with BitMeterOS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <stdio.h>
#include "bmws.h"
#include "client.h"
#include "common.h"

#define HOST_ADAPTER_SQL "SELECT hs AS hs, ad AS ad FROM data GROUP BY hs, ad"
#define WEB_SERVER_NAME_MAX_LEN 32
#define RSS_HOST_NAME_MAX_LEN   32
#define BAD_NUM -1
#define RSS_ITEMS_MIN  1
#define RSS_ITEMS_MAX 20
#define RSS_FREQ_MIN   1
#define RSS_FREQ_MAX   2
#define COLOUR_LEN     6
#define MONITOR_INTERVAL_MIN 1000
#define MONITOR_INTERVAL_MAX 30000
#define HISTORY_INTERVAL_MIN 5000
#define HISTORY_INTERVAL_MAX 60000
#define SUMMARY_INTERVAL_MIN 1000
#define SUMMARY_INTERVAL_MAX 60000

/*
Handles '/config' requests received by the web server.
*/

static void writeConfig(SOCKET fd, int allowAdmin);
static void writeNumConfigValue(SOCKET fd, char* key, char* value);
static void writeNumConfigNumValue(SOCKET fd, char* key, int value);
static void writeTxtConfigValue(SOCKET fd, char* key, char* value);
static void writeHostAdapterList(SOCKET fd);
static int updateWebServerName(char* value);
static int updateRssHostName(char* value);
static int updateRssFreq(char* value);
static int updateRssItems(char* value);
static int updateDlColour(char* value);
static int updateUlColour(char* value);
static int updateMonitorInterval(char* value);
static int updateHistoryInterval(char* value);
static int updateSummaryInterval(char* value);

void processConfigRequest(SOCKET fd, struct Request* req, int allowAdmin){
	struct NameValuePair* params = req->params;

	if (params == NULL){
	 // If there are no request parameters then we send back the current config as a JSON object
		writeConfig(fd, allowAdmin);

	} else {
	 // If there are parameters then this is a config-update request
	 	if (allowAdmin){
	 		int status = SUCCESS;
	 		while(params != NULL){
	 		 // Check which config value we are updating
	 			if (strcmp(CONFIG_WEB_SERVER_NAME, params->name) == 0){
	 				status = updateWebServerName(params->value);

	 			} else if (strcmp(CONFIG_WEB_RSS_HOST, params->name) == 0){
	 				status = updateRssHostName(params->value);

	 			} else if (strcmp(CONFIG_WEB_MONITOR_INTERVAL, params->name) == 0){
	 				status = updateMonitorInterval(params->value);

	 			} else if (strcmp(CONFIG_WEB_HISTORY_INTERVAL, params->name) == 0){
	 				status = updateHistoryInterval(params->value);

	 			} else if (strcmp(CONFIG_WEB_SUMMARY_INTERVAL, params->name) == 0){
	 				status = updateSummaryInterval(params->value);

	 			} else if (strcmp(CONFIG_WEB_RSS_FREQ, params->name) == 0){
	 				status = updateRssFreq(params->value);

	 			} else if (strcmp(CONFIG_WEB_RSS_ITEMS, params->name) == 0){
	 				status = updateRssItems(params->value);

	 			} else if (strcmp(CONFIG_WEB_COLOUR_DL, params->name) == 0){
	 				status = updateDlColour(params->value);

	 			} else if (strcmp(CONFIG_WEB_COLOUR_UL, params->name) == 0){
	 				status = updateUlColour(params->value);

	 			} else {
	 				if (strcmp("_", params->name) == 0){
	 				 // This parameter gets tagged onto every AJAX request from jQuery to prevent caching issues in IE

	 				} else {
		 			 // Not all configs can be updated via the web
		 			 	logMsg(LOG_ERR, "Update request for illegal/unknown config param: %s=%s", params->name, params->value);
		 				status = FAIL;
		 			}
	 			}

	 			if (status == FAIL){
	 				break;
	 			}
	 			params = params->next;
	 		}

 			if (status == SUCCESS){
 				writeHeadersOk(fd, MIME_JSON, TRUE);
 				writeText(fd, "{}");
 			} else {
 				writeHeadersServerError(fd, "Config update failed %s=%s", params->name, params->value); 
 			}

	 	} else {
	 	 // Config updates are an administrative operation
	 		writeHeadersForbidden(fd, "config update");
	 	}
	}
}

static int isValueLenOk(char* value, int maxlen){
	if (strlen(value) > maxlen) {
		logMsg(LOG_ERR, "Attempt to change config value to %s - value too long (maxlen=%d)", value, maxlen);
		return FALSE;

	} else {
		return TRUE;
	}
}
static int isNumericOk(char* value, int minValue, int maxValue){
	long longVal = strToLong(value, BAD_NUM);
	if ((longVal == BAD_NUM) || (longVal < minValue) || (longVal > maxValue)) {
		logMsg(LOG_ERR, "Attempt to update numeric config with invalid/out-of-range value of %s (min=%d max=%d)",
			value, minValue, maxValue);
		return FALSE;

	} else {
		return TRUE;
	}
}
static int isColourOk(char* value){
	if (strlen(value) != COLOUR_LEN) {
		logMsg(LOG_ERR, "Attempt to change config colour value to %s - length must be %d", value, COLOUR_LEN);
		return FALSE;

	} else {
		int i;
		char c; //TODO use regex
		for(i=0; i<COLOUR_LEN; i++){
			c = value[i];
			if (!((c>='0' && c<='9') || (c>='a' && c<='f'))){
				logMsg(LOG_ERR, "Attempt to change config colour value to %s - characters must be 0-9 or a-f.", value);
				return FALSE;
			}
		}
	}
	return TRUE;
}
static int noDodgyChars(char* value){
	if ((strchr(value, '<') != NULL) || (strchr(value, '>') != NULL)) {
		logMsg(LOG_ERR, "Suspicious characters detected in config value %s", value);
		return FALSE;

	} else {
		return TRUE;
	}
}

static int updateMonitorInterval(char* value){
	if (isNumericOk(value, MONITOR_INTERVAL_MIN, MONITOR_INTERVAL_MAX)) {
		setConfigTextValue(CONFIG_WEB_MONITOR_INTERVAL, value);
		return SUCCESS;

	} else {
		return FAIL;
	}
}
static int updateHistoryInterval(char* value){
	if (isNumericOk(value, HISTORY_INTERVAL_MIN, HISTORY_INTERVAL_MAX)) {
		setConfigTextValue(CONFIG_WEB_HISTORY_INTERVAL, value);
		return SUCCESS;

	} else {
		return FAIL;
	}
}
static int updateSummaryInterval(char* value){
	if (isNumericOk(value, SUMMARY_INTERVAL_MIN, SUMMARY_INTERVAL_MAX)) {
		setConfigTextValue(CONFIG_WEB_SUMMARY_INTERVAL, value);
		return SUCCESS;

	} else {
		return FAIL;
	}
}

static int updateColour(char* value, char* configName){
	if (isColourOk(value)) {
		char colTxt[7];
		sprintf(colTxt, "#%s", value);
		setConfigTextValue(configName, colTxt);
		return SUCCESS;

	} else {
		return FAIL;
	}
}

static int updateDlColour(char* value){
	return updateColour(value, CONFIG_WEB_COLOUR_DL);
}

static int updateUlColour(char* value){
	return updateColour(value, CONFIG_WEB_COLOUR_UL);
}

static int updateRssItems(char* value){
	if (isNumericOk(value, RSS_ITEMS_MIN, RSS_ITEMS_MAX)) {
		setConfigTextValue(CONFIG_WEB_RSS_ITEMS, value);
		return SUCCESS;

	} else {
		return FAIL;
	}
}

static int updateRssFreq(char* value){
	if (isNumericOk(value, RSS_FREQ_MIN, RSS_FREQ_MAX)) {
		setConfigTextValue(CONFIG_WEB_RSS_FREQ, value);
		return SUCCESS;

	} else {
		return FAIL;
	}
}

static int updateRssHostName(char* value){
	if (isValueLenOk(value, RSS_HOST_NAME_MAX_LEN) && noDodgyChars(value)) {
		setConfigTextValue(CONFIG_WEB_RSS_HOST, value);
		return SUCCESS;

	} else {
		return FAIL;
	}
}

static int updateWebServerName(char* value){
	if (isValueLenOk(value, WEB_SERVER_NAME_MAX_LEN) && noDodgyChars(value)) {
		setConfigTextValue(CONFIG_WEB_SERVER_NAME, value);
		return SUCCESS;

	} else {
		return FAIL;
	}
}

static void writeConfig(SOCKET fd, int allowAdmin){
 // Write the JSON object out to the stream
    writeHeadersOk(fd, MIME_JS, TRUE);

	writeText(fd, "var config = { ");
	char* val = getConfigText(CONFIG_WEB_MONITOR_INTERVAL, FALSE);
    writeNumConfigValue(fd, "monitorInterval", val);
    free(val);

    writeText(fd, ", ");
    val = getConfigText(CONFIG_WEB_SUMMARY_INTERVAL, FALSE);
    writeNumConfigValue(fd, "summaryInterval", val);
    free(val);

    writeText(fd, ", ");
    val = getConfigText(CONFIG_WEB_HISTORY_INTERVAL, FALSE);
    writeNumConfigValue(fd, "historyInterval", val);
    free(val);

    writeText(fd, ", ");
    writeNumConfigNumValue(fd, "monitorIntervalMin", MONITOR_INTERVAL_MIN);
    writeText(fd, ", ");
    writeNumConfigNumValue(fd, "monitorIntervalMax", MONITOR_INTERVAL_MAX);
    writeText(fd, ", ");
    writeNumConfigNumValue(fd, "historyIntervalMin", HISTORY_INTERVAL_MIN);
    writeText(fd, ", ");
    writeNumConfigNumValue(fd, "historyIntervalMax", HISTORY_INTERVAL_MAX);
    writeText(fd, ", ");
    writeNumConfigNumValue(fd, "summaryIntervalMin", SUMMARY_INTERVAL_MIN);
    writeText(fd, ", ");
    writeNumConfigNumValue(fd, "summaryIntervalMax", SUMMARY_INTERVAL_MAX);

    writeText(fd, ", ");
    val = getConfigText(CONFIG_WEB_SERVER_NAME, FALSE);
    writeTxtConfigValue(fd, "serverName", val);
    free(val);

    writeText(fd, ", ");
    val = getConfigText(CONFIG_WEB_COLOUR_DL, FALSE);
    writeTxtConfigValue(fd, "dlColour", val);
    free(val);

    writeText(fd, ", ");
    val = getConfigText(CONFIG_WEB_COLOUR_UL, FALSE);
    writeTxtConfigValue(fd, "ulColour", val);
    free(val);

    writeText(fd, ", ");
    writeNumConfigValue(fd, "allowAdmin", allowAdmin ? "1" : "0");

    writeText(fd, ", ");
    writeTxtConfigValue(fd, "version", VERSION);

    writeText(fd, ", ");
    val = getConfigText(CONFIG_WEB_RSS_ITEMS, FALSE);
    writeNumConfigValue(fd, "rssItems", val);

    writeText(fd, ", ");
    val = getConfigText(CONFIG_WEB_RSS_FREQ, FALSE);
    writeNumConfigValue(fd, "rssFreq", val);

    writeText(fd, ", ");
    val = getConfigText(CONFIG_WEB_RSS_HOST, FALSE);
    writeTxtConfigValue(fd, "rssHost", val);

    writeText(fd, ", ");
    writeHostAdapterList(fd);

	writeText(fd, " };");
}

static void writeNumConfigValue(SOCKET fd, char* key, char* value){
 // Helper function, writes a key/value pair to the stream
    char txt[64];
    sprintf(txt, "\"%s\" : %s", key, value);
    writeText(fd, txt);
}

static void writeNumConfigNumValue(SOCKET fd, char* key, int value){
 // Helper function, writes a key/value pair to the stream
    char txt[64];
    sprintf(txt, "\"%s\" : %d", key, value);
    writeText(fd, txt);
}

static void writeTxtConfigValue(SOCKET fd, char* key, char* value){
 // Helper function, writes a key/value pair to the stream surrounding the value with quotes
    char txt[64];
    sprintf(txt, "\"%s\" : \"%s\"", key, value);
    writeText(fd, txt);
}

static void writeHostAdapterList(SOCKET fd){
    sqlite3_stmt* stmt;

    prepareSql(&stmt, HOST_ADAPTER_SQL);
    struct Data* result = runSelect(stmt);
    struct Data* currentResult = result;

    writeText(fd, "\"adapters\" : [");
    int firstResult = TRUE;
    while(currentResult != NULL){
        if (firstResult == FALSE){
            writeText(fd, ",");
        }

        writeText(fd, "{");
        if (currentResult->hs == NULL || strcmp("", currentResult->hs) == 0){
            writeTxtConfigValue(fd, "hs", "local");
        } else {
            writeTxtConfigValue(fd, "hs", currentResult->hs);
        }
        writeText(fd, ",");
        writeTxtConfigValue(fd, "ad", currentResult->ad);
        writeText(fd, "}");

        firstResult = FALSE;
        currentResult = currentResult->next;
    }
    writeText(fd, "]");

    sqlite3_finalize(stmt);
    freeData(result);
}
