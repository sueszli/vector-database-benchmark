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

#ifdef _WIN32
	#define __USE_MINGW_ANSI_STDIO 1
#endif
#include <stdlib.h>
#include "sqlite3.h"
#include "bmws.h"
#include "common.h"
#include "client.h"
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#define BAD_PARAM -1

/*
Handles '/query' requests received by the web server.
*/

static void writeCsvRow(SOCKET fd, struct Data* row);

void processQueryRequest(SOCKET fd, struct Request* req){
	struct NameValuePair* params = req->params;

	time_t from  = (time_t) getValueNumForName("from",  params,  BAD_PARAM);
	time_t to    = (time_t) getValueNumForName("to",    params,  BAD_PARAM);
    long   group = getValueNumForName("group", params,  BAD_PARAM);
	char*  ha    = getValueForName("ha", params, NULL);
	int    csv   = getValueNumForName("csv", params, FALSE);

    if (from == BAD_PARAM || to == BAD_PARAM || group == BAD_PARAM){
     // We need all 3 parameters
     	writeHeadersServerError(fd, "processQueryRequest, param bad/missing from=%s, to=%s, group=%d",
     		getValueForName("from",  params, NULL),
     		getValueForName("to",    params, NULL),
     		getValueForName("group", params, NULL));

    } else {
        if (from > to){
         // Allow from/to values in either order
            time_t tmp = from;
            from = to;
            to = tmp;
        }

     /* The client will send the last date that should be included in the query range in the 'to' parameter. When
        computing the timestamp value that corresponds to this, we need to move to the end of the specified date to
        be sure of including all data transferred during that day. */
        struct tm* cal = localtime(&to);
        cal->tm_mday++;
        to = mktime(cal);

     // Set the host/adapter values if appropriate
        char* hs = NULL;
        char* ad = NULL;
        struct HostAdapter* hostAdapter = NULL;

        if (ha != NULL) {
            hostAdapter = getHostAdapter(ha);
            hs = hostAdapter->host;
            ad = hostAdapter->adapter;
        }

        struct Data* result = getQueryValues(from, to, group, hs, ad);

        if (ha != NULL) {
            freeHostAdapter(hostAdapter);
        }

		if (csv){
		 // Export the query results in CSV format
		    writeHeadersOk(fd, MIME_CSV, FALSE);
		    writeHeader(fd, "Content-Disposition", "attachment;filename=bitmeterOsQuery.csv");
		    writeEndOfHeaders(fd);
		    
		    struct Data* thisResult = result;
		    
		    while(thisResult != NULL){
		    	writeCsvRow(fd, thisResult);	
		    	thisResult = thisResult->next;	
		    }
		    
		} else {
		 // Send results back as JSON
	        writeHeadersOk(fd, MIME_JSON, TRUE);
			writeDataToJson(fd, result);	
		}
        
        freeData(result);
    }

}

static void writeCsvRow(SOCKET fd, struct Data* row){
	char datePart[11];
	toDate(datePart, row->ts - row->dr);

	char timePart[9];
	toTime(timePart, row->ts - row->dr);

	char rowTxt[256];
	sprintf(rowTxt, "%s %s,%llu,%llu\n", datePart, timePart, row->dl, row->ul);	
	writeText(fd, rowTxt);
}