/*
    Copyright (C) 1995-2023, The AROS Development Team. All rights reserved.

    Desc: Display an alert in user mode.
*/

#include <aros/debug.h>
#include <exec/alerts.h>
#include <exec/rawfmt.h>
#include <intuition/intuition.h>
#include <proto/exec.h>
#include <proto/intuition.h>

#include "etask.h"
#define DEBUG_NOPRIVATEINLINE
#include "debug_intern.h"
#include "exec_intern.h"
#include "exec_util.h"
#include "exec_locks.h"

/* Remove formating strings from buffer. They might have been added by... disassebler in form of 'mov %dx, ...'. */
static VOID Sanitize(STRPTR buffer)
{
    LONG i = 0;
    while(buffer[i] != 0)
    {
        if (buffer[i] == '%') buffer[i] = ' ';
        i++;
    }
}

static LONG SafeEasyRequest(struct EasyStruct *es, BOOL full, struct IntuitionBase *IntuitionBase)
{
    LONG result;
    APTR req = BuildEasyRequestArgs(NULL, es, 0, NULL);

    if (!req)
    {
        /* Return -1 if requester creation failed. This makes us to fallback to safe-mode alert. */
        return -1;
    }

    do
    {
        result = SysReqHandler(req, NULL, TRUE);
        
        if (full)
        {
            switch (result)
            {
            case 1:
                NewRawDoFmt("*** Logged alert:\n%s\n", RAWFMTFUNC_SERIAL, NULL, es->es_TextFormat);
                result = -2;
                break;
            }
        }
    } while (result == -2);

    FreeSysRequest(req);
    return result;
}

static const char startstring[] = "Program failed\n";
static const char endstring[]   = "\nWait for disk activity to finish.";
static const char deadend_buttons[]          = "More...|Suspend|Reboot|Power off";
static const char recoverable_buttons[]      = "More...|Continue";
static const char full_deadend_buttons[]     = "Log|Suspend|Reboot|Power off";
static const char full_recoverable_buttons[] = "Log|Continue";

LONG Alert_AskSuspend(struct Task *task, ULONG alertNum, char * buffer, struct ExecBase *SysBase)
{
    LONG choice = -1;
    struct IntuitionBase *IntuitionBase;

    /* Arbitrate for the library list */
    EXEC_LOCK_LIST_READ_AND_FORBID(&SysBase->LibList);
    
    /* Look for the library in our list */
    IntuitionBase = (struct IntuitionBase *) FindName (&SysBase->LibList, "intuition.library");

    EXEC_UNLOCK_LIST(&SysBase->LibList);
    if (!IntuitionBase)
         return choice;

    IntuitionBase = (struct IntuitionBase *)TaggedOpenLibrary(TAGGEDOPEN_INTUITION);
    if (IntuitionBase)
    {
        if (buffer)
        {
            struct IntETask *iet = GetIntETask(task);
            char *buf, *end;
            struct EasyStruct es = {
                sizeof (struct EasyStruct),
                0,
                NULL,
                buffer,
                NULL,
            };
            BOOL full = FALSE;

            buf = Alert_AddString(buffer, startstring);
            buf = FormatAlert(buf, alertNum, task, iet ? iet->iet_AlertLocation : NULL, iet ? iet->iet_AlertType : AT_NONE, SysBase);
            end = buf;
            buf = Alert_AddString(buf, endstring);
            *buf = 0;

            es.es_Title = Alert_GetTitle(alertNum);

            /* Determine set of buttons */
            es.es_GadgetFormat = (alertNum & AT_DeadEnd) ? deadend_buttons : recoverable_buttons;

            D(bug("[UserAlert] Body text:\n%s\n", buffer));
reshow:
            choice = SafeEasyRequest(&es, full, IntuitionBase);

            if ((choice == 1) && !full)
            {
                /* 'More' has been pressed. Append full alert data */
                FormatAlertExtra(end, iet->iet_AlertStack, iet ? iet->iet_AlertType : AT_NONE, iet ? &iet->iet_AlertData : NULL, SysBase);

                /* Sanite the text before using in BuildEasyRequestArgs. Make sure all formatting strings are removed */
                Sanitize(end);

                /* Re-post the alert, without 'More...' this time */
                es.es_GadgetFormat = (alertNum & AT_DeadEnd) ? full_deadend_buttons : full_recoverable_buttons;
                full = TRUE;
                goto reshow;
            }
        }

        CloseLibrary(&IntuitionBase->LibNode);
    }
    return choice;
}

static LONG AskSuspend(struct Task *task, ULONG alertNum, struct ExecBase *SysBase)
{
    int allocsize = ALERT_BUFFER_SIZE;
    char *buffer;
    LONG choice;

    if ((PrivExecBase(SysBase)->DebugBase) && (((struct DebugBase *)(PrivExecBase(SysBase)->DebugBase))->db_Flags & DBFF_DISASSEMBLE))
        allocsize <<= 1;

    buffer = AllocMem(allocsize, MEMF_ANY);
    choice = Alert_AskSuspend(task, alertNum, buffer, SysBase);
    FreeMem(buffer, allocsize);

    return choice;
}


/*
 * This function posts alerts in user-mode via Intuition requester.
 * Returns initial alert code if something fails and 0 if it was a recoverable
 * alert and everything went ok.
 * Note that in case of some crashes (e.g. corrupt memory list) this function
 * may crash itself, and this has to be handled on a lower level. This is
 * why we do this trick with iet_AlertCode
 */
ULONG Exec_UserAlert(ULONG alertNum, struct ExecBase *SysBase)
{
    struct Task *task = GET_THIS_TASK;
    struct IntETask *iet;
    LONG res;

    /* Protect ourselves agains really hard crashes where SysBase->ThisTask is NULL.
       Obviously we won't go far away in such a case */
    if (!task)
        return alertNum;

    /* Get internal task structure */
    if ((iet = GetIntETask(task)))
    {
        /*
         * If we already have alert number for this task, we are in double-crash during displaying
         * intuition requester. Well, take the initial alert code (because it's more helpful to the programmer)
         * and proceed with arch-specific Alert().
         * Since this is a double-crash, we may append AT_DeadEnd flag if our situation has become unrecoverable.
         */
        D(bug("[UserAlert] Task alert state: 0x%02X\n", iet->iet_AlertFlags));
        if (iet->iet_AlertFlags & AF_Alert)
        {
            /*
             * Some more logic here. Nested AN_SysScrnType should not make original alert deadend.
             * It just means we were unable to display it using Intuition requested because there
             * are no display drivers at all.
             */
            if (alertNum == AN_SysScrnType)
                return iet->iet_AlertCode;
            else
                return iet->iet_AlertCode | (alertNum & AT_DeadEnd);
        }

        /*
         * Otherwise we can try to put up Intuition requester first. Store alert code in order in ETask
         * in order to indicate crash condition
         */
        iet->iet_AlertFlags |= AF_Alert;
        iet->iet_AlertCode   = alertNum;
    }

    /*
     * AN_SysScrnType is somewhat special. We remember it in the ETask (just in case),
     * but propagate it to supervisor mode immediately. We do it because this error
     * means we don't have any display modes, so we won't be able to bring up the requester.
     */
    if (alertNum == AN_SysScrnType)
        return alertNum;

    /* Issue a requester */
    res = AskSuspend(task, alertNum, SysBase);
    D(bug("[UserAlert] Requester result: %d\n", res));

    /* If AskSuspend() failed, fail back to safe-mode alert */
    if (res == -1)
        return alertNum;

    /* Halt if we need to */
    if (alertNum & AT_DeadEnd)
    {
        switch (res)
        {
        case 0:
            ShutdownA(SD_ACTION_POWEROFF);
            break;

        case 3:
            ColdReboot();
            /* In case if ColdReboot() doesn't work */
            ShutdownA(SD_ACTION_COLDREBOOT);
            break;
        }

        /* Well, stop if the user wants so (or if the reboot didn't work at all) */
        Wait(0);
    }
    return 0;
}
