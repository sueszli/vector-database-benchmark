/*
    Copyright (C) 2020-2023, The AROS Development Team. All rights reserved.

    Desc: i386/x86_64 native PCI device support routines.
*/

#include <aros/debug.h>

#include <proto/kernel.h>
#include <proto/exec.h>
#include <proto/utility.h>
#include <proto/oop.h>
#include <proto/acpica.h>

#include <aros/cpu.h>
#include <exec/types.h>
#include <utility/tagitem.h>
#include <oop/oop.h>
#include <hidd/pci.h>
#include <hardware/pci.h>

#include <acpica/acnames.h>
#include <acpica/accommon.h>

#include <string.h>

#include "pcipc.h"

#define DMSI(x)

#if (1)
/* TODO: Move into a suitable header */

/* Find the most-significant bit thats set */
static inline ULONG fls_long(ULONG x)
{
    return ((sizeof(x) * 8) - __builtin_clz(x));
}

/* Integer base 2 logarithm of x */
static inline ULONG ilog2(ULONG x)
{
    return (fls_long(x) - 1);
}

/* Round up, to nearest power of two */
static inline ULONG roundup_pow_of_two(ULONG x)
{
    return x == 1 ? 1 : (1 << fls_long(x - 1));
}
#endif

OOP_Object *PCIPCDev__Root__New(OOP_Class *cl, OOP_Object *o, struct pRoot_New *msg)
{
    OOP_Object *driver = (OOP_Object *)GetTagData(aHidd_PCIDevice_Driver, 0, msg->attrList);
    ULONG deviceBus = (ULONG)GetTagData(aHidd_PCIDevice_Bus, 0, msg->attrList);
    ULONG deviceDev = (ULONG)GetTagData(aHidd_PCIDevice_Dev, 0, msg->attrList);
    ULONG deviceSub = (ULONG)GetTagData(aHidd_PCIDevice_Sub, 0, msg->attrList);
    struct pRoot_New pcidevNew;
    struct PCIPCBusData *ddata;
    struct TagItem pcidevTags[] =
    {
        { aHidd_Name,                           (IPTR)"pcipc.hidd"      },
        { aHidd_PCIDevice_ExtendedConfig,       0                       },
        { TAG_DONE,                             0                       }
    };
    IPTR mmconfig = 0;
    OOP_Object *deviceObj;

    ddata = OOP_INST_DATA(PSD(cl)->pcipcDriverClass, driver);

    pcidevNew.mID      = msg->mID;
    pcidevNew.attrList = pcidevTags;

    if (msg->attrList)
    {
        pcidevTags[2].ti_Tag  = TAG_MORE;
        pcidevTags[2].ti_Data = (IPTR)msg->attrList;
    }
 
    if(PSD(cl)->pcipc_acpiMcfgTbl) {
        ACPI_MCFG_ALLOCATION *mcfg_alloc;
        int i, nsegs = 0;
        ULONG offset;

        offset = sizeof(ACPI_TABLE_MCFG);
        mcfg_alloc = ACPI_ADD_PTR(ACPI_MCFG_ALLOCATION, PSD(cl)->pcipc_acpiMcfgTbl, offset);

        D(bug("[PCIPC:Device] %s: Parsing MCFG Table allocations...\n", __func__);)
        for (i = 0; offset + sizeof(ACPI_MCFG_ALLOCATION) <= PSD(cl)->pcipc_acpiMcfgTbl->Header.Length; i++)
        {
            D(bug("[PCIPC:Device] %s:     #%u %p - segment %d, bus %d-%d, address 0x%p\n",
                    __func__, i, mcfg_alloc, mcfg_alloc->PciSegment, mcfg_alloc->StartBusNumber, mcfg_alloc->EndBusNumber,
                    mcfg_alloc->Address);
            )
            nsegs++;
            if ((deviceBus <= mcfg_alloc->EndBusNumber) && (deviceBus >= mcfg_alloc->StartBusNumber))
            {
                D(bug("[PCIPC:Device] %s:       * bus %d, dev %d, sub %d\n", __func__, deviceBus, deviceDev, deviceSub);)

                mmconfig = ((IPTR)mcfg_alloc->Address) | (((deviceBus - mcfg_alloc->StartBusNumber) & 255)<<20) | ((deviceDev & 31) << 15) | ((deviceSub & 7) << 12);
                D(bug("[PCIPC:Device] %s:             MMIO @ 0x%p\n", __func__, mmconfig);)
                if (ddata->ecam)
                {
                    D(bug("[PCIPC:Device] %s:             ECAM Access\n", __func__);)
                    pcidevTags[1].ti_Data = mmconfig;
                }
                break;
            }
            offset += sizeof(ACPI_MCFG_ALLOCATION);
            mcfg_alloc = ACPI_ADD_PTR(ACPI_MCFG_ALLOCATION, PSD(cl)->pcipc_acpiMcfgTbl, offset);
        }
        D(bug("[PCIPC:Device] %s: checked %u segment allocation(s)\n", __func__, nsegs);)
    }

    deviceObj = (OOP_Object *)OOP_DoSuperMethod(cl, o, &pcidevNew.mID);
    if (deviceObj)
    {
        struct PCIPCDeviceData *data = OOP_INST_DATA(cl, deviceObj);

        D(bug("[PCIPC:Device] %s: Device Object created @ 0x%p\n", __func__, deviceObj);)

        data->mmconfig = (APTR)mmconfig;
        /* If we didnt detect ECAM support from ACPI, try
         * to detct it now. */
        if (!deviceBus && !deviceDev && !deviceSub && !ddata->ecam)
        {
            struct pHidd_PCIDriver_ReadConfigLong msg;

            msg.mID = HiddPCIDeviceBase + moHidd_PCIDriver_ReadConfigLong;
            msg.device = deviceObj;
            msg.bus = msg.dev = msg.sub = 0;
            msg.reg = PCIEXBAR;
            ddata->ecam = (APTR)OOP_DoMethod(driver, (OOP_Msg)&msg);
            D(bug("[PCIPC:Device] %s: ECAM @ 0x%p\n", __func__, ddata->ecam);)
            if ((ddata->ecam) && (data->mmconfig))
            {
                pcidevTags[1].ti_Data = mmconfig;
                D(bug("[PCIPC:Device] %s: disposing original device object @ 0x%p\n", __func__, deviceObj);)
                OOP_DisposeObject(deviceObj) ;
                D(bug("[PCIPC:Device] %s: creating new instance ...\n", __func__);)
                deviceObj = (OOP_Object *)OOP_DoSuperMethod(cl, o, &pcidevNew.mID);
                if (deviceObj)
                {
                    struct PCIPCDeviceData *data = OOP_INST_DATA(cl, deviceObj);
                    D(bug("[PCIPC:Device] %s: New Host Bridge Device @ 0x%p\n", __func__, deviceObj);)
                    data->mmconfig = (APTR)mmconfig;
                }
            }
        }
    }
    return deviceObj;
}

void PCIPCDev__Root__Get(OOP_Class *cl, OOP_Object *o, struct pRoot_Get *msg)
{
    BOOL handled = FALSE;
    ULONG idx;

    if (IS_PCIDEV_ATTR(msg->attrID, idx))
    {
        switch (idx)
        {
            case aoHidd_PCIDevice_MSICount:
                {
                    IPTR capmsi;

                    handled = TRUE;
                    *msg->storage = 1;

                    OOP_GetAttr(o, aHidd_PCIDevice_CapabilityMSI, &capmsi);
                    if (capmsi)
                    {
                        struct pHidd_PCIDevice_ReadConfigWord cmeth;
                        UWORD msiflags;

                        cmeth.mID = HiddPCIDeviceBase + moHidd_PCIDevice_ReadConfigWord;
                        cmeth.reg = capmsi + PCIMSI_FLAGS;
                        msiflags = (UWORD)OOP_DoMethod(o, &cmeth.mID);
                        *msg->storage = ( 1 << ((msiflags & PCIMSIF_MMC_MASK) >> 1));
                    }
                    break;
                }
        }
    }

    if (!handled)
    {
        OOP_DoSuperMethod(cl, o, (OOP_Msg) msg);
    }
}

VOID PCIPCDev__Hidd_PCIDevice__GetVectorAttribs(OOP_Class *cl, OOP_Object *o, struct pHidd_PCIDevice_GetVectorAttribs *msg)
{
    struct PCIPCDeviceData *data = OOP_INST_DATA(cl, o);
    IPTR capmsi;
    UBYTE vectirq = 0;

    D(bug("[PCIPC:Device] %s()\n", __func__);)

    OOP_GetAttr(o, aHidd_PCIDevice_CapabilityMSI, &capmsi);

    /* Check if MSI is supported */
    if (capmsi)
    {
        struct pHidd_PCIDevice_ReadConfigWord cmeth;
        UWORD msiflags;

        cmeth.mID = HiddPCIDeviceBase + moHidd_PCIDevice_ReadConfigWord;
        cmeth.reg = capmsi + PCIMSI_FLAGS;
        msiflags = (UWORD)OOP_DoMethod(o, &cmeth.mID);
        if (msiflags & PCIMSIF_ENABLE)
        {
            DMSI(bug("[PCIPC:Device] %s: #%u MSI message vectors\n", __func__, (1 <<((msiflags & PCIMSIF_MMEN_MASK) >> 4)));)

            /* MSI is enabled .. but is the requested vector valid? */
            if (msg->vectorno < (1 <<((msiflags & PCIMSIF_MMEN_MASK) >> 4)))
            {
                struct TagItem *tag, *tags;
                  tags=(struct TagItem *)msg->attribs;

                while((tag = NextTagItem(&tags)))
                {
                    switch (tag->ti_Tag)
                    {
                        case tHidd_PCIVector_Int:
                            tag->ti_Data = ((data->msimsg & 0xFF) + msg->vectorno) - HW_IRQ_BASE;
                            break;

                        case tHidd_PCIVector_Native:
                            tag->ti_Data = ((data->msimsg & 0xFF) + msg->vectorno);
                            break;

                        default:
                            break;
                    }
                }
            }
            else
            {
                bug("[PCIPC:Device] %s: Illegal MSI vector %u\n", __func__, msg->vectorno);
            }
        }
        else
        {
            bug("[PCIPC:Device] %s: MSI is dissabled for the device\n", __func__);
        }
    }
    else
    {
        bug("[PCIPC:Device] %s: Device doesn't support MSI\n", __func__);
    }
}


BOOL PCIPCDev__Hidd_PCIDevice__ObtainVectors(OOP_Class *cl, OOP_Object *o, struct pHidd_PCIDevice_ObtainVectors *msg)
{
    struct PCIPCDeviceData *data = OOP_INST_DATA(cl, o);
    IPTR capmsi, capmsix;

    D(bug("[PCIPC:Device] %s()\n", __func__);)

    OOP_GetAttr(o, aHidd_PCIDevice_CapabilityMSI, &capmsi);
    OOP_GetAttr(o, aHidd_PCIDevice_CapabilityMSIX, &capmsix);

    if (capmsix)
    {
        DMSI(bug("[PCIPC:Device] %s: Device has MSI-X capability @ %u\n", __func__, capmsix);)
    }
    if (capmsi)
    {
        union {
            struct pHidd_PCIDevice_WriteConfigWord wcw;
            struct pHidd_PCIDevice_WriteConfigLong wcl;
        } cmeth;
        UWORD vectmin, vectmax, vectcnt, vecthw;
        ULONG apicIRQBase = 0;
        UWORD msiflags;

        DMSI(bug("[PCIPC:Device] %s: Device has MSI capability @ %u\n", __func__, capmsi);)

        cmeth.wcw.mID = HiddPCIDeviceBase + moHidd_PCIDevice_ReadConfigWord;
        cmeth.wcw.reg = capmsi + PCIMSI_FLAGS;
        msiflags = (UWORD)OOP_DoMethod(o, &cmeth.wcw.mID);

        DMSI(bug("[PCIPC:Device] %s: Initial MSI Flags = %04x\n", __func__, msiflags);)
        if (msiflags & PCIMSIF_ENABLE)
            return FALSE;

        vecthw = (1 << ((msiflags & PCIMSIF_MMC_MASK) >> 1));
        DMSI(bug("[PCIPC:Device] %s: Max Device MSI vectors = %u\n", __func__, vecthw);)

        vectmin = (UWORD)GetTagData(tHidd_PCIVector_Min, 0, msg->requirements);
        if (vectmin > vecthw)
            return FALSE;

        vectmax = (UWORD)GetTagData(tHidd_PCIVector_Max, 0, msg->requirements);
        if (vectmax > vecthw)
            vectmax = vecthw;

        for (vectcnt = vectmax; vectcnt >= vectmin; vectcnt--)
        {
            if ((apicIRQBase = KrnAllocIRQ(IRQTYPE_APIC, vectcnt)) != (ULONG)-1)
            {
                DMSI(bug("[PCIPC:Device] %s: Allocated %u vectors (start = IRQ #%u)\n", __func__, vectcnt, apicIRQBase);)
                break;
            }
        }
        if ((apicIRQBase) && (apicIRQBase != -1))
        {
            DMSI(bug("[PCIPC:Device] %s: Configuring Device with %u MSI vectors...\n", __func__, vectcnt);)
            msiflags |= (ilog2(roundup_pow_of_two(vectcnt)) << 4);

            cmeth.wcl.mID = HiddPCIDeviceBase + moHidd_PCIDevice_WriteConfigLong;
            cmeth.wcl.reg = capmsi + PCIMSI_ADDRESSLO;
            cmeth.wcl.val = (0xFEE << 20) | (0 << 12);
            OOP_DoMethod(o, &cmeth.wcl.mID);

            data->msimsg = (UWORD)apicIRQBase + HW_IRQ_BASE;

            if (msiflags & PCIMSIF_64BIT)
            {
                DMSI(bug("[PCIPC:Device] %s: -- 64bit device!\n", __func__);)
                cmeth.wcl.mID = HiddPCIDeviceBase + moHidd_PCIDevice_WriteConfigLong;
                cmeth.wcl.reg = capmsi + PCIMSI_ADDRESSHI;
                cmeth.wcl.val = 0;
                OOP_DoMethod(o, &cmeth.wcl.mID);
                
                cmeth.wcw.mID = HiddPCIDeviceBase + moHidd_PCIDevice_WriteConfigWord;
                cmeth.wcw.reg = capmsi + PCIMSI_DATA64;
                cmeth.wcw.val = data->msimsg;
                OOP_DoMethod(o, &cmeth.wcw.mID);
            }
            else
            {
                cmeth.wcw.mID = HiddPCIDeviceBase + moHidd_PCIDevice_WriteConfigWord;
                cmeth.wcw.reg = capmsi + PCIMSI_DATA32;
                cmeth.wcw.val = data->msimsg;
                OOP_DoMethod(o, &cmeth.wcw.mID);
            }
            DMSI(bug("[PCIPC:Device] %s: -- MSI message = %04x\n", __func__, cmeth.wcw.val);)

            msiflags |= PCIMSIF_ENABLE;
            cmeth.wcw.mID = HiddPCIDeviceBase + moHidd_PCIDevice_WriteConfigWord;
            cmeth.wcw.reg = capmsi + PCIMSI_FLAGS;
            cmeth.wcw.val = msiflags;
            OOP_DoMethod(o, &cmeth.wcw.mID);

            cmeth.wcw.mID = HiddPCIDeviceBase + moHidd_PCIDevice_ReadConfigWord;
            cmeth.wcw.reg = capmsi + PCIMSI_FLAGS;
            msiflags = (UWORD)OOP_DoMethod(o, &cmeth.wcw.mID);
            DMSI(bug("[PCIPC:Device] %s: Configured MSI Flags = %04x\n", __func__, msiflags);)
            if (msiflags & PCIMSIF_ENABLE)
                return TRUE;
        }
    }
    DMSI(bug("[PCIPC:Device] %s: Failed to obtain/enable MSI vectors\n", __func__);)

    return FALSE;
}

VOID PCIPCDev__Hidd_PCIDevice__ReleaseVectors(OOP_Class *cl, OOP_Object *o, struct pHidd_PCIDevice_ReleaseVectors *msg)
{
    D(bug("[PCIPC:Device] %s()\n", __func__);)
    struct PCIPCDeviceData *data = OOP_INST_DATA(cl, o);
    IPTR capmsi;
    UBYTE vectirq = 0;

    D(bug("[PCIPC:Device] %s()\n", __func__);)

    OOP_GetAttr(o, aHidd_PCIDevice_CapabilityMSI, &capmsi);

    if (capmsi)
    {
        union {
            struct pHidd_PCIDevice_WriteConfigWord wcw;
            struct pHidd_PCIDevice_WriteConfigLong wcl;
        } cmeth;
        UWORD msiflags;

        cmeth.wcw.mID = HiddPCIDeviceBase + moHidd_PCIDevice_ReadConfigWord;
        cmeth.wcw.reg = capmsi + PCIMSI_FLAGS;
        msiflags = (UWORD)OOP_DoMethod(o, &cmeth.wcw.mID);
        if (msiflags & PCIMSIF_ENABLE)
        {
            msiflags &= ~PCIMSIF_ENABLE;
            cmeth.wcw.mID = HiddPCIDeviceBase + moHidd_PCIDevice_WriteConfigWord;
            cmeth.wcw.reg = capmsi + PCIMSI_FLAGS;
            cmeth.wcw.val = msiflags;
            OOP_DoMethod(o, &cmeth.wcw.mID);

            cmeth.wcw.mID = HiddPCIDeviceBase + moHidd_PCIDevice_ReadConfigWord;
            cmeth.wcw.reg = capmsi + PCIMSI_FLAGS;
            msiflags = (UWORD)OOP_DoMethod(o, &cmeth.wcw.mID);
            DMSI(bug("[PCIPC:Device] %s: Configured MSI Flags = %04x\n", __func__, msiflags);)
            if (msiflags & PCIMSIF_ENABLE)
            {
                bug("[PCIPC:Device] %s: Failed to disable MSI interrupts for device\n", __func__);
            }
        }
    }
    return;
}
