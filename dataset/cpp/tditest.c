/*
 * COPYRIGHT:   See COPYING in the top level directory
 * PROJECT:     ReactOS TDI test driver
 * FILE:        tditest.c
 * PURPOSE:     Testing TDI drivers
 * PROGRAMMERS: Casper S. Hornstrup (chorns@users.sourceforge.net)
 *              Vizzini (vizzini@plasmic.com)
 * REVISIONS:
 *   CSH 01/08-2000 Created
 *   26-Nov-2003 Vizzini Updated to run properly on Win2ksp4
 */
#include <tditest.h>
#include <pseh/pseh2.h>


#if DBG

/* See debug.h for debug/trace constants */
ULONG DebugTraceLevel = -1;

#endif /* DBG */


HANDLE TdiTransport             = 0;
PFILE_OBJECT TdiTransportObject = NULL;
ULONG LocalAddress;
BOOLEAN OpenError;
KEVENT StopEvent;
HANDLE SendThread;
HANDLE ReceiveThread;

NTSTATUS TdiCall(
    PIRP Irp,
    PDEVICE_OBJECT DeviceObject,
    PIO_STATUS_BLOCK IoStatusBlock,
    BOOLEAN CanCancel)
/*
 * FUNCTION: Calls a transport driver device
 * ARGUMENTS:
 *     Irp           = Pointer to I/O Request Packet
 *     DeviceObject  = Pointer to device object to call
 *     IoStatusBlock = Address of buffer with I/O status block
 *     CanCancel     = TRUE if the IRP can be cancelled, FALSE if not
 * RETURNS:
 *     Status of operation
 * NOTES
 *     All requests are completed synchronously. A request may be cancelled
 */
{
	KEVENT Event;
	PKEVENT Events[2];
	NTSTATUS Status;
	Events[0] = &StopEvent;
	Events[1] = &Event;

	KeInitializeEvent(&Event, NotificationEvent, FALSE);
	Irp->UserEvent = &Event;
	Irp->UserIosb  = IoStatusBlock;

	Status = IoCallDriver(DeviceObject, Irp);

	if (Status == STATUS_PENDING)
		{
			if (CanCancel)
				{
					Status = KeWaitForMultipleObjects(2, (PVOID)Events, WaitAny, Executive, KernelMode, FALSE, NULL, NULL);

					if (KeReadStateEvent(&StopEvent) != 0)
						{
							if (IoCancelIrp(Irp))
								{
									TDI_DbgPrint(MAX_TRACE, ("Cancelled IRP.\n"));
								}
							else
								{
									TDI_DbgPrint(MIN_TRACE, ("Could not cancel IRP.\n"));
								}
							return STATUS_CANCELLED;
						}
				}
			else
				Status = KeWaitForSingleObject(&Event, Executive, KernelMode, FALSE, NULL);
		}

	return (Status == STATUS_SUCCESS)? IoStatusBlock->Status : STATUS_SUCCESS;
}


NTSTATUS TdiOpenDevice(
    PWSTR Protocol,
    ULONG EaLength,
    PFILE_FULL_EA_INFORMATION EaInfo,
    PHANDLE Handle,
    PFILE_OBJECT *Object)
/*
 * FUNCTION: Opens a device
 * ARGUMENTS:
 *     Protocol = Pointer to buffer with name of device
 *     EaLength = Length of EA information
 *     EaInfo   = Pointer to buffer with EA information
 *     Handle   = Address of buffer to place device handle
 *     Object   = Address of buffer to place device object
 * RETURNS:
 *     Status of operation
 */
{
	OBJECT_ATTRIBUTES Attr;
	IO_STATUS_BLOCK Iosb;
	UNICODE_STRING Name;
	NTSTATUS Status;

	RtlInitUnicodeString(&Name, Protocol);
	InitializeObjectAttributes(
		&Attr,                   /* Attribute buffer */
		&Name,                   /* Device name */
		OBJ_CASE_INSENSITIVE,    /* Attributes */
		NULL,                    /* Root directory */
		NULL);                   /* Security descriptor */

	Status = ZwCreateFile(
		Handle,                               /* Return file handle */
		GENERIC_READ | GENERIC_WRITE,         /* Desired access */
		&Attr,                                /* Object attributes */
		&Iosb,                                /* IO status */
		0,                                    /* Initial allocation size */
		FILE_ATTRIBUTE_NORMAL,                /* File attributes */
		FILE_SHARE_READ | FILE_SHARE_WRITE,   /* Share access */
		FILE_OPEN_IF,                         /* Create disposition */
		0,                                    /* Create options */
		EaInfo,                               /* EA buffer */
		EaLength);                            /* EA length */

	if (NT_SUCCESS(Status))
		{
			Status  = ObReferenceObjectByHandle(
				*Handle,                        /* Handle to open file */
				GENERIC_READ | GENERIC_WRITE,   /* Access mode */
				NULL,                           /* Object type */
				KernelMode,                     /* Access mode */
				(PVOID*)Object,                 /* Pointer to object */
				NULL);                          /* Handle information */

			if (!NT_SUCCESS(Status))
				{
					TDI_DbgPrint(MIN_TRACE, ("ObReferenceObjectByHandle() failed with status (0x%X).\n", Status));
					ZwClose(*Handle);
				}
		}
	else
		{
			TDI_DbgPrint(MIN_TRACE, ("ZwCreateFile() failed with status (0x%X)\n", Status));
		}

    return Status;
}


NTSTATUS TdiCloseDevice(
    HANDLE Handle,
    PFILE_OBJECT FileObject)
{
	if (FileObject)
		ObDereferenceObject(FileObject);

	if (Handle)
		ZwClose(Handle);

	return STATUS_SUCCESS;
}


NTSTATUS TdiOpenTransport(
    PWSTR Protocol,
    USHORT Port,
    PHANDLE Transport,
    PFILE_OBJECT *TransportObject)
/*
 * FUNCTION: Opens a transport driver
 * ARGUMENTS:
 *     Protocol        = Pointer to buffer with name of device
 *     Port            = Port number to use
 *     Transport       = Address of buffer to place transport device handle
 *     TransportObject = Address of buffer to place transport object
 * RETURNS:
 *     Status of operation
 */
{
	PFILE_FULL_EA_INFORMATION EaInfo;
	PTA_IP_ADDRESS Address;
	NTSTATUS Status;
	ULONG EaLength;

	/* EaName must be 0-termed, even though TDI_TRANSPORT_ADDRESS_LENGTH does *not* include the 0 */
	EaLength = sizeof(FILE_FULL_EA_INFORMATION) + TDI_TRANSPORT_ADDRESS_LENGTH + sizeof(TA_IP_ADDRESS) + 1;
	EaInfo = (PFILE_FULL_EA_INFORMATION)ExAllocatePool(NonPagedPool, EaLength);

	if (!EaInfo)
		{
			TDI_DbgPrint(MIN_TRACE, ("Insufficient resources.\n"));
			return STATUS_INSUFFICIENT_RESOURCES;
		}

	RtlZeroMemory(EaInfo, EaLength);

	EaInfo->EaNameLength = TDI_TRANSPORT_ADDRESS_LENGTH;

	/* don't copy the 0; we have already zeroed it */
	RtlCopyMemory(EaInfo->EaName, TdiTransportAddress, TDI_TRANSPORT_ADDRESS_LENGTH);

	EaInfo->EaValueLength = sizeof(TA_IP_ADDRESS);
	Address = (PTA_IP_ADDRESS)(EaInfo->EaName + TDI_TRANSPORT_ADDRESS_LENGTH + 1); // 0-term
	Address->TAAddressCount                 = 1;
	Address->Address[0].AddressLength       = TDI_ADDRESS_LENGTH_IP;
	Address->Address[0].AddressType         = TDI_ADDRESS_TYPE_IP;
	Address->Address[0].Address[0].sin_port = WH2N(Port);
	Address->Address[0].Address[0].in_addr  = 0;

	Status = TdiOpenDevice(Protocol, EaLength, EaInfo, Transport, TransportObject);

	ExFreePool(EaInfo);

	return Status;
}


NTSTATUS TdiQueryDeviceControl(
    PFILE_OBJECT FileObject,
    ULONG IoControlCode,
    PVOID InputBuffer,
    ULONG InputBufferLength,
    PVOID OutputBuffer,
    ULONG OutputBufferLength,
    PULONG Return)
/*
 * FUNCTION: Queries a device for information
 * ARGUMENTS:
 *     FileObject         = Pointer to device object
 *     IoControlCode      = I/O control code
 *     InputBuffer        = Pointer to buffer with input data
 *     InputBufferLength  = Length of InputBuffer
 *     OutputBuffer       = Address of buffer to place output data
 *     OutputBufferLength = Length of OutputBuffer
 * RETURNS:
 *     Status of operation
 */
{
	PDEVICE_OBJECT DeviceObject;
	PIO_STACK_LOCATION IoStack;
	IO_STATUS_BLOCK Iosb;
	NTSTATUS Status;
	PIRP Irp;

	DeviceObject = IoGetRelatedDeviceObject(FileObject);
	Irp = IoBuildDeviceIoControlRequest(IoControlCode, DeviceObject, InputBuffer, InputBufferLength, OutputBuffer,
		OutputBufferLength, FALSE, NULL, NULL);

	if (!Irp)
		{
			TDI_DbgPrint(MIN_TRACE, ("IoBuildDeviceIoControlRequest() failed.\n"));
			return STATUS_INSUFFICIENT_RESOURCES;
		}

	IoStack               = IoGetNextIrpStackLocation(Irp);
	IoStack->DeviceObject = DeviceObject;
	IoStack->FileObject   = FileObject;
	Status = TdiCall(Irp, DeviceObject, &Iosb, FALSE);

	if (Return)
		*Return = Iosb.Information;

	return Status;
}


NTSTATUS TdiQueryInformationEx(
    PFILE_OBJECT FileObject,
    ULONG Entity,
    ULONG Instance,
    ULONG Class,
    ULONG Type,
    ULONG Id,
    PVOID OutputBuffer,
    PULONG OutputLength)
/*
 * FUNCTION: Extended query for information
 * ARGUMENTS:
 *     FileObject   = Pointer to transport object
 *     Entity       = Entity
 *     Instance     = Instance
 *     Class        = Entity class
 *     Type         = Entity type
 *     Id           = Entity id
 *     OutputBuffer = Address of buffer to place data
 *     OutputLength = Address of buffer with length of OutputBuffer (updated)
 * RETURNS:
 *     Status of operation
 */
{
 	TCP_REQUEST_QUERY_INFORMATION_EX QueryInfo;

	RtlZeroMemory(&QueryInfo, sizeof(TCP_REQUEST_QUERY_INFORMATION_EX));
	QueryInfo.ID.toi_entity.tei_entity   = Entity;
	QueryInfo.ID.toi_entity.tei_instance = Instance;
	QueryInfo.ID.toi_class = Class;
	QueryInfo.ID.toi_type  = Type;
	QueryInfo.ID.toi_id    = Id;

	return TdiQueryDeviceControl(
		FileObject,                                /* Transport/connection object */
		IOCTL_TCP_QUERY_INFORMATION_EX,            /* Control code */
		&QueryInfo,                                /* Input buffer */
		sizeof(TCP_REQUEST_QUERY_INFORMATION_EX),  /* Input buffer length */
		OutputBuffer,                              /* Output buffer */
		*OutputLength,                             /* Output buffer length */
		OutputLength);                             /* Return information */
}


NTSTATUS TdiQueryAddress(
    PFILE_OBJECT FileObject,
    PULONG Address)
/*
 * FUNCTION: Queries for a local IP address
 * ARGUMENTS:
 *     FileObject = Pointer to file object
 *     Address    = Address of buffer to place local address
 * RETURNS:
 *     Status of operation
 */
{
	ULONG i;
	TDIEntityID *Entities;
	ULONG EntityCount;
	ULONG EntityType;
	IPSNMP_INFO SnmpInfo;
	PIPADDR_ENTRY IpAddress;
	ULONG BufferSize;
	NTSTATUS Status = STATUS_SUCCESS;

	TDI_DbgPrint(MAX_TRACE, ("Called\n"));

	BufferSize = sizeof(TDIEntityID) * 20;
	Entities   = (TDIEntityID*)ExAllocatePool(NonPagedPool, BufferSize);

	if (!Entities)
		{
			TDI_DbgPrint(MIN_TRACE, ("Insufficient resources.\n"));
			return STATUS_INSUFFICIENT_RESOURCES;
		}

	/* Query device for supported entities */
	Status = TdiQueryInformationEx(
		FileObject,          /* File object */
		GENERIC_ENTITY,      /* Entity */
		TL_INSTANCE,         /* Instance */
		INFO_CLASS_GENERIC,  /* Entity class */
		INFO_TYPE_PROVIDER,  /* Entity type */
		ENTITY_LIST_ID,      /* Entity id */
		Entities,            /* Output buffer */
		&BufferSize);        /* Output buffer size */

	if (!NT_SUCCESS(Status))
		{
			TDI_DbgPrint(MIN_TRACE, ("Unable to get list of supported entities (Status = 0x%X).\n", Status));
			ExFreePool(Entities);
			return Status;
		}

	/* Locate an IP entity */
	EntityCount = BufferSize / sizeof(TDIEntityID);

	TDI_DbgPrint(MAX_TRACE, ("EntityCount = %d\n", EntityCount));

	for (i = 0; i < EntityCount; i++)
		{
			if (Entities[i].tei_entity == CL_NL_ENTITY)
				{
					/* Query device for entity type */
					BufferSize = sizeof(EntityType);
					Status = TdiQueryInformationEx(
						FileObject,                  /* File object */
						CL_NL_ENTITY,                /* Entity */
						Entities[i].tei_instance,    /* Instance */
						INFO_CLASS_GENERIC,          /* Entity class */
						INFO_TYPE_PROVIDER,          /* Entity type */
						ENTITY_TYPE_ID,              /* Entity id */
						&EntityType,                 /* Output buffer */
						&BufferSize);                /* Output buffer size */

					if (!NT_SUCCESS(Status) || (EntityType != CL_NL_IP))
						{
							TDI_DbgPrint(MIN_TRACE, ("Unable to get entity of type IP (Status = 0x%X).\n", Status));
							break;
						}

					/* Query device for SNMP information */
					BufferSize = sizeof(SnmpInfo);
 					Status = TdiQueryInformationEx(
						FileObject,                  /* File object */
						CL_NL_ENTITY,                /* Entity */
						Entities[i].tei_instance,    /* Instance */
						INFO_CLASS_PROTOCOL,         /* Entity class */
						INFO_TYPE_PROVIDER,          /* Entity type */
						IP_MIB_STATS_ID,             /* Entity id */
						&SnmpInfo,                   /* Output buffer */
						&BufferSize);                /* Output buffer size */

					if (!NT_SUCCESS(Status) || (SnmpInfo.NumAddr == 0))
						{
							TDI_DbgPrint(MIN_TRACE, ("Unable to get SNMP information or no IP addresses available (Status = 0x%X).\n", Status));
							break;
						}

					/* Query device for all IP addresses */
					if (SnmpInfo.NumAddr != 0)
						{
							BufferSize = SnmpInfo.NumAddr * sizeof(IPADDR_ENTRY);
							IpAddress = (PIPADDR_ENTRY)ExAllocatePool(NonPagedPool, BufferSize);
							if (!IpAddress)
								{
									TDI_DbgPrint(MIN_TRACE, ("Insufficient resources.\n"));
									break;
								}

						Status = TdiQueryInformationEx(
							FileObject,                  /* File object */
							CL_NL_ENTITY,                /* Entity */
							Entities[i].tei_instance,    /* Instance */
							INFO_CLASS_PROTOCOL,         /* Entity class */
							INFO_TYPE_PROVIDER,          /* Entity type */
							IP_MIB_ADDRTABLE_ENTRY_ID,   /* Entity id */
							IpAddress,                   /* Output buffer */
							&BufferSize);                /* Output buffer size */

						if (!NT_SUCCESS(Status))
							{
								TDI_DbgPrint(MIN_TRACE, ("Unable to get IP address (Status = 0x%X).\n", Status));
								ExFreePool(IpAddress);
								break;
							}

						if (SnmpInfo.NumAddr != 1)
							{
								/* Skip loopback address */
								*Address = DN2H(((PIPADDR_ENTRY)((PUCHAR)IpAddress + sizeof(IPADDR_ENTRY)))->Addr);
							}
						else
							{
								/* Select the first address returned */
								*Address = DN2H(IpAddress->Addr);
							}
								ExFreePool(IpAddress);

						}
					else
						{
							Status = STATUS_UNSUCCESSFUL;
							break;
					}
			}
	}

	ExFreePool(Entities);

	TDI_DbgPrint(MAX_TRACE, ("Leaving\n"));

	return Status;
}


NTSTATUS TdiSendDatagram(
    PFILE_OBJECT TransportObject,
    USHORT Port,
    ULONG Address,
    PVOID Buffer,
    ULONG BufferSize)
/*
 * FUNCTION: Sends a datagram
 * ARGUMENTS:
 *     TransportObject = Pointer to transport object
 *     Port            = Remote port
 *     Address         = Remote address
 *     Buffer          = Pointer to buffer with data to send
 *     BufferSize      = Length of Buffer
 * RETURNS:
 *     Status of operation
 */
{
	PIRP Irp;
	PMDL Mdl;
	PDEVICE_OBJECT DeviceObject;
	PTDI_CONNECTION_INFORMATION ConnectInfo;
	PTA_IP_ADDRESS TA;
	PTDI_ADDRESS_IP IpAddress;
	IO_STATUS_BLOCK Iosb;
	NTSTATUS Status;

	DeviceObject = IoGetRelatedDeviceObject(TransportObject);
	ConnectInfo  = (PTDI_CONNECTION_INFORMATION)
		ExAllocatePool(NonPagedPool,
		sizeof(TDI_CONNECTION_INFORMATION) +
		sizeof(TA_IP_ADDRESS));

	if (!ConnectInfo)
		return STATUS_INSUFFICIENT_RESOURCES;

	RtlZeroMemory(ConnectInfo, sizeof(TDI_CONNECTION_INFORMATION) + sizeof(TA_IP_ADDRESS));

	ConnectInfo->RemoteAddressLength = sizeof(TA_IP_ADDRESS);
	ConnectInfo->RemoteAddress       = ((PUCHAR)ConnectInfo + sizeof(TDI_CONNECTION_INFORMATION));

	TA = (PTA_IP_ADDRESS)(ConnectInfo->RemoteAddress);
	TA->TAAddressCount           = 1;
	TA->Address[0].AddressLength = sizeof(TDI_ADDRESS_IP);
	TA->Address[0].AddressType   = TDI_ADDRESS_TYPE_IP;
	IpAddress           = (PTDI_ADDRESS_IP)(TA->Address[0].Address);
	IpAddress->sin_port = WH2N(Port);
	IpAddress->in_addr  = DH2N(Address);
	Irp = TdiBuildInternalDeviceControlIrp(
		TDI_SEND_DATAGRAM,   /* Sub function */
		DeviceObject,        /* Device object */
		TransportObject,     /* File object */
		NULL,                /* Event */
		NULL);               /* Return buffer */

	if (!Irp)
		{
			TDI_DbgPrint(MIN_TRACE, ("TdiBuildInternalDeviceControlIrp() failed.\n"));
			ExFreePool(ConnectInfo);
			return STATUS_INSUFFICIENT_RESOURCES;
		}

	Mdl = IoAllocateMdl(
		Buffer,     /* Virtual address of buffer */
		BufferSize, /* Length of buffer */
		FALSE,      /* Not secondary */
		FALSE,      /* Don't charge quota */
		NULL);      /* Don't use IRP */

	if (!Mdl)
		{
			TDI_DbgPrint(MIN_TRACE, ("IoAllocateMdl() failed.\n"));
			IoFreeIrp(Irp);
			ExFreePool(ConnectInfo);
			return STATUS_INSUFFICIENT_RESOURCES;
		}

	_SEH2_TRY
	{
		MmProbeAndLockPages(Mdl, KernelMode, IoModifyAccess);
	}
	_SEH2_EXCEPT(EXCEPTION_EXECUTE_HANDLER)
	{
		TDI_DbgPrint(MIN_TRACE, ("MmProbeAndLockPages() failed.\n"));
		IoFreeMdl(Mdl);
		IoFreeIrp(Irp);
		ExFreePool(ConnectInfo);
		_SEH2_YIELD(return STATUS_UNSUCCESSFUL);
	} _SEH2_END;

	TdiBuildSendDatagram(
		Irp,               /* I/O Request Packet */
		DeviceObject,      /* Device object */
		TransportObject,   /* File object */
		NULL,              /* Completion routine */
		NULL,              /* Completion context */
		Mdl,               /* Descriptor for data buffer */
		BufferSize,        /* Size of data to send */
		ConnectInfo);      /* Connection information */

	Status = TdiCall(Irp, DeviceObject, &Iosb, FALSE);

	ExFreePool(ConnectInfo);

	return Status;
}


NTSTATUS TdiReceiveDatagram(
    PFILE_OBJECT TransportObject,
    USHORT Port,
    PULONG Address,
    PUCHAR Buffer,
    PULONG BufferSize)
/*
 * FUNCTION: Receives a datagram
 * ARGUMENTS:
 *     TransportObject = Pointer to transport object
 *     Port            = Port to receive on
 *     Address         = Address of buffer to place remote address
 *     Buffer          = Address of buffer to place received data
 *     BufferSize      = Address of buffer with length of Buffer (updated)
 * RETURNS:
 *     Status of operation
 */
{
	PTDI_CONNECTION_INFORMATION ReceiveInfo;
	PTDI_CONNECTION_INFORMATION ReturnInfo;
	PTA_IP_ADDRESS ReturnAddress;
	PDEVICE_OBJECT DeviceObject;
	PTDI_ADDRESS_IP IpAddress;
	IO_STATUS_BLOCK Iosb;
	PVOID MdlBuffer;
	NTSTATUS Status;
	PIRP Irp;
	PMDL Mdl;

	DeviceObject = IoGetRelatedDeviceObject(TransportObject);
	if (!DeviceObject)
		return STATUS_INVALID_PARAMETER;

	ReceiveInfo = (PTDI_CONNECTION_INFORMATION) ExAllocatePool(NonPagedPool,
		sizeof(TDI_CONNECTION_INFORMATION) +
		sizeof(TDI_CONNECTION_INFORMATION) +
		sizeof(TA_IP_ADDRESS));

	if (!ReceiveInfo)
		return STATUS_INSUFFICIENT_RESOURCES;

	MdlBuffer = ExAllocatePool(PagedPool, *BufferSize);
	if (!MdlBuffer)
		return STATUS_INSUFFICIENT_RESOURCES;

	RtlZeroMemory(ReceiveInfo, sizeof(TDI_CONNECTION_INFORMATION) + sizeof(TDI_CONNECTION_INFORMATION) +
		sizeof(TA_IP_ADDRESS));

	RtlCopyMemory(MdlBuffer, Buffer, *BufferSize);

	/* Receive from any address */
	ReceiveInfo->RemoteAddressLength = 0;
	ReceiveInfo->RemoteAddress       = NULL;

	ReturnInfo = (PTDI_CONNECTION_INFORMATION) ((PUCHAR)ReceiveInfo + sizeof(TDI_CONNECTION_INFORMATION));
	ReturnInfo->RemoteAddressLength = sizeof(TA_IP_ADDRESS);
	ReturnInfo->RemoteAddress       = ((PUCHAR)ReturnInfo + sizeof(TDI_CONNECTION_INFORMATION));

	ReturnAddress = (PTA_IP_ADDRESS)(ReturnInfo->RemoteAddress);
	ReturnAddress->TAAddressCount           = 1;
	ReturnAddress->Address[0].AddressLength = sizeof(TDI_ADDRESS_IP);
	ReturnAddress->Address[0].AddressType   = TDI_ADDRESS_TYPE_IP;

	IpAddress = (PTDI_ADDRESS_IP)(ReturnAddress->Address[0].Address);
	IpAddress->sin_port = WH2N(Port);
	IpAddress->in_addr  = DH2N(LocalAddress);

	Irp = TdiBuildInternalDeviceControlIrp(
		TDI_RECEIVE_DATAGRAM,    /* Sub function */
		DeviceObject,            /* Device object */
		TransportObject,         /* File object */
		NULL,                    /* Event */
		NULL);                   /* Return buffer */

	if (!Irp)
		{
			ExFreePool(MdlBuffer);
			ExFreePool(ReceiveInfo);
			return STATUS_INSUFFICIENT_RESOURCES;
		}

	Mdl = IoAllocateMdl(
		MdlBuffer,      /* Virtual address */
		*BufferSize,    /* Length of buffer */
		FALSE,          /* Not secondary */
		FALSE,          /* Don't charge quota */
		NULL);          /* Don't use IRP */

	if (!Mdl)
		{
			IoFreeIrp(Irp);
			ExFreePool(MdlBuffer);
			ExFreePool(ReceiveInfo);
			return STATUS_INSUFFICIENT_RESOURCES;
		}

	_SEH2_TRY
	{
		MmProbeAndLockPages(Mdl, KernelMode, IoModifyAccess);
	}
	_SEH2_EXCEPT(EXCEPTION_EXECUTE_HANDLER)
	{
		TDI_DbgPrint(MIN_TRACE, ("MmProbeAndLockPages() failed.\n"));
		IoFreeMdl(Mdl);
		IoFreeIrp(Irp);
		ExFreePool(MdlBuffer);
		ExFreePool(ReceiveInfo);
		_SEH2_YIELD(return STATUS_INSUFFICIENT_RESOURCES);
	} _SEH2_END;

	TdiBuildReceiveDatagram(
		Irp,                    /* I/O Request Packet */
		DeviceObject,           /* Device object */
		TransportObject,        /* File object */
		NULL,                   /* Completion routine */
		NULL,                   /* Completion context */
		Mdl,                    /* Data buffer */
		*BufferSize,            /* Size of data buffer */
		ReceiveInfo,            /* Connection information */
		ReturnInfo,             /* Connection information */
		TDI_RECEIVE_NORMAL);    /* Flags */

	Status = TdiCall(Irp, DeviceObject, &Iosb, TRUE);

	if (NT_SUCCESS(Status))
		{
			RtlCopyMemory(Buffer, MdlBuffer, Iosb.Information);
			*BufferSize = Iosb.Information;
			*Address    = DN2H(IpAddress->in_addr);
		}

	ExFreePool(MdlBuffer);
	ExFreePool(ReceiveInfo);

	return Status;
}


VOID TdiSendThread(
    PVOID Context)
/*
 * FUNCTION: Send thread
 * ARGUMENTS:
 *     Context = Pointer to context information
 * NOTES:
 *     Transmits an UDP packet every two seconds to ourselves on the chosen port
 */
{
	KEVENT Event;
	PKEVENT Events[2];
	LARGE_INTEGER Timeout;
	NTSTATUS Status = STATUS_SUCCESS;
	UCHAR Data[40]  = "Testing one, two, three, ...";

	if (!OpenError)
		{
			Timeout.QuadPart = 10000000L;           /* Second factor */
			Timeout.QuadPart *= 2;                  /* Number of seconds */
			Timeout.QuadPart = -(Timeout.QuadPart); /* Relative time */

			KeInitializeEvent(&Event, SynchronizationEvent, FALSE);

			Events[0] = &StopEvent;
			Events[1] = &Event;

			while (NT_SUCCESS(Status))
				{
					/* Wait until timeout or stop flag is set */
 					KeWaitForMultipleObjects( 2, (PVOID)Events, WaitAny, Executive, KernelMode, FALSE, &Timeout, NULL);

					if (KeReadStateEvent(&StopEvent) != 0)
						{
							TDI_DbgPrint(MAX_TRACE, ("Received terminate signal...\n"));
							break;
						}

					DbgPrint("Sending data - '%s'\n", Data);

					Status = TdiSendDatagram(TdiTransportObject, TEST_PORT, LocalAddress, Data, sizeof(Data));

					if (!NT_SUCCESS(Status))
						DbgPrint("Failed sending data (Status = 0x%X)\n", Status);
				}
		}

	TDI_DbgPrint(MAX_TRACE, ("Terminating send thread...\n"));

	PsTerminateSystemThread(STATUS_SUCCESS);
}


VOID TdiReceiveThread(
    PVOID Context)
/*
 * FUNCTION: Receive thread
 * ARGUMENTS:
 *     Context = Pointer to context information
 * NOTES:
 *     Waits until an UDP packet is received on the chosen endpoint and displays the data
 */
{
	ULONG Address;
	UCHAR Data[40];
	ULONG Size;
	NTSTATUS Status = STATUS_SUCCESS;

	if (!OpenError)
		{
			while (NT_SUCCESS(Status))
				{
					Size = sizeof(Data);
					RtlZeroMemory(Data, Size);

					Status = TdiReceiveDatagram(TdiTransportObject, TEST_PORT, &Address, Data, &Size);

					if (NT_SUCCESS(Status))
						{
							DbgPrint("Received data - '%s'\n", Data);
						}
					else
						if (Status != STATUS_CANCELLED)
							{
								TDI_DbgPrint(MIN_TRACE, ("Receive error (Status = 0x%X).\n", Status));
							}
						else
							{
								TDI_DbgPrint(MAX_TRACE, ("IRP was cancelled.\n"));
							}
				}
		}

	TDI_DbgPrint(MAX_TRACE, ("Terminating receive thread...\n"));

	PsTerminateSystemThread(STATUS_SUCCESS);
}


VOID TdiOpenThread(
    PVOID Context)
/*
 * FUNCTION: Open thread
 * ARGUMENTS:
 *     Context = Pointer to context information (event)
 */
{
	NTSTATUS Status;

	TDI_DbgPrint(MAX_TRACE, ("Called.\n"));

	OpenError = TRUE;

	Status = TdiOpenTransport(UDP_DEVICE_NAME, TEST_PORT, &TdiTransport, &TdiTransportObject);

	if (NT_SUCCESS(Status))
		{
			Status = TdiQueryAddress(TdiTransportObject, &LocalAddress);

			if (NT_SUCCESS(Status))
				{
					OpenError = FALSE;
					DbgPrint("Using local IP address 0x%X\n", LocalAddress);
				}
			else
				{
					TDI_DbgPrint(MIN_TRACE, ("Unable to determine local IP address.\n"));
				}
			}
	else
		TDI_DbgPrint(MIN_TRACE, ("Cannot open transport (Status = 0x%X).\n", Status));

	TDI_DbgPrint(MAX_TRACE, ("Setting close event.\n"));

	KeSetEvent((PKEVENT)Context, 0, FALSE);

	TDI_DbgPrint(MIN_TRACE, ("Leaving.\n"));
}


VOID TdiUnload(
    PDRIVER_OBJECT DriverObject)
/*
 * FUNCTION: Unload routine
 * ARGUMENTS:
 *     DriverObject = Pointer to a driver object for this driver
 */
{
	PVOID ReceiveThreadObject = 0;
	PVOID SendThreadObject = 0;

	TDI_DbgPrint(MAX_TRACE, ("Setting stop flag\n"));

	/* Get pointers to the thread objects */
	ObReferenceObjectByHandle(SendThread, THREAD_ALL_ACCESS, NULL, KernelMode, &SendThreadObject, NULL);
	ObReferenceObjectByHandle(ReceiveThread, THREAD_ALL_ACCESS, NULL, KernelMode, &ReceiveThreadObject, NULL);

	KeSetEvent(&StopEvent, 0, FALSE);

	/* Wait for send thread to stop */
	KeWaitForSingleObject(SendThreadObject, Executive, KernelMode, FALSE, NULL);

	/* Wait for receive thread to stop */
	KeWaitForSingleObject(ReceiveThreadObject, Executive, KernelMode, FALSE, NULL);

	/* Close device */
	TdiCloseDevice(TdiTransport, TdiTransportObject);
}


NTSTATUS
NTAPI
DriverEntry(
    PDRIVER_OBJECT DriverObject,
    PUNICODE_STRING RegistryPath)
/*
 * FUNCTION: Main driver entry point
 * ARGUMENTS:
 *     DriverObject = Pointer to a driver object for this driver
 *     RegistryPath = Registry node for configuration parameters
 * RETURNS:
 *     Status of driver initialization
 */
{
	KEVENT Event;
	NTSTATUS Status;
	WORK_QUEUE_ITEM WorkItem;

	KeInitializeEvent(&StopEvent, NotificationEvent, FALSE);

	/* Call TdiOpenThread() */
	KeInitializeEvent(&Event, SynchronizationEvent, FALSE);
	ExInitializeWorkItem(&WorkItem, (PWORKER_THREAD_ROUTINE)TdiOpenThread, &Event);
 	ExQueueWorkItem(&WorkItem, DelayedWorkQueue);
	KeWaitForSingleObject(&Event, Executive, KernelMode, TRUE, NULL);

	/* Create a UDP send thread that sends a dgram every 2 seconds */
	Status = PsCreateSystemThread(
		&SendThread,                      /* Thread handle */
		0,                                /* Desired access */
		NULL,                             /* Object attributes */
		NULL,                             /* Process handle */
		NULL,                             /* Client id */
		(PKSTART_ROUTINE)TdiSendThread,   /* Start routine */
		NULL);                            /* Start context */

	if (!NT_SUCCESS(Status))
		{
			TDI_DbgPrint(MIN_TRACE, ("PsCreateSystemThread() failed for send thread (Status = 0x%X).\n", Status));
			return STATUS_INSUFFICIENT_RESOURCES;
		}

	/* Create a UDP receive thread */
	Status = PsCreateSystemThread(
		&ReceiveThread,                       /* Thread handle */
		0,                                    /* Desired access */
		NULL,                                 /* Object attributes */
		NULL,                                 /* Process handle */
		NULL,                                 /* Client id */
		(PKSTART_ROUTINE)TdiReceiveThread,    /* Start routine */
		NULL);                                /* Start context */

	if (!NT_SUCCESS(Status))
		{
			TDI_DbgPrint(MIN_TRACE, ("PsCreateSystemThread() failed for receive thread (Status = 0x%X).\n", Status));
			ZwClose(SendThread);
			return STATUS_INSUFFICIENT_RESOURCES;
		}

	DriverObject->DriverUnload = (PDRIVER_UNLOAD)TdiUnload;

	return STATUS_SUCCESS;
}

/* EOF */

