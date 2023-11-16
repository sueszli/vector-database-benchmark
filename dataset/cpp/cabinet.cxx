/*
 * COPYRIGHT:   See COPYING in the top level directory
 * PROJECT:     ReactOS cabinet manager
 * FILE:        tools/cabman/cabinet.cxx
 * PURPOSE:     Cabinet routines
 * PROGRAMMERS: Casper S. Hornstrup (chorns@users.sourceforge.net)
 *              Colin Finck <mail@colinfinck.de>
 * NOTES:       Define CAB_READ_ONLY for read only version
 * REVISIONS:
 *   CSH 21/03-2001 Created
 *   CSH 15/08-2003 Made it portable
 *   CF  04/05-2007 Made it compatible with 64-bit operating systems
 * TODO:
 *   - Checksum of datablocks should be calculated
 *   - EXTRACT.EXE complains if a disk is created manually
 *   - Folders that are created manually and span disks will result in a damaged cabinet
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
# include <dirent.h>
# include <sys/stat.h>
# include <sys/types.h>
#endif
#include "cabinet.h"
#include "CCFDATAStorage.h"
#include "raw.h"
#include "mszip.h"

#ifndef CAB_READ_ONLY

#if 0
#if DBG

void DumpBuffer(void* Buffer, ULONG Size)
{
    HANDLE FileHandle;
    ULONG BytesWritten;

    /* Create file, overwrite if it already exists */
    FileHandle = CreateFile("dump.bin", // Create this file
        GENERIC_WRITE,                  // Open for writing
        0,                              // No sharing
        NULL,                           // No security
        CREATE_ALWAYS,                  // Create or overwrite
        FILE_ATTRIBUTE_NORMAL,          // Normal file
        NULL);                          // No attribute template
    if (FileHandle == INVALID_HANDLE_VALUE)
    {
        DPRINT(MID_TRACE, ("ERROR OPENING '%u'.\n", (UINT)GetLastError()));
        return;
    }

    if (!WriteFile(FileHandle, Buffer, Size, &BytesWritten, NULL))
    {
        DPRINT(MID_TRACE, ("ERROR WRITING '%u'.\n", (UINT)GetLastError()));
    }

    CloseFile(FileHandle);
}

#endif /* DBG */
#endif

#endif /* CAB_READ_ONLY */


/* CCabinet */

CCabinet::CCabinet()
/*
 * FUNCTION: Default constructor
 */
{
    *CabinetName = '\0';
    *CabinetPrev = '\0';
    *DiskPrev = '\0';
    *CabinetNext = '\0';
    *DiskNext = '\0';

    FileOpen = false;
    CabinetReservedFileBuffer = NULL;
    CabinetReservedFileSize = 0;

    Codec          = NULL;
    CodecId        = -1;
    CodecSelected  = false;

    OutputBuffer = NULL;
    InputBuffer  = NULL;
    MaxDiskSize  = 0;
    BlockIsSplit = false;
    ScratchFile  = NULL;

    FolderUncompSize = 0;
    BytesLeftInBlock = 0;
    ReuseBlock       = false;
    CurrentDataNode  = NULL;
}


CCabinet::~CCabinet()
/*
 * FUNCTION: Default destructor
 */
{
    if (CabinetReservedFileBuffer != NULL)
    {
        free(CabinetReservedFileBuffer);
        CabinetReservedFileBuffer = NULL;
        CabinetReservedFileSize = 0;
    }

    if (CodecSelected)
        delete Codec;
}

bool CCabinet::IsSeparator(char Char)
/*
 * FUNCTION: Determines if a character is a separator
 * ARGUMENTS:
 *     Char = Character to check
 * RETURNS:
 *     Whether it is a separator
 */
{
    if ((Char == '\\') || (Char == '/'))
        return true;
    else
        return false;
}

void CCabinet::ConvertPath(std::string& Path)
/*
 * FUNCTION: Replaces \ or / with the one used by the host environment
 * ARGUMENTS:
 *     Path     = Pointer to string with pathname
 *     Allocate = Specifies whether to allocate memory for the new
 *                string or to change the existing buffer
 * RETURNS:
 *     Pointer to new path
 */
{
    for (size_t i = 0; i < Path.size(); ++i)
    {
#if defined(_WIN32)
        if (Path[i] == '/')
            Path[i] = '\\';
#else
        if (Path[i] == '\\')
            Path[i] = '/';
#endif
    }
}


std::string CCabinet::GetFileName(const std::string& Path)
/*
 * FUNCTION: Returns a pointer to file name
 * ARGUMENTS:
 *     Path = Pointer to string with pathname
 * RETURNS:
 *     Pointer to filename
 */
{
    ULONG i, j;

    j = i = (Path[0] ? (Path[1] == ':' ? 2 : 0) : 0);

    while (Path [i++])
        if (IsSeparator(Path [i - 1]))
            j = i;

    return Path.c_str() + j;
}


void CCabinet::NormalizePath(std::string& Path)
/*
 * FUNCTION: Normalizes a path
 * ARGUMENTS:
 *     Path   = string with pathname
 */
{
    if (Path.length() > 0)
    {
        if (!IsSeparator(Path[Path.length() - 1]))
            Path += DIR_SEPARATOR_CHAR;
    }
}


char* CCabinet::GetCabinetName()
/*
 * FUNCTION: Returns pointer to cabinet file name
 * RETURNS:
 *     Pointer to string with name of cabinet
 */
{
    return CabinetName;
}


void CCabinet::SetCabinetName(const char* FileName)
/*
 * FUNCTION: Sets cabinet file name
 * ARGUMENTS:
 *     FileName = Pointer to string with name of cabinet
 */
{
    strcpy(CabinetName, FileName);
}


void CCabinet::SetDestinationPath(const char* DestinationPath)
/*
 * FUNCTION: Sets destination path
 * ARGUMENTS:
 *    DestinationPath = Pointer to string with name of destination path
 */
{
    DestPath = DestinationPath;
    ConvertPath(DestPath);
    NormalizePath(DestPath);
}

ULONG CCabinet::AddSearchCriteria(const std::string& SearchCriteria, const std::string& TargetFolder)
/*
 * FUNCTION: Adds a criteria to the search criteria list
 * ARGUMENTS:
 *     SearchCriteria = String with the search criteria to add
 * RETURNS:
 *     Status of operation
 */
{
    PSEARCH_CRITERIA Criteria;

    // Add the criteria to the list of search criteria
    Criteria = new SEARCH_CRITERIA;
    if(!Criteria)
    {
        DPRINT(MIN_TRACE, ("Insufficient memory.\n"));
        return CAB_STATUS_NOMEMORY;
    }

    CriteriaList.push_back(Criteria);

    // Set the actual criteria string
    Criteria->Search = SearchCriteria;
    Criteria->TargetFolder = TargetFolder;

    return CAB_STATUS_SUCCESS;
}

void CCabinet::DestroySearchCriteria()
/*
 * FUNCTION: Destroys the list with the search criteria
 */
{
    for (PSEARCH_CRITERIA Criteria : CriteriaList)
    {
        delete Criteria;
    }
    CriteriaList.clear();
}

bool CCabinet::HasSearchCriteria()
/*
 * FUNCTION: Returns whether we have search criteria
 * RETURNS:
 *    Whether we have search criteria or not.
 */
{
    return !CriteriaList.empty();
}

std::string CCabinet::CreateCabFilename(PCFFILE_NODE Node)
{
    std::string fname = GetFileName(Node->FileName);
    if (!Node->TargetFolder.empty())
    {
        fname = Node->TargetFolder + fname;
    }
    return fname;
}

bool CCabinet::SetCompressionCodec(const char* CodecName)
/*
 * FUNCTION: Selects the codec to use for compression
 * ARGUMENTS:
 *    CodecName = Pointer to a string with the name of the codec
 */
{
    if( !strcasecmp(CodecName, "raw") )
        SelectCodec(CAB_CODEC_RAW);
    else if( !strcasecmp(CodecName, "mszip") )
        SelectCodec(CAB_CODEC_MSZIP);
    else
    {
        printf("ERROR: Invalid codec specified!\n");
        return false;
    }

    return true;
}

const char* CCabinet::GetDestinationPath()
/*
 * FUNCTION: Returns destination path
 * RETURNS:
 *    Pointer to string with name of destination path
 */
{
    return DestPath.c_str();
}


bool CCabinet::SetCabinetReservedFile(const char* FileName)
/*
 * FUNCTION: Sets cabinet reserved file
 * ARGUMENTS:
 *    FileName = Pointer to string with name of cabinet reserved file
 */
{
    FILE* FileHandle;
    ULONG BytesRead;
    std::string ConvertedFileName;

    ConvertedFileName = FileName;
    ConvertPath(ConvertedFileName);

    FileHandle = fopen(ConvertedFileName.c_str(), "rb");
    if (FileHandle == NULL)
    {
        DPRINT(MID_TRACE, ("Cannot open cabinet reserved file.\n"));
        return false;
    }

    CabinetReservedFileSize = GetSizeOfFile(FileHandle);
    if (CabinetReservedFileSize == (ULONG)-1)
    {
        DPRINT(MIN_TRACE, ("Cannot read from cabinet reserved file.\n"));
        fclose(FileHandle);
        return false;
    }

    if (CabinetReservedFileSize == 0)
    {
        fclose(FileHandle);
        return false;
    }

    CabinetReservedFileBuffer = malloc(CabinetReservedFileSize);
    if (!CabinetReservedFileBuffer)
    {
        fclose(FileHandle);
        return false;
    }

    BytesRead = fread(CabinetReservedFileBuffer, 1, CabinetReservedFileSize, FileHandle);
    if( BytesRead != CabinetReservedFileSize )
    {
        fclose(FileHandle);
        return false;
    }

    fclose(FileHandle);

    CabinetReservedFile = FileName;

    return true;
}


ULONG CCabinet::GetCurrentDiskNumber()
/*
 * FUNCTION: Returns current disk number
 * RETURNS:
 *     Current disk number
 */
{
    return CurrentDiskNumber;
}


ULONG CCabinet::Open()
/*
 * FUNCTION: Opens a cabinet file
 * RETURNS:
 *     Status of operation
 */
{
    ULONG Status;
    ULONG Index;

    if (!FileOpen)
    {
        ULONG BytesRead;
        ULONG Size;

        OutputBuffer = malloc(CAB_BLOCKSIZE + 12);    // This should be enough
        if (!OutputBuffer)
            return CAB_STATUS_NOMEMORY;

        FileHandle = fopen(CabinetName, "rb");
        if (FileHandle == NULL)
        {
            DPRINT(MID_TRACE, ("Cannot open file.\n"));
            return CAB_STATUS_CANNOT_OPEN;
        }

        FileOpen = true;

        /* Load CAB header */
        if ((Status = ReadBlock(&CABHeader, sizeof(CFHEADER), &BytesRead))
            != CAB_STATUS_SUCCESS)
        {
            DPRINT(MIN_TRACE, ("Cannot read from file (%u).\n", (UINT)Status));
            return CAB_STATUS_INVALID_CAB;
        }

        /* Check header */
        if ((BytesRead                 != sizeof(CFHEADER)) ||
            (CABHeader.Signature       != CAB_SIGNATURE   ) ||
            (CABHeader.Version         != CAB_VERSION     ) ||
            (CABHeader.FolderCount     == 0               ) ||
            (CABHeader.FileCount       == 0               ) ||
            (CABHeader.FileTableOffset < sizeof(CFHEADER)))
        {
            CloseCabinet();
            DPRINT(MID_TRACE, ("File has invalid header.\n"));
            return CAB_STATUS_INVALID_CAB;
        }

        Size = 0;

        /* Read/skip any reserved bytes */
        if (CABHeader.Flags & CAB_FLAG_RESERVE)
        {
            if ((Status = ReadBlock(&Size, sizeof(ULONG), &BytesRead))
                != CAB_STATUS_SUCCESS)
            {
                DPRINT(MIN_TRACE, ("Cannot read from file (%u).\n", (UINT)Status));
                return CAB_STATUS_INVALID_CAB;
            }
            CabinetReserved = Size & 0xFFFF;
            FolderReserved  = (Size >> 16) & 0xFF;
            DataReserved    = (Size >> 24) & 0xFF;

            if (fseek(FileHandle, CabinetReserved, SEEK_CUR) != 0)
            {
                DPRINT(MIN_TRACE, ("fseek() failed.\n"));
                return CAB_STATUS_FAILURE;
            }
        }

        if ((CABHeader.Flags & CAB_FLAG_HASPREV) > 0)
        {
            /* Read name of previous cabinet */
            Status = ReadString(CabinetPrev, 256);
            if (Status != CAB_STATUS_SUCCESS)
                return Status;
            /* Read label of previous disk */
            Status = ReadString(DiskPrev, 256);
            if (Status != CAB_STATUS_SUCCESS)
                return Status;
        }
        else
        {
            strcpy(CabinetPrev, "");
            strcpy(DiskPrev,    "");
        }

        if ((CABHeader.Flags & CAB_FLAG_HASNEXT) > 0)
        {
            /* Read name of next cabinet */
            Status = ReadString(CabinetNext, 256);
            if (Status != CAB_STATUS_SUCCESS)
                return Status;
            /* Read label of next disk */
            Status = ReadString(DiskNext, 256);
            if (Status != CAB_STATUS_SUCCESS)
                return Status;
        }
        else
        {
            strcpy(CabinetNext, "");
            strcpy(DiskNext,    "");
        }

        /* Read all folders */
        for (Index = 0; Index < CABHeader.FolderCount; Index++)
        {
            PCFFOLDER_NODE FolderNode = NewFolderNode();
            if (!FolderNode)
            {
                DPRINT(MIN_TRACE, ("Insufficient resources.\n"));
                return CAB_STATUS_NOMEMORY;
            }

            if (Index == 0)
                FolderNode->UncompOffset = FolderUncompSize;

            FolderNode->Index = Index;

            if ((Status = ReadBlock(&FolderNode->Folder,
                sizeof(CFFOLDER), &BytesRead)) != CAB_STATUS_SUCCESS)
            {
                DPRINT(MIN_TRACE, ("Cannot read from file (%u).\n", (UINT)Status));
                return CAB_STATUS_INVALID_CAB;
            }
        }

        /* Read file entries */
        Status = ReadFileTable();
        if (Status != CAB_STATUS_SUCCESS)
        {
            DPRINT(MIN_TRACE, ("ReadFileTable() failed (%u).\n", (UINT)Status));
            return Status;
        }

        /* Read data blocks for all folders */
        for (PCFFOLDER_NODE Node : FolderList)
        {
            Status = ReadDataBlocks(Node);
            if (Status != CAB_STATUS_SUCCESS)
            {
                DPRINT(MIN_TRACE, ("ReadDataBlocks() failed (%u).\n", (UINT)Status));
                return Status;
            }
        }
    }
    return CAB_STATUS_SUCCESS;
}


void CCabinet::Close()
/*
 * FUNCTION: Closes the cabinet file
 */
{
    if (FileOpen)
    {
        fclose(FileHandle);
        FileOpen = false;
    }
}


ULONG CCabinet::FindFirst(PCAB_SEARCH Search)
/*
 * FUNCTION: Finds the first file in the cabinet that matches a search criteria
 * ARGUMENTS:
 *     Search   = Pointer to search structure
 * RETURNS:
 *     Status of operation
 */
{
    RestartSearch = false;
    Search->Next = FileList.begin();
    return FindNext(Search);
}


ULONG CCabinet::FindNext(PCAB_SEARCH Search)
/*
 * FUNCTION: Finds next file in the cabinet that matches a search criteria
 * ARGUMENTS:
 *     Search = Pointer to search structure
 * RETURNS:
 *     Status of operation
 */
{
    bool bFound = false;
    ULONG Status;

    if (RestartSearch)
    {
        Search->Next  = FileList.begin();

        /* Skip split files already extracted */
        while ((Search->Next != FileList.end()) &&
            ((*Search->Next)->File.FileControlID > CAB_FILE_MAX_FOLDER) &&
            ((*Search->Next)->File.FileOffset <= LastFileOffset))
        {
            DPRINT(MAX_TRACE, ("Skipping file (%s)  FileOffset (0x%X)  LastFileOffset (0x%X).\n",
                (*Search->Next)->FileName.c_str(), (UINT)(*Search->Next)->File.FileOffset, (UINT)LastFileOffset));
            Search->Next++;
        }

        RestartSearch = false;
    }

    /* Check each search criteria against each file */
    while(Search->Next != FileList.end())
    {
        // Some features (like displaying cabinets) don't require search criteria, so we can just break here.
        // If a feature requires it, handle this in the ParseCmdline() function in "main.cxx".
        if (CriteriaList.empty())
            break;

        for (PSEARCH_CRITERIA Criteria : CriteriaList)
        {
            // FIXME: We could handle path\filename here
            if (MatchFileNamePattern((*Search->Next)->FileName.c_str(), Criteria->Search.c_str()))
            {
                bFound = true;
                break;
            }
        }

        if(bFound)
            break;

        Search->Next++;
    }

    if (Search->Next == FileList.end())
    {
        if (strlen(DiskNext) > 0)
        {
            CloseCabinet();

            SetCabinetName(CabinetNext);

            OnDiskChange(CabinetNext, DiskNext);

            Status = Open();
            if (Status != CAB_STATUS_SUCCESS)
                return Status;

            Search->Next = FileList.begin();
            if (Search->Next == FileList.end())
                return CAB_STATUS_NOFILE;
        }
        else
            return CAB_STATUS_NOFILE;
    }

    Search->File     = &(*Search->Next)->File;
    Search->FileName = (*Search->Next)->FileName;
    Search->Next++;
    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::ExtractFile(const char* FileName)
/*
 * FUNCTION: Extracts a file from the cabinet
 * ARGUMENTS:
 *     FileName = Pointer to buffer with name of file
 * RETURNS
 *     Status of operation
 */
{
    ULONG Size;
    ULONG Offset;
    ULONG BytesRead;
    ULONG BytesToRead;
    ULONG BytesWritten;
    ULONG BytesSkipped;
    ULONG BytesToWrite;
    ULONG TotalBytesRead;
    ULONG CurrentOffset;
    PUCHAR Buffer;
    PUCHAR CurrentBuffer;
    FILE* DestFile;
    PCFFILE_NODE File;
    CFDATA CFData;
    ULONG Status;
    bool Skip;
#if defined(_WIN32)
    FILETIME FileTime;
#endif
    CHAR DestName[PATH_MAX];
    CHAR TempName[PATH_MAX];

    Status = LocateFile(FileName, &File);
    if (Status != CAB_STATUS_SUCCESS)
    {
        DPRINT(MID_TRACE, ("Cannot locate file (%u).\n", (UINT)Status));
        return Status;
    }

    LastFileOffset = File->File.FileOffset;

    switch (CurrentFolderNode->Folder.CompressionType & CAB_COMP_MASK)
    {
        case CAB_COMP_NONE:
            SelectCodec(CAB_CODEC_RAW);
            break;

        case CAB_COMP_MSZIP:
            SelectCodec(CAB_CODEC_MSZIP);
            break;

        default:
            return CAB_STATUS_UNSUPPCOMP;
    }

    DPRINT(MAX_TRACE, ("Extracting file at uncompressed offset (0x%X)  Size (%u bytes)  AO (0x%X)  UO (0x%X).\n",
        (UINT)File->File.FileOffset,
        (UINT)File->File.FileSize,
        (UINT)File->DataBlock->AbsoluteOffset,
        (UINT)File->DataBlock->UncompOffset));

    strcpy(DestName, DestPath.c_str());
    strcat(DestName, FileName);

    /* Create destination file, fail if it already exists */
    DestFile = fopen(DestName, "rb");
    if (DestFile != NULL)
    {
        fclose(DestFile);
        /* If file exists, ask to overwrite file */
        if (OnOverwrite(&File->File, FileName))
        {
            DestFile = fopen(DestName, "w+b");
            if (DestFile == NULL)
                return CAB_STATUS_CANNOT_CREATE;
        }
        else
            return CAB_STATUS_FILE_EXISTS;
    }
    else
    {
        DestFile = fopen(DestName, "w+b");
        if (DestFile == NULL)
            return CAB_STATUS_CANNOT_CREATE;
    }

#if defined(_WIN32)
    if (!DosDateTimeToFileTime(File->File.FileDate, File->File.FileTime, &FileTime))
    {
        fclose(DestFile);
        DPRINT(MIN_TRACE, ("DosDateTimeToFileTime() failed (%u).\n", (UINT)GetLastError()));
        return CAB_STATUS_CANNOT_WRITE;
    }

    SetFileTime(DestFile, NULL, &FileTime, NULL);
#else
    //DPRINT(MIN_TRACE, ("FIXME: DosDateTimeToFileTime\n"));
#endif

    SetAttributesOnFile(DestName, File->File.Attributes);

    Buffer = (PUCHAR)malloc(CAB_BLOCKSIZE + 12); // This should be enough
    if (!Buffer)
    {
        fclose(DestFile);
        DPRINT(MIN_TRACE, ("Insufficient memory.\n"));
        return CAB_STATUS_NOMEMORY;
    }

    /* Call OnExtract event handler */
    OnExtract(&File->File, FileName);

    /* Search to start of file */
    if (fseek(FileHandle, (off_t)File->DataBlock->AbsoluteOffset, SEEK_SET) != 0)
    {
        DPRINT(MIN_TRACE, ("fseek() failed.\n"));
        fclose(DestFile);
        free(Buffer);
        return CAB_STATUS_INVALID_CAB;
    }

    Size   = File->File.FileSize;
    Offset = File->File.FileOffset;
    CurrentOffset = File->DataBlock->UncompOffset;

    Skip = true;

    ReuseBlock = (CurrentDataNode == File->DataBlock);
    if (Size > 0)
    {
        do
        {
            DPRINT(MAX_TRACE, ("CO (0x%X)    ReuseBlock (%u)    Offset (0x%X)   Size (%d)  BytesLeftInBlock (%d)\n",
                (UINT)File->DataBlock->UncompOffset, (UINT)ReuseBlock, (UINT)Offset, (UINT)Size,
                (UINT)BytesLeftInBlock));

            if (/*(CurrentDataNode != File->DataBlock) &&*/ (!ReuseBlock) || (BytesLeftInBlock <= 0))
            {
                DPRINT(MAX_TRACE, ("Filling buffer. ReuseBlock (%u)\n", (UINT)ReuseBlock));

                CurrentBuffer  = Buffer;
                TotalBytesRead = 0;
                do
                {
                    DPRINT(MAX_TRACE, ("Size (%u bytes).\n", (UINT)Size));

                    if (((Status = ReadBlock(&CFData, sizeof(CFDATA), &BytesRead)) !=
                        CAB_STATUS_SUCCESS) || (BytesRead != sizeof(CFDATA)))
                    {
                        fclose(DestFile);
                        free(Buffer);
                        DPRINT(MIN_TRACE, ("Cannot read from file (%u).\n", (UINT)Status));
                        return CAB_STATUS_INVALID_CAB;
                    }

                    DPRINT(MAX_TRACE, ("Data block: Checksum (0x%X)  CompSize (%u bytes)  UncompSize (%u bytes)\n",
                        (UINT)CFData.Checksum,
                        CFData.CompSize,
                        CFData.UncompSize));

                    ASSERT(CFData.CompSize <= CAB_BLOCKSIZE + 12);

                    BytesToRead = CFData.CompSize;

                    DPRINT(MAX_TRACE, ("Read: (0x%lX,0x%lX).\n",
                        (unsigned long)CurrentBuffer, (unsigned long)Buffer));

                    if (((Status = ReadBlock(CurrentBuffer, BytesToRead, &BytesRead)) !=
                        CAB_STATUS_SUCCESS) || (BytesToRead != BytesRead))
                    {
                        fclose(DestFile);
                        free(Buffer);
                        DPRINT(MIN_TRACE, ("Cannot read from file (%u).\n", (UINT)Status));
                        return CAB_STATUS_INVALID_CAB;
                    }

                    /* FIXME: Does not work with files generated by makecab.exe */
/*
                    if (CFData.Checksum != 0)
                    {
                        ULONG Checksum = ComputeChecksum(CurrentBuffer, BytesRead, 0);
                        if (Checksum != CFData.Checksum)
                        {
                            CloseFile(DestFile);
                            free(Buffer);
                            DPRINT(MIN_TRACE, ("Bad checksum (is 0x%X, should be 0x%X).\n",
                                Checksum, CFData.Checksum));
                            return CAB_STATUS_INVALID_CAB;
                        }
                    }
*/
                    TotalBytesRead += BytesRead;

                    CurrentBuffer += BytesRead;

                    if (CFData.UncompSize == 0)
                    {
                        if (strlen(DiskNext) == 0)
                        {
                            fclose(DestFile);
                            free(Buffer);
                            return CAB_STATUS_NOFILE;
                        }

                        /* CloseCabinet() will destroy all file entries so in case
                           FileName refers to the FileName field of a CFFOLDER_NODE
                           structure, we have to save a copy of the filename */
                        strcpy(TempName, FileName);

                        CloseCabinet();

                        SetCabinetName(CabinetNext);

                        OnDiskChange(CabinetNext, DiskNext);

                        Status = Open();
                        if (Status != CAB_STATUS_SUCCESS)
                        {
                            fclose(DestFile);
                            free(Buffer);
                            return Status;
                        }

                        /* The first data block of the file will not be
                           found as it is located in the previous file */
                        Status = LocateFile(TempName, &File);
                        if (Status == CAB_STATUS_NOFILE)
                        {
                            DPRINT(MID_TRACE, ("Cannot locate file (%u).\n", (UINT)Status));
                            fclose(DestFile);
                            free(Buffer);
                            return Status;
                        }

                        /* The file is continued in the first data block in the folder */
                        File->DataBlock = CurrentFolderNode->DataList.front();

                        /* Search to start of file */
                        if (fseek(FileHandle, (off_t)File->DataBlock->AbsoluteOffset, SEEK_SET) != 0)
                        {
                            DPRINT(MIN_TRACE, ("fseek() failed.\n"));
                            fclose(DestFile);
                            free(Buffer);
                            return CAB_STATUS_INVALID_CAB;
                        }

                        DPRINT(MAX_TRACE, ("Continuing extraction of file at uncompressed offset (0x%X)  Size (%u bytes)  AO (0x%X)  UO (0x%X).\n",
                            (UINT)File->File.FileOffset,
                            (UINT)File->File.FileSize,
                            (UINT)File->DataBlock->AbsoluteOffset,
                            (UINT)File->DataBlock->UncompOffset));

                        CurrentDataNode = File->DataBlock;
                        ReuseBlock = true;

                        RestartSearch = true;
                    }
                } while (CFData.UncompSize == 0);

                DPRINT(MAX_TRACE, ("TotalBytesRead (%u).\n", (UINT)TotalBytesRead));

                Status = Codec->Uncompress(OutputBuffer, Buffer, TotalBytesRead, &BytesToWrite);
                if (Status != CS_SUCCESS)
                {
                    fclose(DestFile);
                    free(Buffer);
                    DPRINT(MID_TRACE, ("Cannot uncompress block.\n"));
                    if (Status == CS_NOMEMORY)
                        return CAB_STATUS_NOMEMORY;
                    return CAB_STATUS_INVALID_CAB;
                }

                if (BytesToWrite != CFData.UncompSize)
                {
                    DPRINT(MID_TRACE, ("BytesToWrite (%u) != CFData.UncompSize (%d)\n",
                        (UINT)BytesToWrite, CFData.UncompSize));
                    fclose(DestFile);
                    free(Buffer);
                    return CAB_STATUS_INVALID_CAB;
                }

                BytesLeftInBlock = BytesToWrite;
            }
            else
            {
                DPRINT(MAX_TRACE, ("Using same buffer. ReuseBlock (%u)\n", (UINT)ReuseBlock));

                BytesToWrite = BytesLeftInBlock;

                DPRINT(MAX_TRACE, ("Seeking to absolute offset 0x%X.\n",
                    (UINT)(CurrentDataNode->AbsoluteOffset + sizeof(CFDATA) + CurrentDataNode->Data.CompSize)));

                if (((Status = ReadBlock(&CFData, sizeof(CFDATA), &BytesRead)) !=
                    CAB_STATUS_SUCCESS) || (BytesRead != sizeof(CFDATA)))
                {
                    fclose(DestFile);
                    free(Buffer);
                    DPRINT(MIN_TRACE, ("Cannot read from file (%u).\n", (UINT)Status));
                    return CAB_STATUS_INVALID_CAB;
                }

                DPRINT(MAX_TRACE, ("CFData.CompSize 0x%X  CFData.UncompSize 0x%X.\n",
                    CFData.CompSize, CFData.UncompSize));

                /* Go to next data block */
                if (fseek(FileHandle, (off_t)CurrentDataNode->AbsoluteOffset + sizeof(CFDATA) +
                    CurrentDataNode->Data.CompSize, SEEK_SET) != 0)
                {
                    DPRINT(MIN_TRACE, ("fseek() failed.\n"));
                    fclose(DestFile);
                    free(Buffer);
                    return CAB_STATUS_INVALID_CAB;
                }

                ReuseBlock = false;
            }

            if (Skip)
                BytesSkipped = (Offset - CurrentOffset);
            else
                BytesSkipped = 0;

            BytesToWrite -= BytesSkipped;

            if (Size < BytesToWrite)
                BytesToWrite = Size;

            DPRINT(MAX_TRACE, ("Offset (0x%X)  CurrentOffset (0x%X)  ToWrite (%u)  Skipped (%u)(%u)  Size (%u).\n",
                (UINT)Offset,
                (UINT)CurrentOffset,
                (UINT)BytesToWrite,
                (UINT)BytesSkipped, (UINT)Skip,
                (UINT)Size));

            BytesWritten = BytesToWrite;
            if (fwrite((void*)((PUCHAR)OutputBuffer + BytesSkipped),
                 BytesToWrite, 1, DestFile) < 1)
            {
                fclose(DestFile);
                free(Buffer);
                DPRINT(MIN_TRACE, ("Cannot write to file.\n"));

                return CAB_STATUS_CANNOT_WRITE;
            }

            Size -= BytesToWrite;

            CurrentOffset += BytesToWrite;

            /* Don't skip any more bytes */
            Skip = false;
        } while (Size > 0);
    }

    fclose(DestFile);

    free(Buffer);

    return CAB_STATUS_SUCCESS;
}

bool CCabinet::IsCodecSelected()
/*
 * FUNCTION: Returns the value of CodecSelected
 * RETURNS:
 *     Whether a codec is selected
 */
{
    return CodecSelected;
}

void CCabinet::SelectCodec(LONG Id)
/*
 * FUNCTION: Selects codec engine to use
 * ARGUMENTS:
 *     Id = Codec identifier
 */
{
    if (CodecSelected)
    {
        if (Id == CodecId)
            return;

        CodecSelected = false;
        delete Codec;
    }

    switch (Id)
    {
        case CAB_CODEC_RAW:
            Codec = new CRawCodec();
            break;

        case CAB_CODEC_MSZIP:
            Codec = new CMSZipCodec();
            break;

        default:
            return;
    }

    CodecId       = Id;
    CodecSelected = true;
}


#ifndef CAB_READ_ONLY

/* CAB write methods */

ULONG CCabinet::NewCabinet()
/*
 * FUNCTION: Creates a new cabinet
 * RETURNS:
 *     Status of operation
 */
{
    ULONG Status;

    CurrentDiskNumber = 0;

    OutputBuffer = malloc(CAB_BLOCKSIZE + 12); // This should be enough
    InputBuffer  = malloc(CAB_BLOCKSIZE + 12); // This should be enough
    if ((!OutputBuffer) || (!InputBuffer))
    {
        DPRINT(MIN_TRACE, ("Insufficient memory.\n"));
        return CAB_STATUS_NOMEMORY;
    }
    CurrentIBuffer     = InputBuffer;
    CurrentIBufferSize = 0;

    CABHeader.Signature     = CAB_SIGNATURE;
    CABHeader.Reserved1     = 0;            // Not used
    CABHeader.CabinetSize   = 0;            // Not yet known
    CABHeader.Reserved2     = 0;            // Not used
    CABHeader.Reserved3     = 0;            // Not used
    CABHeader.Version       = CAB_VERSION;
    CABHeader.FolderCount   = 0;            // Not yet known
    CABHeader.FileCount     = 0;            // Not yet known
    CABHeader.Flags         = 0;            // Not yet known
    // FIXME: Should be random
    CABHeader.SetID         = 0x534F;
    CABHeader.CabinetNumber = 0;


    TotalFolderSize = 0;
    TotalFileSize   = 0;

    DiskSize = sizeof(CFHEADER);

    InitCabinetHeader();

    // NextFolderNumber is 0-based
    NextFolderNumber = 0;

    CurrentFolderNode = NULL;
    Status = NewFolder();
    if (Status != CAB_STATUS_SUCCESS)
        return Status;

    CurrentFolderNode->Folder.DataOffset = DiskSize - TotalHeaderSize;

    ScratchFile = new CCFDATAStorage;
    if (!ScratchFile)
    {
        DPRINT(MIN_TRACE, ("Insufficient memory.\n"));
        return CAB_STATUS_NOMEMORY;
    }

    Status = ScratchFile->Create();

    CreateNewFolder = false;

    CreateNewDisk = false;

    PrevCabinetNumber = 0;

    return Status;
}


ULONG CCabinet::NewDisk()
/*
 * FUNCTION: Forces a new disk to be created
 * RETURNS:
 *     Status of operation
 */
{
    // NextFolderNumber is 0-based
    NextFolderNumber = 1;

    CreateNewDisk = false;

    DiskSize = sizeof(CFHEADER) + TotalFolderSize + TotalFileSize;

    InitCabinetHeader();

    CurrentFolderNode->TotalFolderSize = 0;

    CurrentFolderNode->Folder.DataBlockCount = 0;

    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::NewFolder()
/*
 * FUNCTION: Forces a new folder to be created
 * RETURNS:
 *     Status of operation
 */
{
    DPRINT(MAX_TRACE, ("Creating new folder.\n"));

    CurrentFolderNode = NewFolderNode();
    if (!CurrentFolderNode)
    {
        DPRINT(MIN_TRACE, ("Insufficient memory.\n"));
        return CAB_STATUS_NOMEMORY;
    }

    switch (CodecId) {
        case CAB_CODEC_RAW:
            CurrentFolderNode->Folder.CompressionType = CAB_COMP_NONE;
            break;

        case CAB_CODEC_MSZIP:
            CurrentFolderNode->Folder.CompressionType = CAB_COMP_MSZIP;
            break;

        default:
            return CAB_STATUS_UNSUPPCOMP;
    }

    /* FIXME: This won't work if no files are added to the new folder */

    DiskSize += sizeof(CFFOLDER);

    TotalFolderSize += sizeof(CFFOLDER);

    NextFolderNumber++;

    CABHeader.FolderCount++;

    LastBlockStart = 0;

    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::WriteFileToScratchStorage(PCFFILE_NODE FileNode)
/*
 * FUNCTION: Writes a file to the scratch file
 * ARGUMENTS:
 *     FileNode = Pointer to file node
 * RETURNS:
 *     Status of operation
 */
{
    ULONG BytesToRead;
    ULONG BytesRead;
    ULONG Status;
    ULONG Size;

    if (!ContinueFile)
    {
        /* Try to open file */
        SourceFile = fopen(FileNode->FileName.c_str(), "rb");
        if (SourceFile == NULL)
        {
            DPRINT(MID_TRACE, ("File not found (%s).\n", FileNode->FileNameOnDisk.c_str()));
            return CAB_STATUS_NOFILE;
        }

        if (CreateNewFolder)
        {
            /* There is always a new folder after
               a split file is completely stored */
            Status = NewFolder();
            if (Status != CAB_STATUS_SUCCESS)
                return Status;
            CreateNewFolder = false;
        }

        /* Call OnAdd event handler */
        OnAdd(&FileNode->File, FileNode->FileName.c_str());

        TotalBytesLeft = FileNode->File.FileSize;

        FileNode->File.FileOffset        = CurrentFolderNode->UncompOffset;
        CurrentFolderNode->UncompOffset += TotalBytesLeft;
        FileNode->File.FileControlID     = (USHORT)(NextFolderNumber - 1);
        CurrentFolderNode->Commit        = true;
        PrevCabinetNumber                = CurrentDiskNumber;

        Size = sizeof(CFFILE) + (ULONG)CreateCabFilename(FileNode).length() + 1;
        CABHeader.FileTableOffset += Size;
        TotalFileSize += Size;
        DiskSize += Size;
    }

    FileNode->Commit = true;

    if (TotalBytesLeft > 0)
    {
        do
        {
            if (TotalBytesLeft > (ULONG)CAB_BLOCKSIZE - CurrentIBufferSize)
                BytesToRead = CAB_BLOCKSIZE - CurrentIBufferSize;
            else
                BytesToRead = TotalBytesLeft;

            if ( (BytesRead = fread(CurrentIBuffer, 1, BytesToRead, SourceFile)) != BytesToRead )
            {
                DPRINT(MIN_TRACE, ("Cannot read from file. BytesToRead (%u)  BytesRead (%u)  CurrentIBufferSize (%u).\n",
                    (UINT)BytesToRead, (UINT)BytesRead, (UINT)CurrentIBufferSize));
                return CAB_STATUS_INVALID_CAB;
            }

            CurrentIBuffer = (unsigned char*)CurrentIBuffer + BytesRead;
            CurrentIBufferSize += (USHORT)BytesRead;

            if (CurrentIBufferSize == CAB_BLOCKSIZE)
            {
                Status = WriteDataBlock();
                if (Status != CAB_STATUS_SUCCESS)
                    return Status;
            }
            TotalBytesLeft -= BytesRead;
        } while ((TotalBytesLeft > 0) && (!CreateNewDisk));
    }

    if (TotalBytesLeft == 0)
    {
        fclose(SourceFile);
        FileNode->Delete = true;

        if (FileNode->File.FileControlID > CAB_FILE_MAX_FOLDER)
        {
            FileNode->File.FileControlID = CAB_FILE_CONTINUED;
            CurrentFolderNode->Delete = true;

            if ((CurrentIBufferSize > 0) || (CurrentOBufferSize > 0))
            {
                Status = WriteDataBlock();
                if (Status != CAB_STATUS_SUCCESS)
                    return Status;
            }

            CreateNewFolder = true;
        }
    }
    else
    {
        if (FileNode->File.FileControlID <= CAB_FILE_MAX_FOLDER)
            FileNode->File.FileControlID = CAB_FILE_SPLIT;
        else
            FileNode->File.FileControlID = CAB_FILE_PREV_NEXT;
    }

    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::WriteDisk(ULONG MoreDisks)
/*
 * FUNCTION: Forces the current disk to be written
 * ARGUMENTS:
 *     MoreDisks = true if there is one or more disks after this disk
 * RETURNS:
 *     Status of operation
 */
{
    ULONG Status;

    ContinueFile = false;
    for (auto it = FileList.begin(); it != FileList.end();)
    {
        Status = WriteFileToScratchStorage(*it);
        if (Status != CAB_STATUS_SUCCESS)
            return Status;

        if (CreateNewDisk)
        {
            /* A data block could span more than two
               disks if MaxDiskSize is very small */
            while (CreateNewDisk)
            {
                DPRINT(MAX_TRACE, ("Creating new disk.\n"));
                CommitDisk(true);
                CloseDisk();
                NewDisk();

                ContinueFile = true;
                CreateNewDisk = false;

                DPRINT(MAX_TRACE, ("First on new disk. CurrentIBufferSize (%u)  CurrentOBufferSize (%u).\n",
                    (UINT)CurrentIBufferSize, (UINT)CurrentOBufferSize));

                if ((CurrentIBufferSize > 0) || (CurrentOBufferSize > 0))
                {
                    Status = WriteDataBlock();
                    if (Status != CAB_STATUS_SUCCESS)
                        return Status;
                }
            }
        }
        else
        {
            ContinueFile = false;
            it++;
        }
    }

    if ((CurrentIBufferSize > 0) || (CurrentOBufferSize > 0))
    {
        /* A data block could span more than two
           disks if MaxDiskSize is very small */

        ASSERT(CreateNewDisk == false);

        do
        {
            if (CreateNewDisk)
            {
                DPRINT(MID_TRACE, ("Creating new disk 2.\n"));
                CommitDisk(true);
                CloseDisk();
                NewDisk();
                CreateNewDisk = false;
            }

            if ((CurrentIBufferSize > 0) || (CurrentOBufferSize > 0))
            {
                Status = WriteDataBlock();
                if (Status != CAB_STATUS_SUCCESS)
                    return Status;
            }
        } while (CreateNewDisk);
    }
    CommitDisk(MoreDisks);

    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::CommitDisk(ULONG MoreDisks)
/*
 * FUNCTION: Commits the current disk
 * ARGUMENTS:
 *     MoreDisks = true if there is one or more disks after this disk
 * RETURNS:
 *     Status of operation
 */
{
    ULONG Status;

    OnCabinetName(CurrentDiskNumber, CabinetName);

    /* Create file, fail if it already exists */
    FileHandle = fopen(CabinetName, "rb");
    if (FileHandle != NULL)
    {
        fclose(FileHandle);
        /* If file exists, ask to overwrite file */
        if (OnOverwrite(NULL, CabinetName))
        {
            FileHandle = fopen(CabinetName, "w+b");
            if (FileHandle == NULL)
                return CAB_STATUS_CANNOT_CREATE;
        }
        else
            return CAB_STATUS_FILE_EXISTS;

    }
    else
    {
        FileHandle = fopen(CabinetName, "w+b");
        if (FileHandle == NULL)
            return CAB_STATUS_CANNOT_CREATE;
    }

    WriteCabinetHeader(MoreDisks != 0);

    Status = WriteFolderEntries();
    if (Status != CAB_STATUS_SUCCESS)
        return Status;

    /* Write file entries */
    WriteFileEntries();

    /* Write data blocks */
    for (PCFFOLDER_NODE FolderNode : FolderList)
    {
        if (FolderNode->Commit)
        {
            Status = CommitDataBlocks(FolderNode);
            if (Status != CAB_STATUS_SUCCESS)
                return Status;
            /* Remove data blocks for folder */
            DestroyDataNodes(FolderNode);
        }
    }

    fclose(FileHandle);

    ScratchFile->Truncate();

    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::CloseDisk()
/*
 * FUNCTION: Closes the current disk
 * RETURNS:
 *     Status of operation
 */
{
    DestroyDeletedFileNodes();

    /* Destroy folder nodes that are completely stored */
    DestroyDeletedFolderNodes();

    CurrentDiskNumber++;

    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::CloseCabinet()
/*
 * FUNCTION: Closes the current cabinet
 * RETURNS:
 *     Status of operation
 */
{
    ULONG Status;

    DestroyFileNodes();

    DestroyFolderNodes();

    if (InputBuffer)
    {
        free(InputBuffer);
        InputBuffer = NULL;
    }

    if (OutputBuffer)
    {
        free(OutputBuffer);
        OutputBuffer = NULL;
    }

    Close();

    if (ScratchFile)
    {
        Status = ScratchFile->Destroy();
        delete ScratchFile;
        return Status;
    }

    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::AddFile(const std::string& FileName, const std::string& TargetFolder)
/*
 * FUNCTION: Adds a file to the current disk
 * ARGUMENTS:
 *     FileName = Pointer to string with file name (full path)
 * RETURNS:
 *     Status of operation
 */
{
    FILE* SrcFile;
    PCFFILE_NODE FileNode;
    std::string NewFileName;

    NewFileName = FileName;
    ConvertPath(NewFileName);

    /* Try to open file */
    SrcFile = fopen(NewFileName.c_str(), "rb");
    if (SrcFile == NULL)
    {
        DPRINT(MID_TRACE, ("File not found (%s).\n", NewFileName.c_str()));
        return CAB_STATUS_CANNOT_OPEN;
    }

    FileNode = NewFileNode();
    if (!FileNode)
    {
        DPRINT(MIN_TRACE, ("Insufficient memory.\n"));
        fclose(SrcFile);
        return CAB_STATUS_NOMEMORY;
    }

    FileNode->FolderNode = CurrentFolderNode;
    FileNode->FileName = NewFileName;
    FileNode->TargetFolder = TargetFolder;
    if (FileNode->TargetFolder.length() > 0 && FileNode->TargetFolder[FileNode->TargetFolder.length() - 1] != '\\')
        FileNode->TargetFolder += '\\';

    /* FIXME: Check for and handle large files (>= 2GB) */
    FileNode->File.FileSize = GetSizeOfFile(SrcFile);
    if (FileNode->File.FileSize == (ULONG)-1)
    {
        DPRINT(MIN_TRACE, ("Cannot read from file.\n"));
        fclose(SrcFile);
        return CAB_STATUS_CANNOT_READ;
    }

    if (GetFileTimes(SrcFile, FileNode) != CAB_STATUS_SUCCESS)
    {
        DPRINT(MIN_TRACE, ("Cannot read file times.\n"));
        fclose(SrcFile);
        return CAB_STATUS_CANNOT_READ;
    }

    if (GetAttributesOnFile(FileNode) != CAB_STATUS_SUCCESS)
    {
        DPRINT(MIN_TRACE, ("Cannot read file attributes.\n"));
        fclose(SrcFile);
        return CAB_STATUS_CANNOT_READ;
    }

    fclose(SrcFile);

    return CAB_STATUS_SUCCESS;
}

bool CCabinet::CreateSimpleCabinet()
/*
 * FUNCTION: Create a simple cabinet based on the files in the criteria list
 */
{
    bool bRet = false;
    ULONG Status;

#if defined(_WIN32)
    HANDLE hFind;
    WIN32_FIND_DATA FindFileData;
#else
    DIR* dirp;
    struct dirent* dp;
    struct stat stbuf;
#endif

    // Initialize a new cabinet
    Status = NewCabinet();
    if (Status != CAB_STATUS_SUCCESS)
    {
        DPRINT(MIN_TRACE, ("Cannot create cabinet (%u).\n", (UINT)Status));
        goto cleanup2;
    }

    // Add each file in the criteria list
    for (PSEARCH_CRITERIA Criteria : CriteriaList)
    {
        // Store the file path with a trailing slash in szFilePath
        std::string szSearchPath = Criteria->Search;
        ConvertPath(szSearchPath);
        auto sep = szSearchPath.find_last_of(DIR_SEPARATOR_CHAR);
        std::string szFilePath;
        std::string pszFile;

        if (sep != std::string::npos)
        {
            pszFile = szSearchPath.substr(sep + 1); // We want the filename, not the dir separator!

            szFilePath = szSearchPath.substr(0, sep + 1);
        }
        else
        {
            pszFile = Criteria->Search;

#if !defined(_WIN32)
            // needed for opendir()
            szFilePath = "./";
#endif
        }

#if defined(_WIN32)
        // Windows: Use the easy FindFirstFile/FindNextFile API for getting all files and checking them against the pattern
        hFind = FindFirstFile(Criteria->Search.c_str(), &FindFileData);

        // Don't stop if a search criteria is not found
        if(hFind == INVALID_HANDLE_VALUE && GetLastError() != ERROR_FILE_NOT_FOUND)
        {
            DPRINT(MIN_TRACE, ("FindFirstFile failed, Criteria: %s, error code is %u\n", Criteria->Search.c_str(), (UINT)GetLastError()));
            goto cleanup;
        }

        do
        {
            if(!(FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            {
                std::string szFile = szFilePath;
                szFile += FindFileData.cFileName;

                Status = AddFile(szFile, Criteria->TargetFolder);

                if(Status != CAB_STATUS_SUCCESS)
                {
                    DPRINT(MIN_TRACE, ("Cannot add file to cabinet (%u).\n", (UINT)Status));
                    FindClose(hFind);
                    goto cleanup;
                }
            }
        }
        while(FindNextFile(hFind, &FindFileData));

        FindClose(hFind);
#else
        // Unix: Use opendir/readdir to loop through all entries, stat to check if it's a file and MatchFileNamePattern to match the file against the pattern
        dirp = opendir(szFilePath.c_str());

        if(dirp)
        {
            while( (dp = readdir(dirp)) )
            {
                std::string szFile = szFilePath;
                szFile += dp->d_name;

                if(stat(szFile.c_str(), &stbuf) == 0)
                {
                    if(stbuf.st_mode != S_IFDIR)
                    {
                        if(MatchFileNamePattern(dp->d_name, pszFile.c_str()))
                        {
                            Status = AddFile(szFile, Criteria->TargetFolder);

                            if(Status != CAB_STATUS_SUCCESS)
                            {
                                DPRINT(MIN_TRACE, ("Cannot add file to cabinet (%u).\n", (UINT)Status));
                                goto cleanup;
                            }
                        }
                    }
                }
                else
                {
                    DPRINT(MIN_TRACE, ("stat failed, error code is %i\n", errno));
                    goto cleanup;
                }
            }

            closedir(dirp);
        }
#endif
    }

    Status = WriteDisk(false);
    if (Status == CAB_STATUS_SUCCESS)
        Status = CloseDisk();
    if (Status != CAB_STATUS_SUCCESS)
    {
        DPRINT(MIN_TRACE, ("Cannot write disk (%u).\n", (UINT)Status));
        goto cleanup;
    }

cleanup:
    CloseCabinet();
    bRet = true;

cleanup2:
    DestroySearchCriteria();
    return bRet;
}

void CCabinet::SetMaxDiskSize(ULONG Size)
/*
 * FUNCTION: Sets the maximum size of the current disk
 * ARGUMENTS:
 *     Size = Maximum size of current disk (0 means no maximum size)
 */
{
    MaxDiskSize = Size;
}

#endif /* CAB_READ_ONLY */


/* Default event handlers */

bool CCabinet::OnOverwrite(PCFFILE File,
                           const char* FileName)
/*
 * FUNCTION: Called when extracting a file and it already exists
 * ARGUMENTS:
 *     File     = Pointer to CFFILE for file being extracted
 *     FileName = Pointer to buffer with name of file (full path)
 * RETURNS
 *     true if the file should be overwritten, false if not
 */
{
    return false;
}


void CCabinet::OnExtract(PCFFILE File,
                         const char* FileName)
/*
 * FUNCTION: Called just before extracting a file
 * ARGUMENTS:
 *     File     = Pointer to CFFILE for file being extracted
 *     FileName = Pointer to buffer with name of file (full path)
 */
{
}


void CCabinet::OnDiskChange(const char* CabinetName,
                            const char* DiskLabel)
/*
 * FUNCTION: Called when a new disk is to be processed
 * ARGUMENTS:
 *     CabinetName = Pointer to buffer with name of cabinet
 *     DiskLabel   = Pointer to buffer with label of disk
 */
{
}

void CCabinet::OnVerboseMessage(const char* Message)
{

}

#ifndef CAB_READ_ONLY

void CCabinet::OnAdd(PCFFILE File,
                     const char* FileName)
/*
 * FUNCTION: Called just before adding a file to a cabinet
 * ARGUMENTS:
 *     File     = Pointer to CFFILE for file being added
 *     FileName = Pointer to buffer with name of file (full path)
 */
{
}


bool CCabinet::OnDiskLabel(ULONG Number, char* Label)
/*
 * FUNCTION: Called when a disk needs a label
 * ARGUMENTS:
 *     Number = Cabinet number that needs a label
 *     Label  = Pointer to buffer to place label of disk
 * RETURNS:
 *     true if a disk label was returned, false if not
 */
{
    return false;
}


bool CCabinet::OnCabinetName(ULONG Number, char* Name)
/*
 * FUNCTION: Called when a cabinet needs a name
 * ARGUMENTS:
 *     Number = Disk number that needs a name
 *     Name   = Pointer to buffer to place name of cabinet
 * RETURNS:
 *     true if a cabinet name was returned, false if not
 */
{
    return false;
}

#endif /* CAB_READ_ONLY */

PCFFOLDER_NODE CCabinet::LocateFolderNode(ULONG Index)
/*
 * FUNCTION: Locates a folder node
 * ARGUMENTS:
 *     Index = Folder index
 * RETURNS:
 *     Pointer to folder node or NULL if the folder node was not found
 */
{
    switch (Index)
    {
        case CAB_FILE_SPLIT:
            return FolderList.back();

        case CAB_FILE_CONTINUED:
        case CAB_FILE_PREV_NEXT:
            return FolderList.front();
    }

    for (PCFFOLDER_NODE Node : FolderList)
    {
        if (Node->Index == Index)
            return Node;
    }
    return NULL;
}


ULONG CCabinet::GetAbsoluteOffset(PCFFILE_NODE File)
/*
 * FUNCTION: Returns the absolute offset of a file
 * ARGUMENTS:
 *     File = Pointer to CFFILE_NODE structure for file
 * RETURNS:
 *     Status of operation
 */
{
    DPRINT(MAX_TRACE, ("FileName '%s'  FileOffset (0x%X)  FileSize (%u).\n",
        File->FileName.c_str(),
        (UINT)File->File.FileOffset,
        (UINT)File->File.FileSize));

    for (PCFDATA_NODE Node : CurrentFolderNode->DataList)
    {
        DPRINT(MAX_TRACE, ("GetAbsoluteOffset(): Comparing (0x%X, 0x%X) (%u).\n",
            (UINT)Node->UncompOffset,
            (UINT)(Node->UncompOffset + Node->Data.UncompSize),
            (UINT)Node->Data.UncompSize));

        /* Node->Data.UncompSize will be 0 if the block is split
           (ie. it is the last block in this cabinet) */
        if ((Node->Data.UncompSize == 0) ||
            ((File->File.FileOffset >= Node->UncompOffset) &&
            (File->File.FileOffset < Node->UncompOffset +
            Node->Data.UncompSize)))
        {
                File->DataBlock = Node;
                return CAB_STATUS_SUCCESS;
        }
    }
    return CAB_STATUS_INVALID_CAB;
}


ULONG CCabinet::LocateFile(const char* FileName,
                           PCFFILE_NODE *File)
/*
 * FUNCTION: Locates a file in the cabinet
 * ARGUMENTS:
 *     FileName = Pointer to string with name of file to locate
 *     File     = Address of pointer to CFFILE_NODE structure to fill
 * RETURNS:
 *     Status of operation
 * NOTES:
 *     Current folder is set to the folder of the file
 */
{
    ULONG Status;

    DPRINT(MAX_TRACE, ("FileName '%s'\n", FileName));

    for (PCFFILE_NODE Node : FileList)
    {
        // FIXME: We could handle path\filename here
        if (strcasecmp(FileName, Node->FileName.c_str()) == 0)
        {
            CurrentFolderNode = LocateFolderNode(Node->File.FileControlID);
            if (!CurrentFolderNode)
            {
                DPRINT(MID_TRACE, ("Folder with index number (%u) not found.\n",
                    Node->File.FileControlID));
                return CAB_STATUS_INVALID_CAB;
            }

            if (Node->DataBlock == NULL)
                Status = GetAbsoluteOffset(Node);
            else
                Status = CAB_STATUS_SUCCESS;

            *File = Node;
            return Status;
        }
    }
    return CAB_STATUS_NOFILE;
}


ULONG CCabinet::ReadString(char* String, LONG MaxLength)
/*
 * FUNCTION: Reads a NULL-terminated string from the cabinet
 * ARGUMENTS:
 *     String    = Pointer to buffer to place string
 *     MaxLength = Maximum length of string
 * RETURNS:
 *     Status of operation
 */
{
    ULONG BytesRead;
    ULONG Status;
    LONG Size;
    bool Found;

    Found  = false;

    Status = ReadBlock(String, MaxLength, &BytesRead);
    if (Status != CAB_STATUS_SUCCESS)
    {
        DPRINT(MIN_TRACE, ("Cannot read from file (%u).\n", (UINT)Status));
        return CAB_STATUS_INVALID_CAB;
    }

    // Find the terminating NULL character
    for (Size = 0; Size < MaxLength; Size++)
    {
        if (String[Size] == '\0')
        {
            Found = true;
            break;
        }
    }

    if (!Found)
    {
        DPRINT(MIN_TRACE, ("Filename in the cabinet file is too long.\n"));
        return CAB_STATUS_INVALID_CAB;
    }

    // Compute the offset of the next CFFILE.
    // We have to subtract from the current offset here, because we read MaxLength characters above and most-probably the file name isn't MaxLength characters long.
    // + 1 to skip the terminating NULL character as well.
    Size = -(MaxLength - Size) + 1;

    if (fseek(FileHandle, (off_t)Size, SEEK_CUR) != 0)
    {
        DPRINT(MIN_TRACE, ("fseek() failed.\n"));
        return CAB_STATUS_INVALID_CAB;
    }

    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::ReadFileTable()
/*
 * FUNCTION: Reads the file table from the cabinet file
 * RETURNS:
 *     Status of operation
 */
{
    ULONG i;
    ULONG Status;
    ULONG BytesRead;
    PCFFILE_NODE File;

    DPRINT(MAX_TRACE, ("Reading file table at absolute offset (0x%X).\n",
        (UINT)CABHeader.FileTableOffset));

    /* Seek to file table */
    if (fseek(FileHandle, (off_t)CABHeader.FileTableOffset, SEEK_SET) != 0)
    {
        DPRINT(MIN_TRACE, ("fseek() failed.\n"));
        return CAB_STATUS_INVALID_CAB;
    }

    for (i = 0; i < CABHeader.FileCount; i++)
    {
        File = NewFileNode();
        if (!File)
        {
            DPRINT(MIN_TRACE, ("Insufficient memory.\n"));
            return CAB_STATUS_NOMEMORY;
        }

        if ((Status = ReadBlock(&File->File, sizeof(CFFILE),
            &BytesRead)) != CAB_STATUS_SUCCESS)
        {
            DPRINT(MIN_TRACE, ("Cannot read from file (%u).\n", (UINT)Status));
            return CAB_STATUS_INVALID_CAB;
        }

        /* Read file name */
        char Buf[PATH_MAX];
        Status = ReadString(Buf, PATH_MAX);
        if (Status != CAB_STATUS_SUCCESS)
            return Status;
        // FIXME: We could split up folder\file.txt here
        File->FileName = Buf;

        DPRINT(MAX_TRACE, ("Found file '%s' at uncompressed offset (0x%X).  Size (%u bytes)  ControlId (0x%X).\n",
            File->FileName.c_str(),
            (UINT)File->File.FileOffset,
            (UINT)File->File.FileSize,
            File->File.FileControlID));

    }
    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::ReadDataBlocks(PCFFOLDER_NODE FolderNode)
/*
 * FUNCTION: Reads all CFDATA blocks for a folder from the cabinet file
 * ARGUMENTS:
 *     FolderNode = Pointer to CFFOLDER_NODE structure for folder
 * RETURNS:
 *     Status of operation
 */
{
    ULONG AbsoluteOffset;
    ULONG UncompOffset;
    PCFDATA_NODE Node;
    ULONG BytesRead;
    ULONG Status;
    ULONG i;

    DPRINT(MAX_TRACE, ("Reading data blocks for folder (%u)  at absolute offset (0x%X).\n",
        (UINT)FolderNode->Index, (UINT)FolderNode->Folder.DataOffset));

    AbsoluteOffset = FolderNode->Folder.DataOffset;
    UncompOffset   = FolderNode->UncompOffset;

    for (i = 0; i < FolderNode->Folder.DataBlockCount; i++)
    {
        Node = NewDataNode(FolderNode);
        if (!Node)
        {
            DPRINT(MIN_TRACE, ("Insufficient memory.\n"));
            return CAB_STATUS_NOMEMORY;
        }

        /* Seek to data block */
        if (fseek(FileHandle, (off_t)AbsoluteOffset, SEEK_SET) != 0)
        {
            DPRINT(MIN_TRACE, ("fseek() failed.\n"));
            return CAB_STATUS_INVALID_CAB;
        }

        if ((Status = ReadBlock(&Node->Data, sizeof(CFDATA),
            &BytesRead)) != CAB_STATUS_SUCCESS)
        {
            DPRINT(MIN_TRACE, ("Cannot read from file (%u).\n", (UINT)Status));
            return CAB_STATUS_INVALID_CAB;
        }

        DPRINT(MAX_TRACE, ("AbsOffset (0x%X)  UncompOffset (0x%X)  Checksum (0x%X)  CompSize (%u)  UncompSize (%u).\n",
            (UINT)AbsoluteOffset,
            (UINT)UncompOffset,
            (UINT)Node->Data.Checksum,
            Node->Data.CompSize,
            Node->Data.UncompSize));

        Node->AbsoluteOffset = AbsoluteOffset;
        Node->UncompOffset   = UncompOffset;

        AbsoluteOffset += sizeof(CFDATA) + Node->Data.CompSize;
        UncompOffset   += Node->Data.UncompSize;
    }

    FolderUncompSize = UncompOffset;

    return CAB_STATUS_SUCCESS;
}


PCFFOLDER_NODE CCabinet::NewFolderNode()
/*
 * FUNCTION: Creates a new folder node
 * RETURNS:
 *     Pointer to node if there was enough free memory available, otherwise NULL
 */
{
    PCFFOLDER_NODE Node;

    Node = new CFFOLDER_NODE;
    if (!Node)
        return NULL;

    Node->Folder.CompressionType = CAB_COMP_NONE;

    FolderList.push_back(Node);

    return Node;
}


PCFFILE_NODE CCabinet::NewFileNode()
/*
 * FUNCTION: Creates a new file node
 * ARGUMENTS:
 *     FolderNode = Pointer to folder node to bind file to
 * RETURNS:
 *     Pointer to node if there was enough free memory available, otherwise NULL
 */
{
    PCFFILE_NODE Node;

    Node = new CFFILE_NODE;
    if (!Node)
        return NULL;

    FileList.push_back(Node);

    return Node;
}


PCFDATA_NODE CCabinet::NewDataNode(PCFFOLDER_NODE FolderNode)
/*
 * FUNCTION: Creates a new data block node
 * ARGUMENTS:
 *     FolderNode = Pointer to folder node to bind data block to
 * RETURNS:
 *     Pointer to node if there was enough free memory available, otherwise NULL
 */
{
    PCFDATA_NODE Node;

    Node = new CFDATA_NODE;
    if (!Node)
        return NULL;

    FolderNode->DataList.push_back(Node);

    return Node;
}


void CCabinet::DestroyDataNodes(PCFFOLDER_NODE FolderNode)
/*
 * FUNCTION: Destroys data block nodes bound to a folder node
 * ARGUMENTS:
 *     FolderNode = Pointer to folder node
 */
{
    for (PCFDATA_NODE Node : FolderNode->DataList)
    {
        delete Node;
    }
    FolderNode->DataList.clear();
}


void CCabinet::DestroyFileNodes()
/*
 * FUNCTION: Destroys file nodes
 */
{
    for (PCFFILE_NODE Node : FileList)
    {
        delete Node;
    }
    FileList.clear();
}


void CCabinet::DestroyDeletedFileNodes()
/*
 * FUNCTION: Destroys file nodes that are marked for deletion
 */
{
    for (auto it = FileList.begin(); it != FileList.end(); )
    {
        PCFFILE_NODE CurNode = *it;

        if (CurNode->Delete)
        {
            it = FileList.erase(it);

            DPRINT(MAX_TRACE, ("Deleting file node: '%s'\n", CurNode->FileName.c_str()));

            TotalFileSize -= (sizeof(CFFILE) + (ULONG)CreateCabFilename(CurNode).length() + 1);

            delete CurNode;
        }
        else
        {
            it++;
        }
    }
}


void CCabinet::DestroyFolderNodes()
/*
 * FUNCTION: Destroys folder nodes
 */
{
    for (PCFFOLDER_NODE Node : FolderList)
    {
        DestroyDataNodes(Node);
        delete Node;
    }
    FolderList.clear();
}


void CCabinet::DestroyDeletedFolderNodes()
/*
 * FUNCTION: Destroys folder nodes that are marked for deletion
 */
{
    for (auto it = FolderList.begin(); it != FolderList.end();)
    {
        PCFFOLDER_NODE CurNode = *it;

        if (CurNode->Delete)
        {
            it = FolderList.erase(it);

            DestroyDataNodes(CurNode);
            delete CurNode;

            TotalFolderSize -= sizeof(CFFOLDER);
        }
        else
        {
            it++;
        }
    }
}


ULONG CCabinet::ComputeChecksum(void* Buffer,
                                ULONG Size,
                                ULONG Seed)
/*
 * FUNCTION: Computes checksum for data block
 * ARGUMENTS:
 *     Buffer = Pointer to data buffer
 *     Size   = Length of data buffer
 *     Seed   = Previously computed checksum
 * RETURNS:
 *     Checksum of buffer
 */
{
    int UlongCount; // Number of ULONGs in block
    ULONG Checksum; // Checksum accumulator
    unsigned char* pb;
    ULONG ul;

    /* FIXME: Doesn't seem to be correct. EXTRACT.EXE
       won't accept checksums computed by this routine */

    DPRINT(MIN_TRACE, ("Checksumming buffer (0x%p)  Size (%u)\n", Buffer, (UINT)Size));

    UlongCount = Size / 4;              // Number of ULONGs
    Checksum   = Seed;                  // Init checksum
    pb         = (unsigned char*)Buffer;         // Start at front of data block

    /* Checksum integral multiple of ULONGs */
    while (UlongCount-- > 0)
    {
        /* NOTE: Build ULONG in big/little-endian independent manner */
        ul = *pb++;                     // Get low-order byte
        ul |= (((ULONG)(*pb++)) <<  8); // Add 2nd byte
        ul |= (((ULONG)(*pb++)) << 16); // Add 3nd byte
        ul |= (((ULONG)(*pb++)) << 24); // Add 4th byte

        Checksum ^= ul;                 // Update checksum
    }

    /* Checksum remainder bytes */
    ul = 0;
    switch (Size % 4)
    {
        case 3:
            ul |= (((ULONG)(*pb++)) << 16); // Add 3rd byte
        case 2:
            ul |= (((ULONG)(*pb++)) <<  8); // Add 2nd byte
        case 1:
            ul |= *pb++;                    // Get low-order byte
        default:
            break;
    }
    Checksum ^= ul;                         // Update checksum

    /* Return computed checksum */
    return Checksum;
}


ULONG CCabinet::ReadBlock(void* Buffer,
                          ULONG Size,
                          PULONG BytesRead)
/*
 * FUNCTION: Read a block of data from file
 * ARGUMENTS:
 *     Buffer    = Pointer to data buffer
 *     Size      = Length of data buffer
 *     BytesRead = Pointer to ULONG that on return will contain
 *                 number of bytes read
 * RETURNS:
 *     Status of operation
 */
{
    *BytesRead = fread(Buffer, 1, Size, FileHandle);
    if ( *BytesRead != Size )
        return CAB_STATUS_INVALID_CAB;
    return CAB_STATUS_SUCCESS;
}

bool CCabinet::MatchFileNamePattern(const char* FileName, const char* Pattern)
/*
 * FUNCTION: Matches a wildcard character pattern against a file
 * ARGUMENTS:
 *     FileName = The file name to check
 *     Pattern  = The pattern
 * RETURNS:
 *     Whether the pattern matches the file
 *
 * COPYRIGHT:
 *     This function is based on Busybox code, Copyright (C) 1998 by Erik Andersen, released under GPL2 or any later version.
 *     Adapted from code written by Ingo Wilken.
 *     Original location: http://www.busybox.net/cgi-bin/viewcvs.cgi/trunk/busybox/utility.c?rev=5&view=markup
 */
{
    const char* retryPattern = NULL;
    const char* retryFileName = NULL;
    char  ch;

    while (*FileName || *Pattern)
    {
        ch = *Pattern++;

        switch (ch)
        {
            case '*':
                retryPattern = Pattern;
                retryFileName = FileName;
                break;

            case '?':
                if (*FileName++ == '\0')
                    return false;

                break;

            default:
                if (*FileName == ch)
                {
                    if (*FileName)
                        FileName++;
                    break;
                }

                if (*FileName)
                {
                    Pattern = retryPattern;
                    FileName = ++retryFileName;
                    break;
                }

                return false;
        }

        if (!Pattern)
            return false;
    }

    return true;
}

#ifndef CAB_READ_ONLY

ULONG CCabinet::InitCabinetHeader()
/*
 * FUNCTION: Initializes cabinet header and optional fields
 * RETURNS:
 *     Status of operation
 */
{
    ULONG TotalSize;
    ULONG Size;

    CABHeader.FileTableOffset = 0;    // Not known yet
    CABHeader.FolderCount     = 0;    // Not known yet
    CABHeader.FileCount       = 0;    // Not known yet
    CABHeader.Flags           = 0;    // Not known yet

    CABHeader.CabinetNumber = (USHORT)CurrentDiskNumber;

    if ((CurrentDiskNumber > 0) && (OnCabinetName(PrevCabinetNumber, CabinetPrev)))
    {
        CABHeader.Flags |= CAB_FLAG_HASPREV;
        if (!OnDiskLabel(PrevCabinetNumber, DiskPrev))
            strcpy(CabinetPrev, "");
    }

    if (OnCabinetName(CurrentDiskNumber + 1, CabinetNext))
    {
        CABHeader.Flags |= CAB_FLAG_HASNEXT;
        if (!OnDiskLabel(CurrentDiskNumber + 1, DiskNext))
            strcpy(DiskNext, "");
    }

    TotalSize = 0;

    if ((CABHeader.Flags & CAB_FLAG_HASPREV) > 0)
    {

        DPRINT(MAX_TRACE, ("CabinetPrev '%s'.\n", CabinetPrev));

        /* Calculate size of name of previous cabinet */
        TotalSize += (ULONG)strlen(CabinetPrev) + 1;

        /* Calculate size of label of previous disk */
        TotalSize += (ULONG)strlen(DiskPrev) + 1;
    }

    if ((CABHeader.Flags & CAB_FLAG_HASNEXT) > 0)
    {

        DPRINT(MAX_TRACE, ("CabinetNext '%s'.\n", CabinetNext));

        /* Calculate size of name of next cabinet */
        Size = (ULONG)strlen(CabinetNext) + 1;
        TotalSize += Size;
        NextFieldsSize = Size;

        /* Calculate size of label of next disk */
        Size = (ULONG)strlen(DiskNext) + 1;
        TotalSize += Size;
        NextFieldsSize += Size;
    }
    else
        NextFieldsSize = 0;

    /* Add cabinet reserved area size if present */
    if (CabinetReservedFileSize > 0)
    {
        CABHeader.Flags |= CAB_FLAG_RESERVE;
        TotalSize += CabinetReservedFileSize;
        TotalSize += sizeof(ULONG); /* For CabinetResSize, FolderResSize, and FileResSize fields */
    }

    DiskSize += TotalSize;

    TotalHeaderSize = sizeof(CFHEADER) + TotalSize;

    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::WriteCabinetHeader(bool MoreDisks)
/*
 * FUNCTION: Writes the cabinet header and optional fields
 * ARGUMENTS:
 *     MoreDisks = true if next cabinet name should be included
 * RETURNS:
 *     Status of operation
 */
{
    ULONG BytesWritten;
    ULONG Size;

    if (MoreDisks)
    {
        CABHeader.Flags |= CAB_FLAG_HASNEXT;
        Size = TotalHeaderSize;
    }
    else
    {
        CABHeader.Flags &= ~CAB_FLAG_HASNEXT;
        DiskSize -= NextFieldsSize;
        Size = TotalHeaderSize - NextFieldsSize;
    }

    /* Set absolute folder offsets */
    BytesWritten = Size + TotalFolderSize + TotalFileSize;
    CABHeader.FolderCount = 0;
    for (PCFFOLDER_NODE FolderNode : FolderList)
    {
        FolderNode->Folder.DataOffset = BytesWritten;

        BytesWritten += FolderNode->TotalFolderSize;

        CABHeader.FolderCount++;
    }

    /* Set absolute offset of file table */
    CABHeader.FileTableOffset = Size + TotalFolderSize;

    /* Count number of files to be committed */
    CABHeader.FileCount = 0;
    for (PCFFILE_NODE FileNode : FileList)
    {
        if (FileNode->Commit)
            CABHeader.FileCount++;
    }

    CABHeader.CabinetSize = DiskSize;

    /* Write header */
    if (fwrite(&CABHeader, sizeof(CFHEADER), 1, FileHandle) < 1)
    {
        DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
        return CAB_STATUS_CANNOT_WRITE;
    }

    /* Write per-cabinet reserved area if present */
    if (CABHeader.Flags & CAB_FLAG_RESERVE)
    {
        ULONG ReservedSize;

        ReservedSize = CabinetReservedFileSize & 0xffff;
        ReservedSize |= (0 << 16); /* Folder reserved area size */
        ReservedSize |= (0 << 24); /* Folder reserved area size */

        BytesWritten = sizeof(ULONG);
        if (fwrite(&ReservedSize, sizeof(ULONG), 1, FileHandle) < 1)
        {
            DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
            return CAB_STATUS_CANNOT_WRITE;
        }

        BytesWritten = CabinetReservedFileSize;
        if (fwrite(CabinetReservedFileBuffer, CabinetReservedFileSize, 1, FileHandle) < 1)
        {
            DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
            return CAB_STATUS_CANNOT_WRITE;
        }
    }

    if ((CABHeader.Flags & CAB_FLAG_HASPREV) > 0)
    {
        DPRINT(MAX_TRACE, ("CabinetPrev '%s'.\n", CabinetPrev));

        /* Write name of previous cabinet */
        Size = (ULONG)strlen(CabinetPrev) + 1;
        BytesWritten = Size;
        if (fwrite(CabinetPrev, Size, 1, FileHandle) < 1)
        {
            DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
            return CAB_STATUS_CANNOT_WRITE;
        }

        DPRINT(MAX_TRACE, ("DiskPrev '%s'.\n", DiskPrev));

        /* Write label of previous disk */
        Size = (ULONG)strlen(DiskPrev) + 1;
        BytesWritten = Size;
        if (fwrite(DiskPrev, Size, 1, FileHandle) < 1)
        {
            DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
            return CAB_STATUS_CANNOT_WRITE;
        }
    }

    if ((CABHeader.Flags & CAB_FLAG_HASNEXT) > 0)
    {
        DPRINT(MAX_TRACE, ("CabinetNext '%s'.\n", CabinetNext));

        /* Write name of next cabinet */
        Size = (ULONG)strlen(CabinetNext) + 1;
        BytesWritten = Size;
        if (fwrite(CabinetNext, Size, 1, FileHandle) < 1)
        {
            DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
            return CAB_STATUS_CANNOT_WRITE;
        }

        DPRINT(MAX_TRACE, ("DiskNext '%s'.\n", DiskNext));

        /* Write label of next disk */
        Size = (ULONG)strlen(DiskNext) + 1;
        BytesWritten = Size;
        if (fwrite(DiskNext, Size, 1, FileHandle) < 1)
        {
            DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
            return CAB_STATUS_CANNOT_WRITE;
        }
    }

    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::WriteFolderEntries()
/*
 * FUNCTION: Writes folder entries
 * RETURNS:
 *     Status of operation
 */
{
    DPRINT(MAX_TRACE, ("Writing folder table.\n"));

    for (PCFFOLDER_NODE FolderNode : FolderList)
    {
        if (FolderNode->Commit)
        {
            DPRINT(MAX_TRACE, ("Writing folder entry. CompressionType (0x%X)  DataBlockCount (%d)  DataOffset (0x%X).\n",
                FolderNode->Folder.CompressionType, FolderNode->Folder.DataBlockCount, (UINT)FolderNode->Folder.DataOffset));

            if (fwrite(&FolderNode->Folder, sizeof(CFFOLDER), 1, FileHandle) < 1)
            {
                DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
                return CAB_STATUS_CANNOT_WRITE;
            }
        }
    }

    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::WriteFileEntries()
/*
 * FUNCTION: Writes file entries for all files
 * RETURNS:
 *     Status of operation
 */
{
    bool SetCont = false;

    DPRINT(MAX_TRACE, ("Writing file table.\n"));

    for (PCFFILE_NODE File : FileList)
    {
        if (File->Commit)
        {
            /* Remove any continued files that ends in this disk */
            if (File->File.FileControlID == CAB_FILE_CONTINUED)
                File->Delete = true;

            /* The file could end in the last (split) block and should therefore
               appear in the next disk too */

            if ((File->File.FileOffset + File->File.FileSize >= LastBlockStart) &&
                (File->File.FileControlID <= CAB_FILE_MAX_FOLDER) && (BlockIsSplit))
            {
                File->File.FileControlID = CAB_FILE_SPLIT;
                File->Delete = false;
                SetCont = true;
            }

            DPRINT(MAX_TRACE, ("Writing file entry. FileControlID (0x%X)  FileOffset (0x%X)  FileSize (%u)  FileName (%s).\n",
                File->File.FileControlID, (UINT)File->File.FileOffset, (UINT)File->File.FileSize, File->FileName.c_str()));

            if (fwrite(&File->File, sizeof(CFFILE), 1, FileHandle) < 1)
            {
                DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
                return CAB_STATUS_CANNOT_WRITE;
            }

            std::string fname = CreateCabFilename(File);
            if (fwrite(fname.c_str(), fname.length() + 1, 1, FileHandle) < 1)
            {
                DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
                return CAB_STATUS_CANNOT_WRITE;
            }

            if (SetCont)
            {
                File->File.FileControlID = CAB_FILE_CONTINUED;
                SetCont = false;
            }
        }
    }
    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::CommitDataBlocks(PCFFOLDER_NODE FolderNode)
/*
 * FUNCTION: Writes data blocks to the cabinet
 * ARGUMENTS:
 *     FolderNode = Pointer to folder node containing the data blocks
 * RETURNS:
 *     Status of operation
 */
{
    ULONG BytesRead;
    ULONG Status;

    if (!FolderNode->DataList.empty())
        Status = ScratchFile->Seek(FolderNode->DataList.front()->ScratchFilePosition);

    for (PCFDATA_NODE DataNode : FolderNode->DataList)
    {
        DPRINT(MAX_TRACE, ("Reading block at (0x%X)  CompSize (%u)  UncompSize (%u).\n",
            (UINT)DataNode->ScratchFilePosition,
            DataNode->Data.CompSize,
            DataNode->Data.UncompSize));

        /* InputBuffer is free for us to use here, so we use it and avoid a
           memory allocation. OutputBuffer can't be used here because it may
           still contain valid data (if a data block spans two or more disks) */
        Status = ScratchFile->ReadBlock(&DataNode->Data, InputBuffer, &BytesRead);
        if (Status != CAB_STATUS_SUCCESS)
        {
            DPRINT(MIN_TRACE, ("Cannot read from scratch file (%u).\n", (UINT)Status));
            return Status;
        }

        if (fwrite(&DataNode->Data, sizeof(CFDATA), 1, FileHandle) < 1)
        {
            DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
            return CAB_STATUS_CANNOT_WRITE;
        }

        if (fwrite(InputBuffer, DataNode->Data.CompSize, 1, FileHandle) < 1)
        {
            DPRINT(MIN_TRACE, ("Cannot write to file.\n"));
            return CAB_STATUS_CANNOT_WRITE;
        }
    }
    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::WriteDataBlock()
/*
 * FUNCTION: Writes the current data block to the scratch file
 * RETURNS:
 *     Status of operation
 */
{
    ULONG Status;
    ULONG BytesWritten;
    PCFDATA_NODE DataNode;

    if (!BlockIsSplit)
    {
        Status = Codec->Compress(OutputBuffer,
            InputBuffer,
            CurrentIBufferSize,
            &TotalCompSize);

        DPRINT(MAX_TRACE, ("Block compressed. CurrentIBufferSize (%u)  TotalCompSize(%u).\n",
            (UINT)CurrentIBufferSize, (UINT)TotalCompSize));

        CurrentOBuffer     = OutputBuffer;
        CurrentOBufferSize = TotalCompSize;
    }

    DataNode = NewDataNode(CurrentFolderNode);
    if (!DataNode)
    {
        DPRINT(MIN_TRACE, ("Insufficient memory.\n"));
        return CAB_STATUS_NOMEMORY;
    }

    DiskSize += sizeof(CFDATA);

    if (MaxDiskSize > 0)
        /* Disk size is limited */
        BlockIsSplit = (DiskSize + CurrentOBufferSize > MaxDiskSize);
    else
        BlockIsSplit = false;

    if (BlockIsSplit)
    {
        DataNode->Data.CompSize   = (USHORT)(MaxDiskSize - DiskSize);
        DataNode->Data.UncompSize = 0;
        CreateNewDisk = true;
    }
    else
    {
        DataNode->Data.CompSize   = (USHORT)CurrentOBufferSize;
        DataNode->Data.UncompSize = (USHORT)CurrentIBufferSize;
    }

    DataNode->Data.Checksum = 0;
    DataNode->ScratchFilePosition = ScratchFile->Position();

    // FIXME: MAKECAB.EXE does not like this checksum algorithm
    //DataNode->Data.Checksum = ComputeChecksum(CurrentOBuffer, DataNode->Data.CompSize, 0);

    DPRINT(MAX_TRACE, ("Writing block. Checksum (0x%X)  CompSize (%u)  UncompSize (%u).\n",
        (UINT)DataNode->Data.Checksum,
        DataNode->Data.CompSize,
        DataNode->Data.UncompSize));

    Status = ScratchFile->WriteBlock(&DataNode->Data,
        CurrentOBuffer, &BytesWritten);
    if (Status != CAB_STATUS_SUCCESS)
        return Status;

    DiskSize += BytesWritten;

    CurrentFolderNode->TotalFolderSize += (BytesWritten + sizeof(CFDATA));
    CurrentFolderNode->Folder.DataBlockCount++;

    CurrentOBuffer = (unsigned char*)CurrentOBuffer + DataNode->Data.CompSize;
    CurrentOBufferSize -= DataNode->Data.CompSize;

    LastBlockStart += DataNode->Data.UncompSize;

    if (!BlockIsSplit)
    {
        CurrentIBufferSize = 0;
        CurrentIBuffer     = InputBuffer;
    }

    return CAB_STATUS_SUCCESS;
}

#if !defined(_WIN32)

void CCabinet::ConvertDateAndTime(time_t* Time,
                                  PUSHORT DosDate,
                                  PUSHORT DosTime)
/*
 * FUNCTION: Returns file times of a file
 * ARGUMENTS:
 *      FileHandle = File handle of file to get file times from
 *      File       = Pointer to CFFILE node for file
 * RETURNS:
 *     Status of operation
 */
{
    struct tm *timedef;

    timedef = localtime(Time);

    DPRINT(MAX_TRACE, ("day: %d, mon: %d, year:%d, hour: %d, min: %d, sec: %d\n",
        timedef->tm_mday, timedef->tm_mon, timedef->tm_year,
        timedef->tm_sec, timedef->tm_min, timedef->tm_hour));

    *DosDate = ((timedef->tm_mday + 1) << 0)
        | ((timedef->tm_mon + 1) << 5)
        | (((timedef->tm_year + 1900) - 1980) << 9);

    *DosTime = (timedef->tm_sec << 0)
        | (timedef->tm_min << 5)
        | (timedef->tm_hour << 11);
}

#endif // !_WIN32


ULONG CCabinet::GetFileTimes(FILE* FileHandle, PCFFILE_NODE File)
/*
 * FUNCTION: Returns file times of a file
 * ARGUMENTS:
 *      FileHandle = File handle of file to get file times from
 *      File       = Pointer to CFFILE node for file
 * RETURNS:
 *     Status of operation
 */
{
#if defined(_WIN32)
    FILETIME FileTime;
    HANDLE FileNo = UlongToHandle(_fileno(FileHandle));

    if (GetFileTime(FileNo, NULL, NULL, &FileTime))
        FileTimeToDosDateTime(&FileTime,
            &File->File.FileDate,
            &File->File.FileTime);
#else
    struct stat stbuf;
    char buf[PATH_MAX];

    // Check for an absolute path
    if (File->FileName.length() > 0 && IsSeparator(File->FileName[0]))
        strcpy(buf, File->FileName.c_str());
    else
    {
        if (!getcwd(buf, sizeof(buf)))
            return CAB_STATUS_CANNOT_READ;
        strcat(buf, DIR_SEPARATOR_STRING);
        strcat(buf, File->FileName.c_str());
    }

    if (stat(buf, &stbuf) == -1)
        return CAB_STATUS_CANNOT_READ;

    ConvertDateAndTime(&stbuf.st_mtime, &File->File.FileDate, &File->File.FileTime);
#endif
    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::GetAttributesOnFile(PCFFILE_NODE File)
/*
 * FUNCTION: Returns attributes on a file
 * ARGUMENTS:
 *      File = Pointer to CFFILE node for file
 * RETURNS:
 *     Status of operation
 */
{
#if defined(_WIN32)
    LONG Attributes;

    Attributes = GetFileAttributes(File->FileName.c_str());
    if (Attributes == -1)
        return CAB_STATUS_CANNOT_READ;

    // 0x37 = READONLY | HIDDEN | SYSTEM | DIRECTORY | ARCHIVE
    // The IDs for these attributes are the same in the CAB file and under Windows
    // If the file has any other attributes, strip them off by the logical AND.
    File->File.Attributes = (USHORT)(Attributes & 0x37);
#else
    struct stat stbuf;
    char buf[PATH_MAX];

    // Check for an absolute path
    if (File->FileName.length() > 0 && IsSeparator(File->FileName[0]))
        strcpy(buf, File->FileName.c_str());
    else
    {
        if (!getcwd(buf, sizeof(buf)))
            return CAB_STATUS_CANNOT_READ;
        strcat(buf, DIR_SEPARATOR_STRING);
        strcat(buf, File->FileName.c_str());
    }

    if (stat(buf, &stbuf) == -1)
        return CAB_STATUS_CANNOT_READ;

#if 0
    File->File.Attributes |= CAB_ATTRIB_READONLY;
    File->File.Attributes |= CAB_ATTRIB_HIDDEN;
    File->File.Attributes |= CAB_ATTRIB_SYSTEM;
#endif

    if (stbuf.st_mode & S_IFDIR)
        File->File.Attributes |= CAB_ATTRIB_DIRECTORY;

    File->File.Attributes |= CAB_ATTRIB_ARCHIVE;

#endif
    return CAB_STATUS_SUCCESS;
}


ULONG CCabinet::SetAttributesOnFile(char* FileName, USHORT FileAttributes)
/*
 * FUNCTION: Sets attributes on a file
 * ARGUMENTS:
 *      FileName       = File name with path
 *      FileAttributes = Attributes of that file
 * RETURNS:
 *     Status of operation
 */
{
#if defined(_WIN32)
    // 0x37 = READONLY | HIDDEN | SYSTEM | DIRECTORY | ARCHIVE
    // The IDs for these attributes are the same in the CAB file and under Windows
    // If the file has any other attributes, strip them off by the logical AND.
    SetFileAttributes(FileName, (DWORD)(FileAttributes & 0x37));

    return CAB_STATUS_SUCCESS;
#else
    //DPRINT(MIN_TRACE, ("FIXME: SetAttributesOnFile() is unimplemented\n"));
    return CAB_STATUS_SUCCESS;
#endif
}

#endif /* CAB_READ_ONLY */

/* EOF */
