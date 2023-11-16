"""
Contains a snapshot of the state of the Download at a specific point in time.

Author(s): Arno Bakker
"""
import logging
from tribler.core.utilities.simpledefs import DownloadStatus, UPLOAD
DOWNLOAD_STATUS_MAP = [DownloadStatus.WAITING_FOR_HASHCHECK, DownloadStatus.HASHCHECKING, DownloadStatus.METADATA, DownloadStatus.DOWNLOADING, DownloadStatus.SEEDING, DownloadStatus.SEEDING, DownloadStatus.ALLOCATING_DISKSPACE, DownloadStatus.HASHCHECKING]

class DownloadState:
    """
    Contains a snapshot of the state of the Download at a specific
    point in time. Using a snapshot instead of providing live data and
    protecting access via locking should be faster.

    cf. libtorrent torrent_status
    """

    def __init__(self, download, lt_status, error):
        if False:
            while True:
                i = 10
        '\n        Internal constructor.\n        @param download The download this state belongs too.\n        @param lt_status The libtorrent status object\n        @param tr_status Any Tribler specific information regarding the download\n        '
        self._logger = logging.getLogger(self.__class__.__name__)
        self.download = download
        self.lt_status = lt_status
        self.error = error

    def get_download(self):
        if False:
            i = 10
            return i + 15
        ' Returns the Download object of which this is the state '
        return self.download

    def get_progress(self):
        if False:
            while True:
                i = 10
        ' The general progress of the Download as a percentage. When status is\n         * HASHCHECKING it is the percentage of already downloaded\n           content checked for integrity.\n         * DOWNLOADING/SEEDING it is the percentage downloaded.\n        @return Progress as a float (0..1).\n        '
        return self.lt_status.progress if self.lt_status else 0

    def get_status(self) -> DownloadStatus:
        if False:
            for i in range(10):
                print('nop')
        ' Returns the status of the torrent.\n        @return DownloadStatus* '
        if self.lt_status:
            if self.lt_status.paused:
                return DownloadStatus.STOPPED
            return DOWNLOAD_STATUS_MAP[self.lt_status.state]
        if self.get_error():
            return DownloadStatus.STOPPED_ON_ERROR
        return DownloadStatus.STOPPED

    def get_error(self):
        if False:
            i = 10
            return i + 15
        ' Returns the Exception that caused the download to be moved to STOPPED_ON_ERROR status.\n        @return An error message\n        '
        return self.error or (self.lt_status.error if self.lt_status and self.lt_status.error else None)

    def get_current_speed(self, direct):
        if False:
            while True:
                i = 10
        '\n        Returns the current up or download speed.\n        @return The speed in bytes/s.\n        '
        if not self.lt_status or self.get_status() not in [DownloadStatus.DOWNLOADING, DownloadStatus.SEEDING]:
            return 0
        if direct == UPLOAD:
            return self.lt_status.upload_rate
        return self.lt_status.download_rate

    def get_current_payload_speed(self, direct):
        if False:
            return 10
        '\n        Returns the current up or download payload speed.\n        @return The speed in bytes/s.\n        '
        if not self.lt_status or self.get_status() not in [DownloadStatus.DOWNLOADING, DownloadStatus.SEEDING]:
            return 0
        if direct == UPLOAD:
            return self.lt_status.upload_payload_rate
        return self.lt_status.download_payload_rate

    def get_total_transferred(self, direct):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the total amount of up or downloaded bytes.\n        @return The amount in bytes.\n        '
        if not self.lt_status:
            return 0
        elif direct == UPLOAD:
            return self.lt_status.total_upload
        return self.lt_status.total_download

    def get_seeding_ratio(self):
        if False:
            i = 10
            return i + 15
        if self.lt_status and self.lt_status.total_done > 0:
            return self.lt_status.all_time_upload / float(self.lt_status.total_done)
        return 0

    def get_seeding_time(self):
        if False:
            for i in range(10):
                print('nop')
        return self.lt_status.finished_time if self.lt_status else 0

    def get_eta(self):
        if False:
            while True:
                i = 10
        '\n        Returns the estimated time to finish of download.\n        @return The time in ?, as ?.\n        '
        return (1.0 - self.get_progress()) * (float(self.download.get_def().get_length()) / max(1e-06, self.lt_status.download_rate)) if self.lt_status else 0.0

    def get_num_seeds_peers(self):
        if False:
            return 10
        '\n        Returns the sum of the number of seeds and peers.\n        @return A tuple (num seeds, num peers)\n        '
        if not self.lt_status or self.get_status() not in [DownloadStatus.DOWNLOADING, DownloadStatus.SEEDING]:
            return (0, 0)
        total = self.lt_status.list_peers
        seeds = self.lt_status.list_seeds
        return (seeds, total - seeds)

    def get_pieces_complete(self):
        if False:
            while True:
                i = 10
        ' Returns a list of booleans indicating whether we have completely\n        received that piece of the content. The list of pieces for which\n        we provide this info depends on which files were selected for download\n        using DownloadConfig.set_selected_files().\n        @return A list of booleans\n        '
        return self.lt_status.pieces if self.lt_status else []

    def get_pieces_total_complete(self):
        if False:
            i = 10
            return i + 15
        ' Returns the number of total and completed pieces\n        @return A tuple containing two integers, total and completed nr of pieces\n        '
        return (len(self.lt_status.pieces), sum(self.lt_status.pieces)) if self.lt_status else (0, 0)

    def get_files_completion(self):
        if False:
            return 10
        ' Returns a list of filename, progress tuples indicating the progress\n        for every file selected using set_selected_files. Progress is a float\n        between 0 and 1\n        '
        completion = []
        if self.lt_status and self.download.handle and self.download.handle.is_valid():
            files = self.download.get_def().get_files_with_length()
            try:
                progress = self.download.handle.file_progress(flags=1)
            except RuntimeError:
                progress = None
            if progress and len(progress) == len(files):
                for (index, (path, size)) in enumerate(files):
                    completion_frac = float(progress[index]) / size if size > 0 else 1
                    completion.append((path, completion_frac))
        return completion

    def get_selected_files(self):
        if False:
            for i in range(10):
                print('nop')
        selected_files = self.download.config.get_selected_files()
        if len(selected_files) > 0:
            return selected_files

    def get_availability(self):
        if False:
            return 10
        ' Return overall the availability of all pieces, using connected peers\n        Availability is defined as the number of complete copies of a piece, thus seeders\n        increment the availability by 1. Leechers provide a subset of piece thus we count the\n        overall availability of all pieces provided by the connected peers and use the minimum\n        of this + the average of all additional pieces.\n        '
        if not self.lt_status:
            return 0
        nr_seeders_complete = 0
        merged_bitfields = [0] * len(self.lt_status.pieces)
        peers = self.get_peerlist()
        for peer in peers:
            completed = peer.get('completed', 0)
            have = peer.get('have', [])
            if completed == 1 or (have and all(have)):
                nr_seeders_complete += 1
            elif have and len(have) == len(merged_bitfields):
                for i in range(len(have)):
                    if have[i]:
                        merged_bitfields[i] += 1
        if merged_bitfields:
            nr_leechers_complete = min(merged_bitfields)
            nr_more_than_min = len([x for x in merged_bitfields if x > nr_leechers_complete])
            fraction_additonal = float(nr_more_than_min) / len(merged_bitfields)
            return nr_seeders_complete + nr_leechers_complete + fraction_additonal
        return nr_seeders_complete

    def get_peerlist(self):
        if False:
            return 10
        ' Returns a list of dictionaries, one for each connected peer\n        containing the statistics for that peer.\n        '
        return self.download.get_peerlist()