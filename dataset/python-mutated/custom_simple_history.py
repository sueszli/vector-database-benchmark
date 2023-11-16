from django.db import models
from simple_history.models import HistoricalRecords

class NonWritingHistoricalRecords(HistoricalRecords):
    """
    Custom implementation of the HistoricalRecords class to prevent the signals
    being connected to the model. This allows us to stop the writing of records
    and then in a subsequent release, remove the database table.
    """

    def finalize(self, sender, **kwargs):
        if False:
            while True:
                i = 10
        super(NonWritingHistoricalRecords, self).finalize(sender, **kwargs)
        models.signals.post_save.disconnect(self.post_save, sender=sender)
        models.signals.post_delete.disconnect(self.post_delete, sender=sender)