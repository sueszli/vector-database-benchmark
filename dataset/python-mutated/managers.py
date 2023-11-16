from django.db import models

class DebugQueryManager(models.Manager):

    def get_queryset(self):
        if False:
            for i in range(10):
                print('nop')
        import traceback
        lines = traceback.format_stack()
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        for line in lines[-10:-1]:
            print(line)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        queryset = super().get_queryset()
        return queryset