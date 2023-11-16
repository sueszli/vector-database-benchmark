import pyrogram

class StopTransmission:

    def stop_transmission(self):
        if False:
            return 10
        'Stop downloading or uploading a file.\n\n        This method must be called inside a progress callback function in order to stop the transmission at the\n        desired time. The progress callback is called every time a file chunk is uploaded/downloaded.\n\n        Example:\n            .. code-block:: python\n\n                # Stop transmission once the upload progress reaches 50%\n                async def progress(current, total, client):\n                    if (current * 100 / total) > 50:\n                        client.stop_transmission()\n\n                async with app:\n                    await app.send_document(\n                        "me", "file.zip",\n                        progress=progress,\n                        progress_args=(app,))\n        '
        raise pyrogram.StopTransmission