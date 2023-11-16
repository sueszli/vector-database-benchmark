class RetryIntervalCalculator:
    """Retry interval calculator interface."""

    def calculate_sleep_duration(self, current_attempt: int) -> float:
        if False:
            i = 10
            return i + 15
        'Calculates an interval duration in seconds.\n\n        Args:\n            current_attempt: the number of the current attempt (zero-origin; 0 means no retries are done so far)\n        Returns:\n            calculated interval duration in seconds\n        '
        raise NotImplementedError()