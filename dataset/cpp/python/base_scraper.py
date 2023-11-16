"""Base Scraper."""
import datetime
import logging
import re
import typing
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

logger = logging.getLogger("udemy_enroller")


class ScraperStates(Enum):
    """Scraper states."""

    DISABLED = "DISABLED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"


class BaseScraper(ABC):
    """Base scraper logic."""

    def __init__(self):
        """Initialize."""
        self._state = None
        self.scraper_name = None
        self.max_pages = None
        self.last_page = None
        self.current_page = 0

    @abstractmethod
    async def run(self):
        """Run method that must be implemented in subclasses."""
        return

    @abstractmethod
    async def get_links(self) -> typing.List[str]:
        """Get links method that must be implemented in subclasses."""
        return

    @property
    def state(self) -> str:
        """State property."""
        return self._state

    @state.setter
    def state(self, value) -> None:
        """Set the state of the scraper."""
        if any([ss for ss in ScraperStates if ss.value == value]):
            self._state = value

    def set_state_disabled(self) -> None:
        """Set state to disable."""
        self.state = ScraperStates.DISABLED.value
        logger.info(f"{self.scraper_name} scraper disabled")

    def set_state_running(self) -> None:
        """Set state to running."""
        self.state = ScraperStates.RUNNING.value
        logger.info(f"{self.scraper_name} scraper is running")

    def set_state_complete(self) -> None:
        """Set state to complete."""
        self.state = ScraperStates.COMPLETE.value
        logger.info(f"{self.scraper_name} scraper complete")

    def is_disabled(self) -> bool:
        """Determine whether a scraper is disabled."""
        return self.state == ScraperStates.DISABLED.value

    def is_complete(self) -> bool:
        """Determine whether a scraper has completed."""
        return self.state == ScraperStates.COMPLETE.value

    def should_run(self) -> bool:
        """Determine whether a scraper should run."""
        should_run = not self.is_disabled() and not self.is_complete()
        if should_run:
            self.set_state_running()
        return should_run

    @staticmethod
    def time_run(func):
        """Log execution time of the function that is wrapped."""

        async def wrapper(self):
            start_time = datetime.datetime.utcnow()
            try:
                response = await func(self)
            except Exception as e:
                logger.exception(
                    f"Error while running {self.scraper_name} scraper: {e}"
                )
                self.is_complete()
                return []
            end_time = datetime.datetime.utcnow()
            logger.info(
                f"Got {len(response)} links from {self.DOMAIN} in {(end_time - start_time).total_seconds():.2f} seconds"
            )
            return response

        return wrapper

    def max_pages_reached(self) -> bool:
        """
        Return a boolean of whether we should continue checking site.

        :return:
        """
        should_run = True

        if self.max_pages is not None:
            should_run = self.max_pages > self.current_page

            if not should_run:
                logger.info(
                    f"Stopping loop. We have reached max number of pages to scrape: {self.max_pages}"
                )
                self.set_state_complete()

        if self.last_page == self.current_page:
            logger.info(
                f"Stopping loop. We have reached the last page to scrape: {self.last_page}"
            )
            self.set_state_complete()

        return should_run

    @staticmethod
    def validate_coupon_url(url: str) -> Optional[str]:
        """
        Validate the udemy coupon url passed in.

        If it matches the pattern it is returned else it returns None
        :param url: The url to check the udemy coupon pattern for
        :return: The validated url or None
        """
        url_pattern = r"^https:\/\/www.udemy.com.*couponCode=.*$"
        matching = re.match(url_pattern, url)
        if matching is not None:
            matching = matching.group()
        return matching
