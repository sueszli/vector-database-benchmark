#!/usr/bin/env python
"""Download packages from appveyor artifacts
"""

import os
import re
import sys
import logging
import datetime as dt
from pathlib import Path
from argparse import ArgumentParser

import requests

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

API_URL = "https://ci.appveyor.com/api"
REPOS = "psycopg/psycopg2"
WORKFLOW_NAME = "Build packages"


class ScriptError(Exception):
    """Controlled exception raised by the script."""


def main():
    opt = parse_cmdline()
    try:
        token = os.environ["APPVEYOR_TOKEN"]
    except KeyError:
        raise ScriptError("please set a APPVEYOR_TOKEN to download artifacts")

    s = requests.Session()
    s.headers["Content-Type"] = "application/json"
    s.headers["Authorization"] = f"Bearer {token}"

    if opt.build:
        logger.info("fetching build %s", opt.build)
        resp = s.get(f"{API_URL}/projects/{REPOS}/build/{opt.build}")
    else:
        logger.info("fetching last run")
        resp = s.get(f"{API_URL}/projects/{REPOS}")

    resp.raise_for_status()
    data = resp.json()

    updated_at = dt.datetime.fromisoformat(
        re.sub(r"\.\d+", "", data["build"]["finished"])
    )
    now = dt.datetime.now(dt.timezone.utc)
    age = now - updated_at
    logger.info(
        f"found build {data['build']['version']} updated {pretty_interval(age)} ago"
    )
    if age > dt.timedelta(hours=6):
        logger.warning("maybe it's a bit old?")

    jobs = data["build"]["jobs"]
    for job in jobs:
        if job["status"] != "success":
            raise ScriptError(f"status for job {job['jobId']} is {job['status']}")

        logger.info(f"fetching artifacts info for {job['name']}")
        resp = s.get(f"{API_URL}/buildjobs/{job['jobId']}/artifacts/")
        resp.raise_for_status()
        afs = resp.json()
        for af in afs:
            fn = af["fileName"]
            if fn.startswith("dist/"):
                fn = fn.split("/", 1)[1]
            dest = Path("wheelhouse") / fn
            logger.info(f"downloading {dest}")
            resp = s.get(
                f"{API_URL}/buildjobs/{job['jobId']}/artifacts/{af['fileName']}"
            )
            resp.raise_for_status()
            if not dest.parent.exists():
                dest.parent.mkdir(parents=True)

            with dest.open("wb") as f:
                f.write(resp.content)

    logger.info("now you can run: 'twine upload -s wheelhouse/*'")


def parse_cmdline():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--build", help="build version to download [default: latest]")
    opt = parser.parse_args()
    return opt


def pretty_interval(td):
    secs = td.total_seconds()
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{int(days)} days, {int(hours)} hours, {int(mins)} minutes"
    elif hours:
        return f"{int(hours)} hours, {int(mins)} minutes"
    else:
        return f"{int(mins)} minutes"


if __name__ == "__main__":
    try:
        sys.exit(main())

    except ScriptError as e:
        logger.error("%s", e)
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("user interrupt")
        sys.exit(1)
