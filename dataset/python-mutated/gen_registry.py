from __future__ import annotations
import concurrent.futures
import functools
import json
import os
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any
import pins
import requests
import tqdm
from google.cloud import storage
import ibis
if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
EXAMPLES_DIRECTORY = Path(__file__).parent
pins.config.pins_options.quiet = True
Metadata = dict[str, dict[str, str] | None]

def make_descriptions(descriptions_dir: Path) -> Iterator[tuple[str, str]]:
    if False:
        return 10
    return ((file.name, file.read_text().strip()) for file in descriptions_dir.glob('*'))

def make_keys(registry: Path) -> dict[str, str]:
    if False:
        while True:
            i = 10
    return ((key.split(os.extsep, maxsplit=1)[0], key) for (key, _) in (row.split(maxsplit=1) for row in map(str.strip, registry.read_text().splitlines())))

def add_wowah_example(data_path, *, client: storage.Client, metadata: Metadata) -> None:
    if False:
        print('Hello World!')
    bucket = client.get_bucket('ibis-tutorial-data')
    args = []
    for blob in bucket.list_blobs(prefix='wowah_data'):
        name = blob.name
        if name.endswith('_raw.parquet'):
            tail = name.rsplit(os.sep, 1)[-1]
            path = data_path.joinpath(f'wowah_{tail}' if not tail.startswith('wowah') else tail)
            args.append((path, blob))
            metadata[path.with_suffix('').name] = {}
    with concurrent.futures.ThreadPoolExecutor() as e:
        for fut in concurrent.futures.as_completed((e.submit(lambda path, blob: path.write_bytes(blob.download_as_bytes()), path, blob) for (path, blob) in args)):
            fut.result()

def add_movielens_example(data_path: Path, *, metadata: Metadata, source_zip: Path | None=None):
    if False:
        return 10
    filename = 'ml-latest-small.zip'
    if source_zip is not None and source_zip.exists():
        raw_bytes = source_zip.read_bytes()
    else:
        resp = requests.get(f'https://files.grouplens.org/datasets/movielens/{filename}')
        resp.raise_for_status()
        raw_bytes = resp.content
    with tempfile.TemporaryDirectory() as d:
        con = ibis.duckdb.connect()
        d = Path(d)
        all_data = d / filename
        all_data.write_bytes(raw_bytes)
        with zipfile.ZipFile(all_data) as zf:
            members = [name for name in zf.namelist() if name.endswith('.csv')]
            zf.extractall(d, members=members)
        for (member, csv_path) in zip(members, map(d.joinpath, members)):
            parquet_path = data_path.joinpath(member.replace('ml-latest-small/', 'ml_latest_small_')).with_suffix('.parquet')
            metadata[parquet_path.with_suffix('').name] = {}
            con.read_csv(csv_path).to_parquet(parquet_path, codec='zstd')

def add_imdb_example(data_path: Path) -> None:
    if False:
        print('Hello World!')

    def convert_to_parquet(base: Path, *, con: ibis.backends.duckdb.Base, description: str, bar: tqdm.tqdm) -> None:
        if False:
            while True:
                i = 10
        dest = data_path.joinpath('imdb_' + Path(base).with_suffix('').with_suffix('.parquet').name.replace('.', '_', 1))
        con.read_csv(f'https://datasets.imdbws.com/{base}', nullstr='\\N', header=1, quote='').to_parquet(dest, compression='zstd')
        dest.parents[1].joinpath('descriptions', dest.with_suffix('').name).write_text(description)
        bar.update()
    meta = {'name.basics.tsv.gz': "Contains the following information for names:\n* nconst (string) - alphanumeric unique identifier of the name/person\n* primaryName (string) - name by which the person is most often credited\n* birthYear - in YYYY format\n* deathYear - in YYYY format if applicable, else '\\N'\n* primaryProfession (array of strings) - the top-3 professions of the person\n* knownForTitles (array of tconsts) - titles the person is known for", 'title.akas.tsv.gz': 'Contains the following information for titles:\n* titleId (string) - a tconst, an alphanumeric unique identifier of the title\n* ordering (integer) - a number to uniquely identify rows for a given titleId\n* title (string) - the localized title\n* region (string) - the region for this version of the title\n* language (string) - the language of the title\n* types (array) - Enumerated set of attributes for this alternative title. One or more of the following: "alternative", "dvd", "festival", "tv", "video", "working", "original", "imdbDisplay". New values may be added in the future without warning\n* attributes (array) - Additional terms to describe this alternative title, not enumerated\n* isOriginalTitle (boolean) - 0: not original title; 1: original title', 'title.basics.tsv.gz': "Contains the following information for titles:\n* tconst (string) - alphanumeric unique identifier of the title\n* titleType (string) - the type/format of the title (e.g. movie, short, tvseries, tvepisode, video, etc)\n* primaryTitle (string) - the more popular title / the title used by the filmmakers on promotional materials at the point of release\n* originalTitle (string) - original title, in the original language\n* isAdult (boolean) - 0: non-adult title; 1: adult title\n* startYear (YYYY) - represents the release year of a title. In the case of TV Series, it is the series start year\n* endYear (YYYY) - TV Series end year. '\\N' for all other title types\n* runtimeMinutes - primary runtime of the title, in minutes\n* genres (string array) - includes up to three genres associated with the title", 'title.crew.tsv.gz': 'Contains the director and writer information for all the titles in IMDb. Fields include:\n* tconst (string) - alphanumeric unique identifier of the title\n* directors (array of nconsts) - director(s) of the given title\n* writers (array of nconsts) - writer(s) of the given title', 'title.episode.tsv.gz': 'Contains the tv episode information. Fields include:\n* tconst (string) - alphanumeric identifier of episode\n* parentTconst (string) - alphanumeric identifier of the parent TV Series\n* seasonNumber (integer) - season number the episode belongs to\n* episodeNumber (integer) - episode number of the tconst in the TV series', 'title.principals.tsv.gz': "Contains the principal cast/crew for titles\n* tconst (string) - alphanumeric unique identifier of the title\n* ordering (integer) - a number to uniquely identify rows for a given titleId\n* nconst (string) - alphanumeric unique identifier of the name/person\n* category (string) - the category of job that person was in\n* job (string) - the specific job title if applicable, else '\\N'\n* characters (string) - the name of the character played if applicable, else '\\N'", 'title.ratings.tsv.gz': 'Contains the IMDb rating and votes information for titles\n* tconst (string) - alphanumeric unique identifier of the title\n* averageRating - weighted average of all the individual user ratings\n* numVotes - number of votes the title has received'}
    bar = tqdm.tqdm(total=len(meta))
    with concurrent.futures.ThreadPoolExecutor() as e:
        for fut in concurrent.futures.as_completed((e.submit(convert_to_parquet, base, con=ibis.duckdb.connect(), description=description, bar=bar) for (base, description) in meta.items())):
            fut.result()

def main(parser):
    if False:
        print('Hello World!')
    args = parser.parse_args()
    data_path = EXAMPLES_DIRECTORY / 'data'
    descriptions_path = EXAMPLES_DIRECTORY / 'descriptions'
    data_path.mkdir(parents=True, exist_ok=True)
    descriptions_path.mkdir(parents=True, exist_ok=True)
    metadata = {}
    add_movielens_example(data_path, metadata=metadata, source_zip=Path(ml_source_zip) if (ml_source_zip := args.movielens_source_zip) is not None else None)
    add_imdb_example(data_path)
    add_wowah_example(data_path, client=storage.Client(), metadata=metadata)
    subprocess.check_call(['Rscript', str(EXAMPLES_DIRECTORY / 'gen_examples.R')])
    verify_case(parser, metadata)
    if not args.dry_run:
        board = pins.board_gcs(args.bucket)

        def write_pin(path: Path, *, board: pins.Board, metadata: Metadata, bar: tqdm.tqdm) -> None:
            if False:
                print('Hello World!')
            pathname = path.name
            suffixes = path.suffixes
            name = pathname[:-sum(map(len, suffixes))]
            description = metadata.get(name, {}).get('description')
            board.pin_upload(paths=[str(path)], name=name, title=f'`{pathname}` dataset', description=description)
            bar.update()
        data_paths = list(data_path.glob('*'))
        write_pin = functools.partial(write_pin, board=board, metadata=metadata, bar=tqdm.tqdm(total=len(data_paths)))
        with concurrent.futures.ThreadPoolExecutor() as e:
            for fut in concurrent.futures.as_completed((e.submit(write_pin, path) for path in data_paths)):
                fut.result()
        metadata.update(((key, {'description': value}) for (key, value) in make_descriptions(descriptions_path)))
        with EXAMPLES_DIRECTORY.joinpath('metadata.json').open(mode='w') as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write('\n')

def verify_case(parser: argparse.ArgumentParser, data: Mapping[str, Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    counter = Counter(map(str.lower, data.keys()))
    invalid_keys = [key for (key, count) in counter.items() if count > 1]
    if invalid_keys:
        parser.error(f'keys {invalid_keys} are incompatible with case-insensitive file systems')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Set up the pin board from a GCS bucket.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--bucket', default=ibis.examples._BUCKET, help='GCS bucket in which to store data')
    parser.add_argument('-I', '--imdb-source-dir', help='Directory containing imdb source data', default=None, type=str)
    parser.add_argument('-M', '--movielens-source-zip', help='MovieLens data zip file', default=None, type=str)
    parser.add_argument('-d', '--dry-run', action='store_true', help='Avoid executing any code that writes to the example data bucket')
    main(parser)