import csv
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
from logger import get_logger
logger = get_logger(__name__, 'download.log')

# API key import
load_dotenv()
API_KEY = os.getenv('XENO_CANTO_API_KEY')
if not API_KEY:
    raise ValueError("XENO_CANTO_API_KEY is not set")

class Species:
    def __init__(self, species: str, raw_dir: Path):
        self.RAW_DIR = raw_dir
        self.base_url= f'https://xeno-canto.org/api/3/recordings?query=sp:"{species}"&key={API_KEY}'
        with requests.get(self.base_url) as r: # get all metadata
            self.data = r.json()
        self.english_name = '_'.join(self.data["recordings"][0]['en'].replace('-', ' ').split(' '))

    def page_records(self, page: int):
        page_url = self.base_url + '&page=' + str(page)
        with requests.get(page_url) as r:
            page_data = r.json()
            logger.info("Page: %s", page_data['page'])
            recordings = page_data['recordings']
        return recordings

    @staticmethod # still called as obj.record_metadata(...) but no self
    def record_metadata(record: dict):
        return {
            # Identification
            'id': record.get('id', None),
            'gen': record.get('gen', None),
            'sp': record.get('sp', None),
            'ssp': record.get('ssp', None),
            'grp': record.get('grp', None),
            'en': record.get('en', None),

            # Recording details
            'rec': record.get('rec', None),
            'cnt': record.get('cnt', None),
            'loc': record.get('loc', None),
            'lat': record.get('lat', None),
            'lng': record.get('lng', None),

            # Audio characteristics
            'type': record.get('type', None),
            'sex': record.get('sex', None),
            'stage': record.get('stage', None),
            'method': record.get('method', None),

            # URLs and files
            'url': record.get('url', None),
            'file': record.get('file', None),
            'file-name': record.get('file-name', None),
            'sono': record.get('sono', None),
            'osci': record.get('osci', None),

            # License and quality
            'lic': record.get('lic', None),
            'q': record.get('q', None),

            # Temporal data
            'length': record.get('length', None),
            'time': record.get('time', None),
            'date': record.get('date', None),
            'uploaded': record.get('uploaded', None),

            # Additional information
            'also': record.get('also', None),
            'rmk': record.get('rmk', None),
            'animal-seen': record.get('animal-seen', None),
            'playback-used': record.get('playback-used', None),
            'temp': record.get('temp', None),
            'regnr': record.get('regnr', None),
            'auto': record.get('auto', None),
            'dvc': record.get('dvc', None),
            'mic': record.get('mic', None),
            'smp': record.get('smp', None),
        }

    def download_audio(self, metadata: dict):
        file_name = f"{self.english_name}_{metadata['id']}.mp3"
        if os.path.isfile(f"{self.RAW_DIR}/{self.english_name}_mp3/{file_name}"): # Skip already downloaded audio files
            logger.info("File %s already exists", file_name)
            return
        response = requests.get(metadata['file'], stream=True, timeout=30)  # download
        with open(f"{self.RAW_DIR}/{self.english_name}_mp3/{file_name}", 'wb') as f:  # save
            f.write(response.content)
            logger.info('Downloaded: %s', file_name)

    def write_csv(self, records: list):
        with open(f'{self.RAW_DIR}/{self.english_name}.csv', 'w', encoding='utf-8', newline='') as f: # Write information in .csv file
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    def download(self):
        # create folder for the species which store all the downloaded .mp3 files
        folder_path = self.RAW_DIR / self.english_name / "_mp3"
        os.makedirs(folder_path, exist_ok=True)

        records_list = []
        # loop over every page
        for page in range(1, self.data["numPages"] + 1):

            # get metadata of all the recordings on the page
            records = self.page_records(page)

            # loop over every record
            for record in records:
                # get all metadata of a record and append to list rows which will be used to create .csv file
                metadata = self.record_metadata(record)
                records_list.append(metadata)

                # download and save .mp3 audio file
                self.download_audio(metadata)

        logger.info("Downloaded %d recordings for %s", len(records_list), self.english_name)

        # write records to a .csv file
        self.write_csv(records_list)
