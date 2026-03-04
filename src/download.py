import csv
import requests
import os
import time
import logging

API_KEY = os.getenv('XENO_CANTO_API_KEY')
if not API_KEY:
    raise EnvironmentError("XENO_CANTO_API_KEY environment variable is not set.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '../data/raw')
LOG_DIR = os.path.dirname(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler(os.path.join(LOG_DIR, '../logs/download.log'), encoding='utf-8', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s :: %(message)s'))
logger.addHandler(file_handler)

class Species:
    def __init__(self, species):
        self.species = species
        self.base_url= f'https://xeno-canto.org/api/3/recordings?query=sp:"{self.species}"&key={API_KEY}'
        with requests.get(self.base_url) as r:
            self.data = r.json()
        self.rows = []
        if not self.data.get('recordings'):
            raise ValueError("No recordings found for %s", self.species)
        self.english_name = '_'.join(self.data['recordings'][0]['en'].replace('-', ' ').split(' '))
        self.pages = self.data['numPages']

    def make_directory(self):
        os.makedirs(f"{OUTPUT_DIR}/{self.english_name}_mp3", exist_ok=True)

    def page_recordings(self, page):
        page_url = self.base_url + '&page=' + str(page)
        with requests.get(page_url) as r:
            page_data = r.json()
            logger.info("Page: %s", page_data['page'])
            recordings = page_data['recordings']
        return recordings

    def record_metadata(self, record):

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

    def download_audio(self, metadata):
        try:
            file_name = f"{self.english_name}_{metadata['id']}.mp3"

            if os.path.isfile(f"{OUTPUT_DIR}/{self.english_name}_mp3/{file_name}"):
                logger.info("File %s already exists", file_name)
                return

            response = requests.get(metadata['file'], stream=True, timeout=30)  # download
            with open(f"{OUTPUT_DIR}/{self.english_name}_mp3/{file_name}", 'wb') as f:  # save
                f.write(response.content)
                logger.info('Downloaded: %s', file_name)
        except Exception as e:
            logger.warning("❌ Failed %s: %s", metadata['id'], e)

    def write_csv(self):
        # Write information in .csv file
        with open(f'{OUTPUT_DIR}/{self.english_name}.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    def download(self):
        self.make_directory()
        # loop over every page
        for page in range(1, self.pages + 1):
            # get all recording's metadata in the page
            recordings = self.page_recordings(page)

            # loop over every recording instance
            for record in recordings:
                # get all metadata of a record and append to list rows which will be used to create .csv file
                metadata = self.record_metadata(record)
                self.rows.append(metadata)

                # download and save .mp3 audio file
                self.download_audio(metadata)

                time.sleep(0.5)
            time.sleep(1)

        logger.info("Downloaded %d recordings for %s", len(self.rows), self.english_name)
        self.write_csv()
