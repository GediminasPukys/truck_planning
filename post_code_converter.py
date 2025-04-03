import asyncio
import aiohttp
import pandas as pd
from typing import Optional, Set, List, Dict
from dataclasses import dataclass
import os
import json
from pathlib import Path
import logging
from datetime import datetime
from functools import partial
from itertools import islice

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Location:
    postcode: str
    country_code: str
    latitude: float
    longitude: float
    timestamp: str

    def to_dict(self) -> dict:
        return {
            'postcode': self.postcode,
            'country_code': self.country_code,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'timestamp': self.timestamp
        }


class AsyncGeocoder:
    def __init__(self, results_dir: str = None, max_concurrent: int = 5):
        """
        Initialize the geocoder with results directory and concurrency control.
        """
        if results_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.join(script_dir, 'results')

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.processed_postcodes: Set[str] = set()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.load_existing_results()

    def _get_country_name(self, country_code: str) -> str:
        """Convert country code to full name for better geocoding results."""
        country_mapping = {
            'AT': 'Austria',
            'DE': 'Germany',
            'CH': 'Switzerland',
            'FR': 'France',
            'IT': 'Italy',
            'LT': 'Lithuania',
            'NL': 'Netherlands'
        }
        return country_mapping.get(country_code, country_code)

    def _get_result_file_path(self, postcode: str, country_code: str) -> Path:
        """Get the path for the result file."""
        sanitized_postcode = postcode.replace(' ', '_')
        return self.results_dir / f"{country_code}_{sanitized_postcode}.json"

    def load_existing_results(self) -> None:
        """Load all previously processed postcodes."""
        for result_file in self.results_dir.glob('*.json'):
            postcode = result_file.stem.split('_', 1)[1].replace('_', ' ')
            self.processed_postcodes.add(postcode)
        logger.info(f"Loaded {len(self.processed_postcodes)} existing results")

    async def get_coordinates_with_retry(
            self,
            postcode: str,
            country_code: str,
            session: aiohttp.ClientSession,
            max_retries: int = 3
    ) -> Optional[Location]:
        """
        Try to get coordinates with retries on failure.
        """
        async with self.semaphore:  # Control concurrency
            for attempt in range(max_retries):
                try:
                    result = await self._get_coordinates(postcode, country_code, session)
                    if result:
                        return result
                    await asyncio.sleep(1 + attempt)  # Exponential backoff
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed for {postcode}: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
            return None

    async def _get_coordinates(
            self,
            postcode: str,
            country_code: str,
            session: aiohttp.ClientSession
    ) -> Optional[Location]:
        """
        Core function to convert a single postcode to coordinates.
        """
        result_file = self._get_result_file_path(postcode, country_code)

        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                return Location(**data)

        country_name = self._get_country_name(country_code)
        search_query = f"{postcode}, {country_name}"

        headers = {'User-Agent': 'PostcodeConverter/1.0'}
        params = {
            'q': search_query,
            'format': 'json',
            'limit': 1
        }

        async with session.get(
                "https://nominatim.openstreetmap.org/search",
                headers=headers,
                params=params
        ) as response:
            if response.status == 200:
                results = await response.json()
                if results:
                    location = Location(
                        postcode=postcode,
                        country_code=country_code,
                        latitude=float(results[0]['lat']),
                        longitude=float(results[0]['lon']),
                        timestamp=datetime.now().isoformat()
                    )

                    with open(result_file, 'w') as f:
                        json.dump(location.to_dict(), f, indent=2)

                    logger.info(f"Successfully processed {country_code} {postcode}")
                    return location

            logger.warning(f"No results found for {postcode}, {country_name}")
            return None


async def process_postcodes_batch(
        postcodes: List[Dict],
        geocoder: AsyncGeocoder,
        session: aiohttp.ClientSession
) -> List[Optional[Location]]:
    """Process a batch of postcodes concurrently."""
    tasks = []
    for item in postcodes:
        full_post = item['POST_CODE'].strip()
        if len(full_post) < 4:
            continue

        country_code = full_post[:2]
        postcode = full_post[3:]

        if postcode in geocoder.processed_postcodes:
            logger.info(f"Skipping already processed postcode: {postcode}")
            continue

        task = geocoder.get_coordinates_with_retry(postcode, country_code, session)
        tasks.append(task)

    return await asyncio.gather(*tasks)


async def process_all_postcodes(input_file: str, batch_size: int = 10):
    """
    Process all postcodes in batches with proper rate limiting.
    """
    geocoder = AsyncGeocoder(max_concurrent=5)  # Limit concurrent requests

    # Read input file
    df = pd.read_csv(input_file)
    total_records = len(df)
    logger.info(f"Processing {total_records} postcodes in batches of {batch_size}")

    # Process in batches
    connector = aiohttp.TCPConnector(limit_per_host=5)
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i in range(0, total_records, batch_size):
            batch = df.iloc[i:i + batch_size].to_dict('records')
            logger.info(f"Processing batch {i // batch_size + 1}/{(total_records + batch_size - 1) // batch_size}")

            results = await process_postcodes_batch(batch, geocoder, session)
            valid_results = [r for r in results if r is not None]
            logger.info(f"Batch complete: {len(valid_results)} successful out of {len(batch)}")

            # Add delay between batches to respect rate limits
            await asyncio.sleep(2)


async def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "/Users/gpukys/Downloads/post_codes2.csv")
    results_dir = os.path.join(script_dir, "results")

    print(f"Script directory: {script_dir}")
    print(f"Input file path: {input_file}")
    print(f"Results directory: {results_dir}")

    await process_all_postcodes(input_file)


if __name__ == "__main__":
    asyncio.run(main())