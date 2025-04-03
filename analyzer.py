import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from scipy.spatial.distance import cdist
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN  # Correct import syntax
from statistics import mean, stdev

"""
Required packages:
pip install pandas numpy scipy scikit-learn matplotlib

Note: Although we import 'sklearn', make sure to install it using 'pip install scikit-learn'
"""

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = Path(results_dir)
        self.df = None
        self.outliers = {}

    def read_json_files(self) -> pd.DataFrame:
        """Read all JSON files from results directory into a DataFrame."""
        data = []
        for json_file in self.results_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    record = json.load(f)
                    data.append(record)
            except Exception as e:
                logger.error(f"Error reading {json_file}: {str(e)}")

        if not data:
            raise ValueError("No valid JSON files found")

        df = pd.DataFrame(data)
        logger.info(f"Read {len(df)} records from JSON files")
        return df

    def detect_outliers_by_country(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Detect coordinate outliers for each country using multiple methods:
        1. DBSCAN clustering
        2. Distance from country centroid
        3. Statistical outliers (z-score)
        """
        outliers = {}

        for country in df['country_code'].unique():
            country_data = df[df['country_code'] == country].copy()
            if len(country_data) < 2:
                continue

            # Create coordinate array
            coords = country_data[['latitude', 'longitude']].values

            # Method 1: DBSCAN Clustering
            eps = 2.0  # 2 degrees ~ roughly 200km
            min_samples = 2
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            dbscan_outliers = (db.labels_ == -1)

            # Method 2: Distance from centroid
            centroid = coords.mean(axis=0)
            distances = cdist([centroid], coords)[0]
            distance_threshold = np.percentile(distances, 95)
            distance_outliers = distances > distance_threshold

            # Method 3: Statistical outliers
            z_scores = np.abs(stats.zscore(coords))
            stat_outliers = np.any(z_scores > 3, axis=1)

            # Combine outlier detection methods
            combined_outliers = (dbscan_outliers | distance_outliers | stat_outliers)

            if combined_outliers.any():
                outliers[country] = country_data[combined_outliers].copy()
                outliers[country]['outlier_distance'] = distances[combined_outliers]

                logger.info(f"Found {combined_outliers.sum()} outliers in {country}")

        return outliers

    def analyze_country_distributions(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate statistical distributions for each country."""
        stats = {}

        for country in df['country_code'].unique():
            country_data = df[df['country_code'] == country]

            stats[country] = {
                'lat_mean': country_data['latitude'].mean(),
                'lat_std': country_data['latitude'].std(),
                'lon_mean': country_data['longitude'].mean(),
                'lon_std': country_data['longitude'].std(),
                'count': len(country_data),
                'bbox': {
                    'min_lat': country_data['latitude'].min(),
                    'max_lat': country_data['latitude'].max(),
                    'min_lon': country_data['longitude'].min(),
                    'max_lon': country_data['longitude'].max()
                }
            }

        return stats

    def process_results(self):
        """Main processing function."""
        # Read all JSON files
        df = self.read_json_files()
        self.df = df

        # Detect outliers
        self.outliers = self.detect_outliers_by_country(df)

        # Calculate country statistics
        self.country_stats = self.analyze_country_distributions(df)

        # Save consolidated results
        self.save_results()

    def save_results(self):
        """Save consolidated results and analysis."""
        # Save main CSV
        output_file = self.results_dir / 'consolidated_results.csv'
        self.df.to_csv(output_file, index=False)
        logger.info(f"Saved consolidated results to {output_file}")

        # Save outliers
        outliers_file = self.results_dir / 'outliers.csv'
        outliers_df = pd.concat(self.outliers.values()) if self.outliers else pd.DataFrame()
        outliers_df.to_csv(outliers_file, index=False)
        logger.info(f"Saved outliers to {outliers_file}")

        # Save country statistics
        stats_file = self.results_dir / 'country_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(self.country_stats, f, indent=2)
        logger.info(f"Saved country statistics to {stats_file}")

        # Generate summary report
        self.generate_report()

    def generate_report(self):
        """Generate a detailed analysis report."""
        report_file = self.results_dir / 'analysis_report.txt'

        with open(report_file, 'w') as f:
            f.write("Geocoding Results Analysis Report\n")
            f.write("================================\n\n")

            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"Total records processed: {len(self.df)}\n")
            f.write(f"Number of countries: {self.df['country_code'].nunique()}\n")
            f.write("\n")

            # Country-specific statistics
            f.write("Country Statistics:\n")
            for country, stats in self.country_stats.items():
                f.write(f"\n{country}:\n")
                f.write(f"  Records: {stats['count']}\n")
                f.write(f"  Latitude range: {stats['bbox']['min_lat']:.4f} to {stats['bbox']['max_lat']:.4f}\n")
                f.write(f"  Longitude range: {stats['bbox']['min_lon']:.4f} to {stats['bbox']['max_lon']:.4f}\n")

                if country in self.outliers:
                    outlier_count = len(self.outliers[country])
                    f.write(f"  Outliers detected: {outlier_count}\n")

            logger.info(f"Generated analysis report at {report_file}")


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'

    analyzer = ResultsAnalyzer(results_dir)
    analyzer.process_results()


if __name__ == "__main__":
    main()