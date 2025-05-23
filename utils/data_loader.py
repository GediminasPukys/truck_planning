# utils/data_loader.py (Updated with Premium column support and extended validation)
import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from geopy.distance import geodesic


class DataValidator:
    """Class to handle data validation rules and checks"""

    @staticmethod
    def validate_coordinates(lat, lon):
        """Validate latitude and longitude values"""
        try:
            lat = float(lat)
            lon = float(lon)
            return -90 <= lat <= 90 and -180 <= lon <= 180
        except:
            return False

    @staticmethod
    def validate_timestamp(timestamp):
        """Validate timestamp format"""
        try:
            if isinstance(timestamp, str):
                pd.to_datetime(timestamp)
            return True
        except:
            return False

    @staticmethod
    def validate_numeric(value):
        """Validate numeric values"""
        try:
            float_val = float(str(value).replace(',', '.'))
            return float_val > 0
        except:
            return False

    @staticmethod
    def validate_premium(value):
        """Validate premium values (can be zero but not negative)"""
        try:
            float_val = float(str(value).replace(',', '.'))
            return float_val >= 0
        except:
            return False


class ExtendedDataValidator(DataValidator):
    """Extended data validator with time horizon analysis"""

    @staticmethod
    def analyze_time_span(df, start_col, end_col=None):
        """Analyze time span of data"""
        try:
            df[start_col] = pd.to_datetime(df[start_col])
            if end_col:
                df[end_col] = pd.to_datetime(df[end_col])
                earliest = df[start_col].min()
                latest = df[end_col].max()
            else:
                earliest = df[start_col].min()
                latest = df[start_col].max()

            return {
                'earliest': earliest,
                'latest': latest,
                'span_days': (latest - earliest).days,
                'total_records': len(df)
            }
        except:
            return None

    @staticmethod
    def validate_time_windows(df, start_col, end_col):
        """Validate that time windows are logical"""
        try:
            df[start_col] = pd.to_datetime(df[start_col])
            df[end_col] = pd.to_datetime(df[end_col])

            # Check for invalid windows (start >= end)
            invalid_windows = df[start_col] >= df[end_col]

            # Check for very short windows (less than 1 hour)
            short_windows = (df[end_col] - df[start_col]).dt.total_seconds() < 3600

            # Check for very long windows (more than 7 days)
            long_windows = (df[end_col] - df[start_col]).dt.days > 7

            return {
                'invalid_count': invalid_windows.sum(),
                'invalid_indices': list(df[invalid_windows].index),
                'short_count': short_windows.sum(),
                'long_count': long_windows.sum(),
                'avg_window_hours': (df[end_col] - df[start_col]).dt.total_seconds().mean() / 3600
            }
        except:
            return None


class DataLoader:
    """Class to handle data loading and validation"""

    def __init__(self):
        self.validator = ExtendedDataValidator()

    def load_trucks_data(self, file):
        """
        Load and validate trucks data from CSV file

        Required columns:
        - truck_id: unique identifier
        - truck type: type of truck
        - Address (drop off): drop-off location name
        - Latitude (dropoff): drop-off latitude
        - Longitude (dropoff): drop-off longitude
        - Timestamp (dropoff): drop-off time
        - avg moving speed, km/h: average speed
        - price per km, Eur: price per kilometer
        - waiting time price per h, EUR: waiting time price
        """
        try:
            # Read CSV file
            df = pd.read_csv(file)

            # Check required columns
            required_columns = [
                'truck_id',
                'truck type',
                'Address (drop off)',
                'Latitude (dropoff)',
                'Longitude (dropoff)',
                'Timestamp (dropoff)',
                'avg moving speed, km/h',
                'price per km, Eur',
                'waiting time price per h, EUR'
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns in trucks file: {', '.join(missing_columns)}")
                return None

            # Validate data
            validation_errors = []

            # Check for duplicate truck IDs
            if df['truck_id'].duplicated().any():
                validation_errors.append("Duplicate truck IDs found")

            # Validate coordinates
            invalid_coords = df.apply(
                lambda row: not self.validator.validate_coordinates(
                    row['Latitude (dropoff)'],
                    row['Longitude (dropoff)']
                ),
                axis=1
            )
            if invalid_coords.any():
                validation_errors.append(
                    f"Invalid coordinates found in rows: {list(df[invalid_coords].index)}"
                )

            # Validate timestamp
            invalid_timestamps = df.apply(
                lambda row: not self.validator.validate_timestamp(row['Timestamp (dropoff)']),
                axis=1
            )
            if invalid_timestamps.any():
                validation_errors.append(
                    f"Invalid timestamps found in rows: {list(df[invalid_timestamps].index)}"
                )

            # Validate numeric values
            numeric_columns = [
                'avg moving speed, km/h',
                'price per km, Eur',
                'waiting time price per h, EUR'
            ]

            for col in numeric_columns:
                try:
                    invalid_numbers = df.apply(
                        lambda row: not self.validator.validate_numeric(row[col]),
                        axis=1
                    )
                    if invalid_numbers.any():
                        validation_errors.append(
                            f"Invalid {col} values found in rows: {list(df[invalid_numbers].index)}"
                        )
                except Exception as e:
                    validation_errors.append(f"Error validating {col}: {str(e)}")

            # Report validation errors
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                return None

            # Clean and convert data types
            df['Timestamp (dropoff)'] = pd.to_datetime(df['Timestamp (dropoff)'])
            df['Latitude (dropoff)'] = pd.to_numeric(df['Latitude (dropoff)'])
            df['Longitude (dropoff)'] = pd.to_numeric(df['Longitude (dropoff)'])
            df['avg moving speed, km/h'] = pd.to_numeric(df['avg moving speed, km/h'])
            df['price per km, Eur'] = pd.to_numeric(df['price per km, Eur'].astype(str).str.replace(',', '.'))
            df['waiting time price per h, EUR'] = pd.to_numeric(
                df['waiting time price per h, EUR'].astype(str).str.replace(',', '.'))

            # Extended validation: Check time span
            time_analysis = self.validator.analyze_time_span(df, 'Timestamp (dropoff)')
            if time_analysis:
                st.info(f"Trucks time span: {time_analysis['span_days']} days "
                        f"({time_analysis['earliest'].strftime('%Y-%m-%d')} to "
                        f"{time_analysis['latest'].strftime('%Y-%m-%d')})")

            # Add validation success message
            st.success(f"Successfully loaded {len(df)} trucks")

            return df

        except Exception as e:
            st.error(f"Error loading trucks file: {str(e)}")
            return None

    def load_cargo_data(self, file):
        """
        Load and validate cargo data from CSV file - ENHANCED VERSION

        Required columns:
        - Origin: origin location name
        - Origin_Latitude: origin latitude
        - Origin_Longitude: origin longitude
        - Available_From: availability start time
        - Available_To: availability end time
        - Delivery_Location: delivery location name
        - Delivery_Latitude: delivery latitude
        - Delivery_Longitude: delivery longitude
        - Cargo_Type: type of cargo
        - Premium: revenue for delivering cargo (optional, will be generated if missing)
        """
        try:
            # Read CSV file
            df = pd.read_csv(file)

            # Check required columns
            required_columns = [
                'Origin',
                'Origin_Latitude',
                'Origin_Longitude',
                'Available_From',
                'Available_To',
                'Delivery_Location',
                'Delivery_Latitude',
                'Delivery_Longitude',
                'Cargo_Type'
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns in cargo file: {', '.join(missing_columns)}")
                return None

            # Handle Premium column
            premium_generated = False
            if 'Premium' not in df.columns:
                st.warning("Premium column not found in cargo data. Generating based on distance and cargo type.")
                premium_generated = True
                df['Premium'] = self.generate_premium_values(df)
            else:
                # Validate existing premium values
                df['Premium'] = pd.to_numeric(df['Premium'], errors='coerce')

                # Handle invalid/missing premium values
                invalid_premiums = df['Premium'].isna() | (df['Premium'] < 0)
                if invalid_premiums.any():
                    st.warning(
                        f"Invalid Premium values found in {invalid_premiums.sum()} rows. Regenerating those values.")
                    df.loc[invalid_premiums, 'Premium'] = self.generate_premium_values(df.loc[invalid_premiums])

            # Validate data
            validation_errors = []

            # Validate origin coordinates
            try:
                invalid_origin_coords = df.apply(
                    lambda row: not self.validator.validate_coordinates(
                        row['Origin_Latitude'],
                        row['Origin_Longitude']
                    ),
                    axis=1
                )
                if invalid_origin_coords.any():
                    validation_errors.append(
                        f"Invalid origin coordinates found in rows: {list(df[invalid_origin_coords].index)}"
                    )
            except Exception as e:
                validation_errors.append(f"Error validating origin coordinates: {str(e)}")

            # Validate delivery coordinates
            try:
                invalid_delivery_coords = df.apply(
                    lambda row: not self.validator.validate_coordinates(
                        row['Delivery_Latitude'],
                        row['Delivery_Longitude']
                    ),
                    axis=1
                )
                if invalid_delivery_coords.any():
                    validation_errors.append(
                        f"Invalid delivery coordinates found in rows: {list(df[invalid_delivery_coords].index)}"
                    )
            except Exception as e:
                validation_errors.append(f"Error validating delivery coordinates: {str(e)}")

            # Validate timestamps
            try:
                invalid_from_times = df.apply(
                    lambda row: not self.validator.validate_timestamp(row['Available_From']),
                    axis=1
                )
                if invalid_from_times.any():
                    validation_errors.append(
                        f"Invalid Available_From timestamps found in rows: {list(df[invalid_from_times].index)}"
                    )
            except Exception as e:
                validation_errors.append(f"Error validating Available_From timestamps: {str(e)}")

            try:
                invalid_to_times = df.apply(
                    lambda row: not self.validator.validate_timestamp(row['Available_To']),
                    axis=1
                )
                if invalid_to_times.any():
                    validation_errors.append(
                        f"Invalid Available_To timestamps found in rows: {list(df[invalid_to_times].index)}"
                    )
            except Exception as e:
                validation_errors.append(f"Error validating Available_To timestamps: {str(e)}")

            # Clean and convert data types first
            df['Available_From'] = pd.to_datetime(df['Available_From'])
            df['Available_To'] = pd.to_datetime(df['Available_To'])

            # Extended time window validation
            time_window_analysis = self.validator.validate_time_windows(df, 'Available_From', 'Available_To')
            if time_window_analysis:
                if time_window_analysis['invalid_count'] > 0:
                    validation_errors.append(
                        f"Invalid time windows (From >= To) found in rows: {time_window_analysis['invalid_indices']}"
                    )

                # Report insights about time windows
                if time_window_analysis['short_count'] > 0:
                    st.warning(
                        f"{time_window_analysis['short_count']} cargo items have very short time windows (< 1 hour)")

                if time_window_analysis['long_count'] > 0:
                    st.info(f"{time_window_analysis['long_count']} cargo items have long time windows (> 7 days)")

                st.info(f"Average cargo availability window: {time_window_analysis['avg_window_hours']:.1f} hours")

            # Report validation errors
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                return None

            # Clean and convert remaining data types
            df['Origin_Latitude'] = pd.to_numeric(df['Origin_Latitude'])
            df['Origin_Longitude'] = pd.to_numeric(df['Origin_Longitude'])
            df['Delivery_Latitude'] = pd.to_numeric(df['Delivery_Latitude'])
            df['Delivery_Longitude'] = pd.to_numeric(df['Delivery_Longitude'])
            df['Premium'] = pd.to_numeric(df['Premium'])

            # Extended validation: Analyze time span and distribution
            time_analysis = self.validator.analyze_time_span(df, 'Available_From', 'Available_To')
            if time_analysis:
                st.info(f"Cargo time span: {time_analysis['span_days']} days "
                        f"({time_analysis['earliest'].strftime('%Y-%m-%d')} to "
                        f"{time_analysis['latest'].strftime('%Y-%m-%d')})")

                if time_analysis['span_days'] > 30:
                    st.info(
                        "📅 Long time horizon detected - consider using Extended optimization mode for better results")

            # Premium statistics
            premium_stats = df['Premium'].describe()
            st.info(f"Premium range: €{premium_stats['min']:.2f} - €{premium_stats['max']:.2f} "
                    f"(avg: €{premium_stats['mean']:.2f})")

            if premium_generated:
                st.info(
                    "💡 Premium values were automatically generated. You can replace them with actual revenue data for better optimization.")

            # Add validation success message
            st.success(f"Successfully loaded {len(df)} cargo items")

            return df

        except Exception as e:
            st.error(f"Error loading cargo file: {str(e)}")
            return None

    def generate_premium_values(self, cargo_df):
        """Generate realistic premium values based on distance and cargo type"""
        premiums = []

        # Premium multipliers by cargo type
        type_multipliers = {
            'general': 1.0,
            'frozen': 1.3,  # Higher premium for temperature-controlled
            'liquid': 1.2,  # Higher premium for specialized handling
            'hazardous': 1.5,  # Highest premium for dangerous goods
        }

        for _, cargo in cargo_df.iterrows():
            try:
                # Calculate distance
                origin = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])
                delivery = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])
                distance = geodesic(origin, delivery).kilometers

                # Base premium calculation
                base_premium = distance * np.random.uniform(1.5, 2.5)  # €1.5-2.5 per km

                # Apply cargo type multiplier
                cargo_type = str(cargo['Cargo_Type']).lower()
                multiplier = type_multipliers.get(cargo_type, 1.0)

                # Add some randomness for market variation
                final_premium = base_premium * multiplier * np.random.uniform(0.9, 1.1)

                # Round to reasonable precision
                premiums.append(round(final_premium, 2))

            except Exception as e:
                # Fallback to a reasonable default
                premiums.append(round(np.random.uniform(200, 800), 2))

        return premiums

    def validate_truck_cargo_compatibility(self, trucks_df, cargo_df):
        """Extended compatibility validation"""
        if trucks_df is None or cargo_df is None:
            return False

        try:
            # Type compatibility
            truck_types = set(trucks_df['truck type'].str.lower())
            cargo_types = set(cargo_df['Cargo_Type'].str.lower())
            matching_types = truck_types.intersection(cargo_types)

            if not matching_types:
                st.error("❌ No matching truck and cargo types found!")
                st.write("**Truck types:**", list(truck_types))
                st.write("**Cargo types:**", list(cargo_types))
                return False

            # Time compatibility
            trucks_df['Timestamp (dropoff)'] = pd.to_datetime(trucks_df['Timestamp (dropoff)'])
            cargo_df['Available_From'] = pd.to_datetime(cargo_df['Available_From'])
            cargo_df['Available_To'] = pd.to_datetime(cargo_df['Available_To'])

            earliest_truck = trucks_df['Timestamp (dropoff)'].min()
            latest_truck = trucks_df['Timestamp (dropoff)'].max()
            earliest_cargo = cargo_df['Available_From'].min()
            latest_cargo = cargo_df['Available_To'].max()

            # Check for reasonable time overlap
            if latest_truck < earliest_cargo:
                st.warning("⚠️ All trucks are available before cargo becomes available")
            elif earliest_truck > latest_cargo:
                st.warning("⚠️ All trucks are available after all cargo windows close")

            # Display compatibility summary
            st.success("✅ Basic truck-cargo compatibility confirmed")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Matching Types:**")
                for match_type in matching_types:
                    truck_count = trucks_df[trucks_df['truck type'].str.lower() == match_type].shape[0]
                    cargo_count = cargo_df[cargo_df['Cargo_Type'].str.lower() == match_type].shape[0]
                    st.write(f"- {match_type.title()}: {truck_count} trucks, {cargo_count} cargo items")

            with col2:
                st.write("**Time Ranges:**")
                st.write(f"Trucks: {earliest_truck.strftime('%Y-%m-%d')} to {latest_truck.strftime('%Y-%m-%d')}")
                st.write(f"Cargo: {earliest_cargo.strftime('%Y-%m-%d')} to {latest_cargo.strftime('%Y-%m-%d')}")

                # Calculate optimal operation period
                operation_start = max(earliest_truck, earliest_cargo - pd.Timedelta(days=1))
                operation_end = latest_cargo + pd.Timedelta(days=2)
                st.write(f"**Suggested operation period:**")
                st.write(f"{operation_start.strftime('%Y-%m-%d')} to {operation_end.strftime('%Y-%m-%d')}")

            return True

        except Exception as e:
            st.error(f"Error validating compatibility: {str(e)}")
            return False


def load_data(trucks_file, cargo_file):
    """
    Enhanced main function to load both trucks and cargo data with extended validation
    Returns tuple of (trucks_df, cargo_df) or (None, None) if validation fails
    """
    loader = DataLoader()

    trucks_df = loader.load_trucks_data(trucks_file) if trucks_file is not None else None
    cargo_df = loader.load_cargo_data(cargo_file) if cargo_file is not None else None

    if trucks_df is not None and cargo_df is not None:
        # Enhanced cross-validation
        if loader.validate_truck_cargo_compatibility(trucks_df, cargo_df):
            return trucks_df, cargo_df
        else:
            st.error("Truck-cargo compatibility validation failed")
            return None, None

    return trucks_df, cargo_df