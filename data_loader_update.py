# Modified functions for utils/data_loader.py

def load_cargo_data(self, file):
    """
    Load and validate cargo data from CSV file

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
    - Premium: revenue for delivering cargo (optional, default=0)
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

        # Add Premium column with default value if missing
        if 'Premium' not in df.columns:
            st.warning("Premium column not found in cargo data. Using default value of 0.")
            df['Premium'] = 0
        else:
            # Validate premium values (ensure they are numeric and positive)
            if not pd.to_numeric(df['Premium'], errors='coerce').notna().all():
                st.warning("Invalid Premium values found in cargo data. Replacing with 0.")
                df['Premium'] = pd.to_numeric(df['Premium'], errors='coerce').fillna(0)
            
            # Ensure all values are positive
            if (df['Premium'] < 0).any():
                st.warning("Negative Premium values found in cargo data. Taking absolute values.")
                df['Premium'] = df['Premium'].abs()

        # Validate data
        validation_errors = []

        # Validate origin coordinates
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

        # Validate delivery coordinates
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

        # Validate timestamps
        invalid_from_times = df.apply(
            lambda row: not self.validator.validate_timestamp(row['Available_From']),
            axis=1
        )
        if invalid_from_times.any():
            validation_errors.append(
                f"Invalid Available_From timestamps found in rows: {list(df[invalid_from_times].index)}"
            )

        invalid_to_times = df.apply(
            lambda row: not self.validator.validate_timestamp(row['Available_To']),
            axis=1
        )
        if invalid_to_times.any():
            validation_errors.append(
                f"Invalid Available_To timestamps found in rows: {list(df[invalid_to_times].index)}"
            )

        # Validate time windows
        df['Available_From'] = pd.to_datetime(df['Available_From'])
        df['Available_To'] = pd.to_datetime(df['Available_To'])
        invalid_windows = df['Available_From'] >= df['Available_To']
        if invalid_windows.any():
            validation_errors.append(
                f"Invalid time windows (From >= To) found in rows: {list(df[invalid_windows].index)}"
            )

        # Report validation errors
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            return None

        # Clean and convert data types
        df['Origin_Latitude'] = pd.to_numeric(df['Origin_Latitude'])
        df['Origin_Longitude'] = pd.to_numeric(df['Origin_Longitude'])
        df['Delivery_Latitude'] = pd.to_numeric(df['Delivery_Latitude'])
        df['Delivery_Longitude'] = pd.to_numeric(df['Delivery_Longitude'])
        df['Premium'] = pd.to_numeric(df['Premium'])

        # Add validation success message
        st.success(f"Successfully loaded {len(df)} cargo items")

        return df

    except Exception as e:
        st.error(f"Error loading cargo file: {str(e)}")
        return None