# check_data.py
# This is a standalone script to validate input data before running the main application

import pandas as pd
import sys
import os

def check_truck_data(file_path):
    """Check if truck data is valid"""
    try:
        # Read CSV file
        trucks_df = pd.read_csv(file_path)
        
        # Check required columns
        required_columns = [
            'truck_id', 'truck type', 'Address (drop off)',
            'Latitude (dropoff)', 'Longitude (dropoff)', 'Timestamp (dropoff)',
            'price per km, Eur', 'waiting time price per h, EUR'
        ]
        
        missing_columns = [col for col in required_columns if col not in trucks_df.columns]
        if missing_columns:
            print(f"ERROR: Missing required columns in trucks file: {', '.join(missing_columns)}")
            return False
        
        # Check data types
        try:
            trucks_df['Latitude (dropoff)'] = pd.to_numeric(trucks_df['Latitude (dropoff)'])
            trucks_df['Longitude (dropoff)'] = pd.to_numeric(trucks_df['Longitude (dropoff)'])
            trucks_df['Timestamp (dropoff)'] = pd.to_datetime(trucks_df['Timestamp (dropoff)'])
            trucks_df['price per km, Eur'] = pd.to_numeric(trucks_df['price per km, Eur'])
            trucks_df['waiting time price per h, EUR'] = pd.to_numeric(trucks_df['waiting time price per h, EUR'])
        except Exception as e:
            print(f"ERROR: Invalid data types in trucks file: {str(e)}")
            return False
        
        # Check coordinates
        invalid_lat = (trucks_df['Latitude (dropoff)'] < -90) | (trucks_df['Latitude (dropoff)'] > 90)
        invalid_lon = (trucks_df['Longitude (dropoff)'] < -180) | (trucks_df['Longitude (dropoff)'] > 180)
        if invalid_lat.any() or invalid_lon.any():
            print("ERROR: Invalid coordinates in trucks file")
            if invalid_lat.any():
                print(f"Invalid latitudes in rows: {list(trucks_df[invalid_lat].index)}")
            if invalid_lon.any():
                print(f"Invalid longitudes in rows: {list(trucks_df[invalid_lon].index)}")
            return False
        
        # Check for duplicate truck IDs
        if trucks_df['truck_id'].duplicated().any():
            print(f"WARNING: Duplicate truck IDs found in rows: {list(trucks_df[trucks_df['truck_id'].duplicated()].index)}")
        
        # Check truck types
        truck_types = trucks_df['truck type'].unique()
        print(f"Truck types found: {', '.join(truck_types)}")
        
        # Display summary
        print(f"VALID: Trucks data is valid. Found {len(trucks_df)} trucks.")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to validate trucks file: {str(e)}")
        return False


def check_cargo_data(file_path):
    """Check if cargo data is valid"""
    try:
        # Read CSV file
        cargo_df = pd.read_csv(file_path)
        
        # Check required columns
        required_columns = [
            'Origin', 'Origin_Latitude', 'Origin_Longitude',
            'Available_From', 'Available_To', 'Delivery_Location',
            'Delivery_Latitude', 'Delivery_Longitude', 'Cargo_Type'
        ]
        
        missing_columns = [col for col in required_columns if col not in cargo_df.columns]
        if missing_columns:
            print(f"ERROR: Missing required columns in cargo file: {', '.join(missing_columns)}")
            return False
        
        # Check for Premium column
        if 'Premium' not in cargo_df.columns:
            print("WARNING: Premium column not found in cargo data. This will lead to unprofitable assignments.")
            print("Consider adding a Premium column with positive values.")
        else:
            # Check Premium values
            cargo_df['Premium'] = pd.to_numeric(cargo_df['Premium'], errors='coerce')
            if cargo_df['Premium'].isna().any():
                print("WARNING: Invalid Premium values found in cargo data.")
            if (cargo_df['Premium'] <= 0).any():
                print("WARNING: Zero or negative Premium values found in cargo data.")
            
            # Show Premium statistics
            print(f"Premium statistics: min={cargo_df['Premium'].min()}, max={cargo_df['Premium'].max()}, avg={cargo_df['Premium'].mean():.2f}")
        
        # Check data types
        try:
            cargo_df['Origin_Latitude'] = pd.to_numeric(cargo_df['Origin_Latitude'])
            cargo_df['Origin_Longitude'] = pd.to_numeric(cargo_df['Origin_Longitude'])
            cargo_df['Delivery_Latitude'] = pd.to_numeric(cargo_df['Delivery_Latitude'])
            cargo_df['Delivery_Longitude'] = pd.to_numeric(cargo_df['Delivery_Longitude'])
            cargo_df['Available_From'] = pd.to_datetime(cargo_df['Available_From'])
            cargo_df['Available_To'] = pd.to_datetime(cargo_df['Available_To'])
        except Exception as e:
            print(f"ERROR: Invalid data types in cargo file: {str(e)}")
            return False
        
        # Check coordinates
        invalid_o_lat = (cargo_df['Origin_Latitude'] < -90) | (cargo_df['Origin_Latitude'] > 90)
        invalid_o_lon = (cargo_df['Origin_Longitude'] < -180) | (cargo_df['Origin_Longitude'] > 180)
        invalid_d_lat = (cargo_df['Delivery_Latitude'] < -90) | (cargo_df['Delivery_Latitude'] > 90)
        invalid_d_lon = (cargo_df['Delivery_Longitude'] < -180) | (cargo_df['Delivery_Longitude'] > 180)
        
        if invalid_o_lat.any() or invalid_o_lon.any() or invalid_d_lat.any() or invalid_d_lon.any():
            print("ERROR: Invalid coordinates in cargo file")
            if invalid_o_lat.any():
                print(f"Invalid origin latitudes in rows: {list(cargo_df[invalid_o_lat].index)}")
            if invalid_o_lon.any():
                print(f"Invalid origin longitudes in rows: {list(cargo_df[invalid_o_lon].index)}")
            if invalid_d_lat.any():
                print(f"Invalid delivery latitudes in rows: {list(cargo_df[invalid_d_lat].index)}")
            if invalid_d_lon.any():
                print(f"Invalid delivery longitudes in rows: {list(cargo_df[invalid_d_lon].index)}")
            return False
        
        # Check time windows
        invalid_window = cargo_df['Available_From'] >= cargo_df['Available_To']
        if invalid_window.any():
            print(f"ERROR: Invalid time windows (From >= To) in rows: {list(cargo_df[invalid_window].index)}")
            return False
        
        # Check cargo types
        cargo_types = cargo_df['Cargo_Type'].unique()
        print(f"Cargo types found: {', '.join(cargo_types)}")
        
        # Display summary
        print(f"VALID: Cargo data is valid. Found {len(cargo_df)} cargo items.")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to validate cargo file: {str(e)}")
        return False


def check_compatibility(trucks_file, cargo_file):
    """Check compatibility between trucks and cargo data"""
    try:
        # Read files
        trucks_df = pd.read_csv(trucks_file)
        cargo_df = pd.read_csv(cargo_file)
        
        # Check type compatibility
        truck_types = set(trucks_df['truck type'].str.lower())
        cargo_types = set(cargo_df['Cargo_Type'].str.lower())
        
        matching_types = truck_types.intersection(cargo_types)
        
        print(f"Truck types: {', '.join(truck_types)}")
        print(f"Cargo types: {', '.join(cargo_types)}")
        print(f"Matching types: {', '.join(matching_types)}")
        
        if not matching_types:
            print("ERROR: No matching truck and cargo types found!")
            return False
        
        # Check unmatched types
        unmatched_cargo = cargo_types - matching_types
        if unmatched_cargo:
            print(f"WARNING: Some cargo types have no matching trucks: {', '.join(unmatched_cargo)}")
        
        unmatched_trucks = truck_types - cargo_types
        if unmatched_trucks:
            print(f"INFO: Some truck types have no matching cargo: {', '.join(unmatched_trucks)}")
        
        # Check time compatibility
        cargo_df['Available_From'] = pd.to_datetime(cargo_df['Available_From'])
        cargo_df['Available_To'] = pd.to_datetime(cargo_df['Available_To'])
        trucks_df['Timestamp (dropoff)'] = pd.to_datetime(trucks_df['Timestamp (dropoff)'])
        
        earliest_truck = trucks_df['Timestamp (dropoff)'].min()
        latest_cargo = cargo_df['Available_To'].max()
        
        print(f"Earliest truck availability: {earliest_truck}")
        print(f"Latest cargo availability: {latest_cargo}")
        
        if earliest_truck > latest_cargo:
            print("WARNING: All trucks are available after all cargo availability windows end!")
            return False
        
        # Count trucks and cargo by type
        truck_counts = trucks_df['truck type'].str.lower().value_counts().to_dict()
        cargo_counts = cargo_df['Cargo_Type'].str.lower().value_counts().to_dict()
        
        print("Truck counts by type:")
        for t_type, count in truck_counts.items():
            print(f"  {t_type}: {count}")
        
        print("Cargo counts by type:")
        for c_type, count in cargo_counts.items():
            print(f"  {c_type}: {count}")
        
        # Display summary
        print("VALID: Trucks and cargo data are compatible.")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to check compatibility: {str(e)}")
        return False


def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python check_data.py <trucks_file> <cargo_file>")
        return
    
    trucks_file = sys.argv[1]
    cargo_file = sys.argv[2]
    
    print(f"Checking trucks file: {trucks_file}")
    trucks_valid = check_truck_data(trucks_file)
    
    print("\n" + "="*80 + "\n")
    
    print(f"Checking cargo file: {cargo_file}")
    cargo_valid = check_cargo_data(cargo_file)
    
    print("\n" + "="*80 + "\n")
    
    if trucks_valid and cargo_valid:
        print("Checking compatibility between trucks and cargo")
        check_compatibility(trucks_file, cargo_file)
    
    print("\n" + "="*80 + "\n")
    
    if trucks_valid and cargo_valid:
        print("Summary: Both data files are valid. You can proceed with the optimization.")
    else:
        issues = []
        if not trucks_valid:
            issues.append("trucks file")
        if not cargo_valid:
            issues.append("cargo file")
        
        print(f"Summary: There are issues with the {' and '.join(issues)}. Please fix them before running the optimization.")


if __name__ == "__main__":
    main()