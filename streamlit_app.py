import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_folium import folium_static
from geopy.distance import geodesic

from utils.data_loader import load_data
from utils.time_cost_calculator import optimize_assignments
from utils.visualization import create_map
from utils.time_cost_calculator import TimeCostCalculator, calculate_total_metrics
from utils.route_planner import RoutePlanner

def show_welcome_message():
    """Display welcome message and instructions"""
    st.markdown("""
    ## 👋 Welcome to the Truck-Cargo Assignment Optimizer!

    This application helps optimize the assignment of trucks to cargo loads by:
    - Minimizing total costs (distance + waiting time)
    - Respecting time windows for pickup and delivery
    - Matching truck and cargo types

    ### 📝 How to use:
    1. Upload your trucks and cargo CSV files using the sidebar
    2. Review the optimization results
    3. Explore the interactive map visualization
    4. Export the results if needed

    ### 📄 Required file formats:

    **Trucks CSV:**
    - truck_id
    - truck type
    - Address (drop off)
    - Latitude (dropoff)
    - Longitude (dropoff)
    - Timestamp (dropoff)
    - avg moving speed, km/h
    - price per km, Eur
    - waiting time price per h, EUR

    **Cargo CSV:**
    - Origin
    - Origin_Latitude
    - Origin_Longitude
    - Available_From
    - Available_To
    - Delivery_Location
    - Delivery_Latitude
    - Delivery_Longitude
    - Cargo_Type
    """)


def format_time(timestamp):
    """Format timestamp for display"""
    try:
        return pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp


def main():
    st.set_page_config(
        page_title="Truck-Cargo Assignment Optimizer",
        page_icon="🚚",
        layout="wide"
    )

    st.title("🚚 Truck and Cargo Assignment Optimizer")

    # Add sidebar with file uploaders and options
    with st.sidebar:
        st.header("📂 Upload Data Files")

        st.subheader("Trucks Data")
        trucks_file = st.file_uploader(
            "Upload CSV file with truck positions",
            type=['csv'],
            key='trucks'
        )

        st.subheader("Cargo Data")
        cargo_file = st.file_uploader(
            "Upload CSV file with cargo positions",
            type=['csv'],
            key='cargo'
        )

        st.markdown("---")

    # Load and validate data
    if trucks_file is not None and cargo_file is not None:
        try:
            trucks_df = pd.read_csv(trucks_file)
            cargo_df = pd.read_csv(cargo_file)

            # Data validation
            required_truck_columns = [
                'truck_id', 'truck type', 'Address (drop off)',
                'Latitude (dropoff)', 'Longitude (dropoff)', 'Timestamp (dropoff)',
                'price per km, Eur', 'waiting time price per h, EUR'
            ]

            required_cargo_columns = [
                'Origin', 'Origin_Latitude', 'Origin_Longitude',
                'Available_From', 'Available_To', 'Delivery_Location',
                'Delivery_Latitude', 'Delivery_Longitude', 'Cargo_Type'
            ]

            if not all(col in trucks_df.columns for col in required_truck_columns):
                st.error(f"Missing required columns in trucks file: {required_truck_columns}")
                return

            if not all(col in cargo_df.columns for col in required_cargo_columns):
                st.error(f"Missing required columns in cargo file: {required_cargo_columns}")
                return

            # Add selection filters in sidebar
            with st.sidebar:
                st.header("🎯 Filter Selection")

                # Truck selection
                st.subheader("Select Trucks")
                truck_filter_method = st.radio(
                    "Truck Filter Method",
                    ["All Trucks", "Select by ID", "Select by Type"],
                    key="truck_filter"
                )

                if truck_filter_method == "Select by ID":
                    selected_trucks = st.multiselect(
                        "Choose Trucks",
                        options=sorted(trucks_df['truck_id'].unique()),
                        default=sorted(trucks_df['truck_id'].unique())
                    )
                    trucks_df = trucks_df[trucks_df['truck_id'].isin(selected_trucks)]
                elif truck_filter_method == "Select by Type":
                    selected_truck_types = st.multiselect(
                        "Choose Truck Types",
                        options=sorted(trucks_df['truck type'].unique()),
                        default=sorted(trucks_df['truck type'].unique())
                    )
                    trucks_df = trucks_df[trucks_df['truck type'].isin(selected_truck_types)]

                # Cargo selection
                st.subheader("Select Cargo")
                cargo_filter_method = st.radio(
                    "Cargo Filter Method",
                    ["All Cargo", "Select by Type", "Select by Location"],
                    key="cargo_filter"
                )

                if cargo_filter_method == "Select by Type":
                    selected_cargo_types = st.multiselect(
                        "Choose Cargo Types",
                        options=sorted(cargo_df['Cargo_Type'].unique()),
                        default=sorted(cargo_df['Cargo_Type'].unique())
                    )
                    cargo_df = cargo_df[cargo_df['Cargo_Type'].isin(selected_cargo_types)]
                elif cargo_filter_method == "Select by Location":
                    selected_origins = st.multiselect(
                        "Choose Origin Locations",
                        options=sorted(cargo_df['Origin'].unique()),
                        default=sorted(cargo_df['Origin'].unique())
                    )
                    selected_destinations = st.multiselect(
                        "Choose Delivery Locations",
                        options=sorted(cargo_df['Delivery_Location'].unique()),
                        default=sorted(cargo_df['Delivery_Location'].unique())
                    )
                    cargo_df = cargo_df[
                        cargo_df['Origin'].isin(selected_origins) &
                        cargo_df['Delivery_Location'].isin(selected_destinations)
                        ]

                st.markdown("---")

                # Optimization settings
                st.header("⚙️ Settings")
                standard_speed = st.number_input(
                    "Standard Speed (km/h)",
                    value=73,
                    min_value=1,
                    help="Speed used for travel time calculations"
                )

                max_waiting_hours = st.number_input(
                    "Maximum Waiting Hours",
                    value=24,
                    min_value=1,
                    help="Maximum allowed waiting time at pickup"
                )

                max_distance = st.number_input(
                    "Maximum Distance (km)",
                    value=250,
                    min_value=1,
                    help="Maximum allowed distance for a single route"
                )

                show_debug = st.checkbox(
                    "Show Debug Information",
                    value=False,
                    help="Display additional debug information"
                )

            # Display filtered data preview
            with st.expander("📊 Preview Filtered Data"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Selected Trucks ({len(trucks_df)} total)")
                    st.dataframe(trucks_df)
                with col2:
                    st.subheader(f"Selected Cargo ({len(cargo_df)} total)")
                    st.dataframe(cargo_df)

            # Proceed with optimization only if there are trucks and cargo
            if len(trucks_df) == 0:
                st.error("No trucks selected! Please adjust your truck filters.")
                return

            if len(cargo_df) == 0:
                st.error("No cargo selected! Please adjust your cargo filters.")
                return

            try:
                # Initialize route planner with custom speed
                planner = RoutePlanner(standard_speed_kmh=standard_speed)
                route_chains = {}

                with st.spinner('Planning optimal routes...'):
                    # Plan routes for each selected truck
                    for idx, truck in trucks_df.iterrows():
                        # Get route chain for this truck
                        chain = planner.plan_route_chain(
                            truck,
                            cargo_df,
                            truck['Timestamp (dropoff)'],
                            pd.to_datetime(truck['Timestamp (dropoff)']) - pd.Timedelta(hours=8)
                        )

                        if chain:
                            route_chains[idx] = chain

                if route_chains:
                    # Display results using the existing display_route_results function
                    display_route_results(route_chains, trucks_df)

                    # Create and display map
                    st.header("🗺️ Route Visualization")
                    map_obj = create_map(trucks_df, cargo_df, route_chains)
                    folium_static(map_obj)

                    # Export functionality remains the same...
                else:
                    st.warning("No valid routes found with the current selection. Try adjusting your filters.")

            except Exception as e:
                st.error(f"Error during route planning: {str(e)}")
                if show_debug:
                    st.exception(e)
                st.info("Please check your input data and try again.")

        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
            st.info("Please make sure your CSV files have the correct format.")
            if show_debug:
                st.exception(e)
    else:
        show_welcome_message()

def display_route_results(route_chains, trucks_df):
    """Display route planning results in Streamlit"""
    # Summary metrics
    total_deliveries = sum(len(chain) for chain in route_chains.values())
    total_distance = sum(
        sum(route['total_distance'] for route in chain)
        for chain in route_chains.values()
    )
    total_rest_stops = sum(
        sum(len(route['rest_stops_to_pickup']) + len(route['rest_stops_to_delivery'])
            for route in chain)
        for chain in route_chains.values()
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Deliveries", total_deliveries)
    with col2:
        st.metric("Total Distance", f"{total_distance:.1f} km")
    with col3:
        st.metric("Required Rest Stops", total_rest_stops)

    # Display route chains
    st.subheader("📋 Detailed Route Plans")
    for truck_idx, chain in route_chains.items():
        truck = trucks_df.iloc[truck_idx]
        with st.expander(f"Truck {truck['truck_id']} Route Plan"):
            route_data = []
            for i, route in enumerate(chain, 1):
                rest_stops_count = len(route['rest_stops_to_pickup']) + len(route['rest_stops_to_delivery'])
                total_rest_duration = sum(
                    stop['duration'] for stop in route['rest_stops_to_pickup'] + route['rest_stops_to_delivery']
                )

                route_data.append({
                    "Stop": i,
                    "Cargo Type": route['cargo']['Cargo_Type'],
                    "Pickup Location": route['cargo']['Origin'],
                    "Delivery Location": route['cargo']['Delivery_Location'],
                    "Pickup Time": route['pickup_time'],
                    "Delivery Time": route['delivery_time'],
                    "Distance (km)": f"{route['total_distance']:.1f}",
                    "Rest Stops": rest_stops_count,
                    "Rest Duration (h)": f"{total_rest_duration:.1f}" if rest_stops_count > 0 else "-",
                    "Waiting Time (h)": f"{route['waiting_time']:.1f}"
                })
            st.dataframe(pd.DataFrame(route_data))

            # Show rest stop details if any exist
            rest_stops = []
            for i, route in enumerate(chain, 1):
                for rest in route['rest_stops_to_pickup']:
                    rest_stops.append({
                        "Route": i,
                        "Phase": "To Pickup",
                        "Time": rest['time'],
                        "Duration (h)": rest['duration'],
                        "Type": rest['type']
                    })
                for rest in route['rest_stops_to_delivery']:
                    rest_stops.append({
                        "Route": i,
                        "Phase": "To Delivery",
                        "Time": rest['time'],
                        "Duration (h)": rest['duration'],
                        "Type": rest['type']
                    })

            if rest_stops:
                st.write("Rest Stop Details:")
                st.dataframe(pd.DataFrame(rest_stops))

def display_results(trucks_df, cargo_df, assignments, time_info, rejection_info, show_debug):
    """Display optimization results with detailed information"""

    # Calculate metrics using the updated function
    metrics = calculate_total_metrics(assignments, time_info, rejection_info)

    # Display main metrics
    st.header("📊 Optimization Results")

    # Show summary metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cost", f"€{metrics['total_cost']:.2f}")
        st.metric("Distance Cost", f"€{metrics['total_distance_cost']:.2f}")
        st.metric("Waiting Cost", f"€{metrics['total_waiting_cost']:.2f}")
    with col2:
        st.metric("Total Distance", f"{metrics['total_distance']:.2f} km")
        st.metric("Average Distance", f"{metrics['average_distance']:.2f} km")
        st.metric("Total Waiting", f"{metrics['total_waiting_hours']:.2f} h")
    with col3:
        st.metric("Successful Assignments", metrics['assignments_count'])
        st.metric("Assignment Rate", f"{metrics['assignment_rate'] * 100:.1f}%")
        st.metric("Total Rejected", metrics['rejection_stats']['total_rejected'])

    # Create detailed assignments table
    st.subheader("📋 Detailed Assignments")

    assignments_data = []
    for truck_idx, cargo_idx in assignments:
        truck = trucks_df.iloc[truck_idx]
        cargo = cargo_df.iloc[cargo_idx]
        info = time_info[(truck_idx, cargo_idx)]

        assignments_data.append({
            "Truck ID": truck['truck_id'],
            "Truck Type": truck['truck type'],
            "Drop-off Location": truck['Address (drop off)'],
            "Drop-off Time": format_time(truck['Timestamp (dropoff)']),
            "Cargo Type": cargo['Cargo_Type'],
            "Cargo Origin": cargo['Origin'],
            "Cargo Destination": cargo['Delivery_Location'],
            "Available From": format_time(cargo['Available_From']),
            "Available To": format_time(cargo['Available_To']),
            "Pickup Time": format_time(info['pickup_time']),
            "Distance (km)": f"{info['distance']:.2f}",
            "Distance Cost (€)": f"{info['distance_cost']:.2f}",
            "Waiting (h)": f"{info['waiting_hours']:.2f}",
            "Waiting Cost (€)": f"{info['waiting_cost']:.2f}",
            "Total Cost (€)": f"{info['total_cost']:.2f}"
        })

    assignments_df = pd.DataFrame(assignments_data)
    st.dataframe(assignments_df)

    # Add export button for assignments
    st.download_button(
        label="📥 Download Assignments CSV",
        data=assignments_df.to_csv(index=False).encode('utf-8'),
        file_name="assignments.csv",
        mime="text/csv"
    )

    # Display map
    st.header("🗺️ Map Visualization")
    map_obj = create_map(trucks_df, cargo_df, assignments, time_info)
    folium_static(map_obj)

    # Display rejection statistics if any
    if rejection_info:
        st.header("📊 Rejection Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Rejection Reasons")
            for reason, count in metrics['rejection_stats']['by_reason'].items():
                st.write(f"- {reason}: {count}")

        with col2:
            st.subheader("Rejection Metrics")
            st.write(f"- Rejected by Distance: {metrics['rejection_stats']['rejected_by_distance']}")
            st.write(f"- Rejected by Waiting Time: {metrics['rejection_stats']['rejected_by_waiting_time']}")
            st.write(f"- Rejected by Time Window: {metrics['rejection_stats']['rejected_by_time_window']}")
            if metrics['rejection_stats']['average_rejected_distance'] > 0:
                st.write(
                    f"- Average Rejected Distance: {metrics['rejection_stats']['average_rejected_distance']:.2f} km")
            if metrics['rejection_stats']['average_rejected_waiting_time'] > 0:
                st.write(
                    f"- Average Rejected Waiting Time: {metrics['rejection_stats']['average_rejected_waiting_time']:.2f} h")

    # Display unassigned vehicles and cargo
    assigned_trucks = set(t for t, _ in assignments)
    assigned_cargo = set(c for _, c in assignments)

    if len(assigned_trucks) < len(trucks_df) or len(assigned_cargo) < len(cargo_df):
        st.header("⚠️ Unassigned Items")
        col1, col2 = st.columns(2)

        with col1:
            unassigned_trucks = trucks_df[~trucks_df.index.isin(assigned_trucks)]
            if not unassigned_trucks.empty:
                st.subheader("Unassigned Trucks")
                st.dataframe(unassigned_trucks)

        with col2:
            unassigned_cargo = cargo_df[~cargo_df.index.isin(assigned_cargo)]
            if not unassigned_cargo.empty:
                st.subheader("Unassigned Cargo")
                st.dataframe(unassigned_cargo)

    if show_debug:
        st.header("🔍 Debug Information")
        with st.expander("Show Details"):
            st.write("Time Info:", time_info)
            st.write("Assignments:", assignments)
            st.write("Rejection Info:", rejection_info)
            st.write("Metrics:", metrics)

if __name__ == "__main__":
    main()