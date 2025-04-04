import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_folium import folium_static
from geopy.distance import geodesic
import os

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
    - Ensuring maximum distance constraints are met

    ### 📝 How to use:
    1. Upload your trucks and cargo CSV files using the sidebar
    2. Or click "Use Sample Data" to try the app with example data
    3. Set parameters like maximum distance and waiting time
    4. Review the optimization results
    5. Explore the interactive map visualization
    6. Filter trucks and routes using the map controls

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


def load_sample_data():
    """Load sample data from files included in the project"""
    try:
        # Get the current directory where streamlit_app.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the sample CSV files
        trucks_df = pd.read_csv(os.path.join(current_dir, 'trucks.csv'))
        cargo_df = pd.read_csv(os.path.join(current_dir, 'cargos.csv'))

        return trucks_df, cargo_df
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None, None


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

        # Sample data button
        use_sample_data = st.button("📊 Use Sample Data", help="Load example truck and cargo data")

        st.subheader("Or upload your own data:")

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

    # Initialize data variables
    trucks_df = None
    cargo_df = None

    # Load sample data if button is clicked
    if use_sample_data:
        with st.spinner('Loading sample data...'):
            trucks_df, cargo_df = load_sample_data()
            if trucks_df is not None and cargo_df is not None:
                st.success("Sample data loaded successfully!")
    # Load user-uploaded data if available
    elif trucks_file is not None and cargo_file is not None:
        try:
            trucks_df = pd.read_csv(trucks_file)
            cargo_df = pd.read_csv(cargo_file)
        except Exception as e:
            st.error(f"Error loading uploaded files: {str(e)}")
            st.info("Please make sure your CSV files have the correct format.")

    # Proceed if we have data (either sample or uploaded)
    if trucks_df is not None and cargo_df is not None:
        try:
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

                # Add standard_speed with more informative help text
                standard_speed = st.number_input(
                    "Standard Speed (km/h)",
                    value=73,
                    min_value=1,
                    help="Speed used for travel time calculations"
                )

                # Add max_waiting_hours with improved description
                max_waiting_hours = st.number_input(
                    "Maximum Waiting Hours",
                    value=24,
                    min_value=1,
                    help="Maximum allowed waiting time at pickup location"
                )

                # Add max_distance with more prominent display
                max_distance = st.number_input(
                    "Maximum Distance (km)",
                    value=250,
                    min_value=1,
                    help="Maximum allowed distance between truck position and pickup location"
                )

                # Add this notice about the maximum distance parameter
                st.info("Maximum distance setting limits how far a truck can travel to pickup cargo.")

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
                # Set constraints manually
                planner.max_distance_km = max_distance
                planner.max_waiting_hours = max_waiting_hours
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
                    st.info(
                        "Use the layer controls to show/hide different elements on the map. Unassigned cargo is shown in gray.")

                    # Create the map with all cargo (assigned and unassigned)
                    map_obj = create_map(trucks_df, cargo_df, route_chains)

                    # Display the map with folium_static
                    folium_static(map_obj, width=1000, height=600)

                    # Export functionality
                    st.subheader("Export Results")
                    export_button = st.download_button(
                        label="📥 Download Route Summary (CSV)",
                        data=generate_route_summary_csv(route_chains, trucks_df),
                        file_name="route_summary.csv",
                        mime="text/csv"
                    )

                else:
                    st.warning(
                        "No valid routes found with the current selection. Try adjusting your filters or increasing the maximum distance setting.")

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


def generate_route_summary_csv(route_chains, trucks_df):
    """Generate a CSV summary of all routes for export"""
    data = []
    for truck_idx, chain in route_chains.items():
        truck = trucks_df.iloc[truck_idx]

        for i, route in enumerate(chain, 1):
            cargo = route['cargo']

            data.append({
                "Truck ID": truck['truck_id'],
                "Truck Type": truck['truck type'],
                "Stop Number": i,
                "Cargo Type": cargo['Cargo_Type'],
                "Origin": cargo['Origin'],
                "Destination": cargo['Delivery_Location'],
                "Pickup Time": route['pickup_time'],
                "Delivery Time": route['delivery_time'],
                "Distance to Pickup (km)": f"{route['distance_to_pickup']:.1f}",
                "Distance to Delivery (km)": f"{route['distance_to_delivery']:.1f}",
                "Total Distance (km)": f"{route['total_distance']:.1f}",
                "Waiting Time (h)": f"{route['waiting_time']:.1f}",
                "Rest Stops": len(route['rest_stops_to_pickup']) + len(route['rest_stops_to_delivery'])
            })

    return pd.DataFrame(data).to_csv(index=False).encode('utf-8')


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

    # Calculate total waiting time
    total_waiting_hours = sum(
        sum(route['waiting_time'] for route in chain)
        for chain in route_chains.values()
    )

    # Display in a nice grid of metrics
    st.header("📊 Optimization Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Deliveries", total_deliveries)
    with col2:
        st.metric("Total Distance", f"{total_distance:.1f} km")
    with col3:
        st.metric("Total Waiting Time", f"{total_waiting_hours:.1f} h")
    with col4:
        st.metric("Required Rest Stops", total_rest_stops)

    # Display route chains
    st.subheader("📋 Detailed Route Plans")

    # Display truck routes in tabs
    if len(route_chains) > 1:
        tabs = st.tabs([f"Truck {trucks_df.iloc[idx]['truck_id']}" for idx in route_chains.keys()])

        for i, (truck_idx, chain) in enumerate(route_chains.items()):
            with tabs[i]:
                display_truck_route(truck_idx, chain, trucks_df)
    else:
        # If only one truck, don't use tabs
        for truck_idx, chain in route_chains.items():
            display_truck_route(truck_idx, chain, trucks_df)


def display_truck_route(truck_idx, chain, trucks_df):
    """Display detailed route for a single truck"""
    truck = trucks_df.iloc[truck_idx]

    # Show truck info
    st.write(
        f"**Type:** {truck['truck type']} • **Speed:** {truck['avg moving speed, km/h']} km/h • **Price/km:** €{truck['price per km, Eur']} • **Waiting price/h:** €{truck['waiting time price per h, EUR']}")

    # Starting position info
    st.write("**Starting Position:**")
    st.write(
        f"Location: {truck['Address (drop off)']} • Coordinates: ({truck['Latitude (dropoff)']:.4f}, {truck['Longitude (dropoff)']:.4f}) • Available from: {pd.to_datetime(truck['Timestamp (dropoff)']).strftime('%Y-%m-%d %H:%M')}")

    # Format route data
    route_data = []

    for i, route in enumerate(chain, 1):
        rest_stops_count = len(route['rest_stops_to_pickup']) + len(route['rest_stops_to_delivery'])
        total_rest_duration = sum(
            stop['duration'] for stop in route['rest_stops_to_pickup'] + route['rest_stops_to_delivery']
        )

        # Add pickup route information
        pickup_info = f"From: {truck['Address (drop off)'] if i == 1 else chain[i - 2]['cargo']['Delivery_Location']}"
        pickup_info += f" • To: {route['cargo']['Origin']}"
        pickup_info += f" • Distance: {route['distance_to_pickup']:.1f} km"
        pickup_info += f" • Rest stops: {len(route['rest_stops_to_pickup'])}"

        # Add delivery route information
        delivery_info = f"From: {route['cargo']['Origin']}"
        delivery_info += f" • To: {route['cargo']['Delivery_Location']}"
        delivery_info += f" • Distance: {route['distance_to_delivery']:.1f} km"
        delivery_info += f" • Rest stops: {len(route['rest_stops_to_delivery'])}"

        route_data.append({
            "Stop": i,
            "Cargo Type": route['cargo']['Cargo_Type'],
            "Pickup Route": pickup_info,
            "Delivery Route": delivery_info,
            "Pickup Location": route['cargo']['Origin'],
            "Delivery Location": route['cargo']['Delivery_Location'],
            "Pickup Time": route['pickup_time'].strftime('%Y-%m-%d %H:%M'),
            "Delivery Time": route['delivery_time'].strftime('%Y-%m-%d %H:%M'),
            "Total Distance (km)": f"{route['total_distance']:.1f}",
            "Rest Stops": rest_stops_count,
            "Rest Duration (h)": f"{total_rest_duration:.1f}" if rest_stops_count > 0 else "-",
            "Waiting Time (h)": f"{route['waiting_time']:.1f}"
        })

    # Create a dataframe for display
    route_df = pd.DataFrame(route_data)

    # Simplified table for the main view
    simplified_view = route_df[['Stop', 'Cargo Type', 'Pickup Location', 'Delivery Location',
                                'Pickup Time', 'Delivery Time', 'Total Distance (km)', 'Waiting Time (h)']]
    st.dataframe(simplified_view, use_container_width=True)

    # Detailed route information in an expander
    with st.expander("View Detailed Route Information"):
        for i, route in enumerate(route_data, 1):
            st.subheader(f"Stop {i}: {route['Cargo Type']}")
            st.write("**Pickup Route:**")
            st.write(route['Pickup Route'])
            st.write("**Delivery Route:**")
            st.write(route['Delivery Route'])
            st.write(f"**Pickup Time:** {route['Pickup Time']} • **Delivery Time:** {route['Delivery Time']}")
            st.write(
                f"**Total Distance:** {route['Total Distance (km)']} km • **Waiting Time:** {route['Waiting Time (h)']} h • **Rest Stops:** {route['Rest Stops']}")
            st.markdown("---")

    # Show rest stop details if any exist
    rest_stops = []
    for i, route in enumerate(chain, 1):
        for rest in route['rest_stops_to_pickup']:
            rest_stops.append({
                "Route": i,
                "Phase": "To Pickup",
                "Time": rest['time'].strftime('%Y-%m-%d %H:%M'),
                "Duration (h)": rest['duration'],
                "Type": rest['type']
            })
        for rest in route['rest_stops_to_delivery']:
            rest_stops.append({
                "Route": i,
                "Phase": "To Delivery",
                "Time": rest['time'].strftime('%Y-%m-%d %H:%M'),
                "Duration (h)": rest['duration'],
                "Type": rest['type']
            })

    if rest_stops:
        st.write("**Rest Stop Details:**")
        st.dataframe(pd.DataFrame(rest_stops), use_container_width=True)


if __name__ == "__main__":
    main()