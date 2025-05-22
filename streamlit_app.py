import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
from streamlit_folium import folium_static
from geopy.distance import geodesic
import os
import sys
import io

from utils.data_loader import load_data
from utils.time_cost_calculator import TimeCostCalculator, calculate_total_metrics
from utils.profit_optimizer import FleetProfitOptimizer, ProfitCalculator
from utils.visualization import create_map
from utils.route_planner import RoutePlanner
from utils.diagnostics_module import run_comprehensive_diagnostics, generate_diagnostic_report


def show_welcome_message():
    """Display welcome message and instructions"""
    st.markdown("""
    ## 👋 Welcome to the Truck-Cargo Assignment Profit Optimizer!

    This application helps optimize the assignment of trucks to cargo loads by:
    - Maximizing total profit across the fleet
    - Respecting time windows for pickup and delivery
    - Matching truck and cargo types
    - Ensuring maximum distance and waiting time constraints are met
    - Optimizing fleet utilization across the operating period

    ### 📝 How to use:
    1. Upload your trucks and cargo CSV files using the sidebar
    2. Or click "Use Sample Data" to try the app with example data
    3. Set parameters like maximum distance, waiting time, prices, and operating period
    4. Click "Maximize Fleet Profit" to run the optimization
    5. Review the profit results and route assignments
    6. Explore the interactive map visualization
    7. Filter trucks and routes using the map controls

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
    - Premium (profit for delivering cargo)
    """)


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
                "Rest Stops": len(route['rest_stops_to_pickup']) + len(route['rest_stops_to_delivery']),
                "Premium (EUR)": f"{cargo['Premium']:.2f}",
                "Profit (EUR)": f"{route['profit']:.2f}"
            })

    return pd.DataFrame(data).to_csv(index=False).encode('utf-8')


def display_route_results(route_chains, trucks_df):
    """Display route planning results with profit information in Streamlit"""
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

    # Calculate total profit
    total_profit = sum(
        sum(route['profit'] for route in chain)
        for chain in route_chains.values()
    )

    # Calculate total premium
    total_premium = sum(
        sum(route['cargo']['Premium'] for route in chain)
        for chain in route_chains.values()
    )

    # Display in a nice grid of metrics
    st.header("📊 Optimization Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Deliveries", total_deliveries)
        st.metric("Total Distance", f"{total_distance:.1f} km")
    with col2:
        st.metric("Total Waiting Time", f"{total_waiting_hours:.1f} h")
        st.metric("Required Rest Stops", total_rest_stops)
    with col3:
        st.metric("Total Premium", f"{total_premium:.2f} EUR")
        st.metric("Total Profit", f"{total_profit:.2f} EUR")

    # Display profit by truck
    st.subheader("Profit by Truck")

    profit_by_truck = {}
    for truck_idx, chain in route_chains.items():
        truck_id = trucks_df.iloc[truck_idx]['truck_id']
        profit = sum(route['profit'] for route in chain)
        profit_by_truck[truck_id] = profit

    # Create a simple bar chart of profit by truck
    if profit_by_truck:
        fig, ax = plt.subplots(figsize=(10, 4))
        trucks = list(profit_by_truck.keys())
        profits = list(profit_by_truck.values())

        bars = ax.bar(trucks, profits)
        ax.set_xlabel('Truck ID')
        ax.set_ylabel('Profit (EUR)')
        ax.set_title('Profit by Truck')

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        st.pyplot(fig)

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
    """Display detailed route for a single truck including profit information"""
    truck = trucks_df.iloc[truck_idx]

    # Calculate total profit for this truck
    total_profit = sum(route['profit'] for route in chain)
    total_premium = sum(route['cargo']['Premium'] for route in chain)
    total_distance = sum(route['total_distance'] for route in chain)

    # Show truck info with profit metrics
    st.write(
        f"**Type:** {truck['truck type']} • **Speed:** {truck['avg moving speed, km/h']} km/h • "
        f"**Price/km:** €{truck['price per km, Eur']} • **Waiting price/h:** €{truck['waiting time price per h, EUR']}"
    )

    # Show profit summary
    st.write(
        f"**Total Profit:** €{total_profit:.2f} • **Total Premium:** €{total_premium:.2f} • "
        f"**Total Distance:** {total_distance:.1f} km"
    )

    # Starting position info
    st.write("**Starting Position:**")
    st.write(
        f"Location: {truck['Address (drop off)']} • Coordinates: ({truck['Latitude (dropoff)']:.4f}, {truck['Longitude (dropoff)']:.4f}) • "
        f"Available from: {pd.to_datetime(truck['Timestamp (dropoff)']).strftime('%Y-%m-%d %H:%M')}"
    )

    # Create a unified task table with all operations in order
    unified_tasks = []

    # Set starting position for the truck
    current_position = truck['Address (drop off)']
    current_coordinates = (truck['Latitude (dropoff)'], truck['Longitude (dropoff)'])
    current_time = pd.to_datetime(truck['Timestamp (dropoff)'])

    # Process each route in the chain
    for i, route in enumerate(chain, 1):
        cargo = route['cargo']

        # 1. Add pickup route as a task
        from_location = current_position
        to_location = cargo['Origin']

        pickup_task = {
            "Task Number": len(unified_tasks) + 1,
            "Stop Number": i,
            "Task Type": "Pickup Route",
            "From Location": from_location,
            "To Location": to_location,
            "Cargo Type": cargo['Cargo_Type'],
            "Distance (km)": f"{route['distance_to_pickup']:.1f}",
            "Start Time": current_time.strftime('%Y-%m-%d %H:%M'),
            "End Time": route['pickup_time'].strftime('%Y-%m-%d %H:%M') if not route['rest_stops_to_pickup'] else None,
            "Duration (h)": f"{route['distance_to_pickup'] / st.session_state.standard_speed:.1f}",
            "Waiting Time (h)": f"{route['waiting_time']:.1f}" if route['waiting_time'] > 0 else "0.0",
            "Premium": f"{cargo['Premium']:.2f}",
            "Profit": f"{route['profit']:.2f}"
        }
        unified_tasks.append(pickup_task)

        # 2. Add rest stops during pickup route
        for rest_idx, rest in enumerate(route['rest_stops_to_pickup']):
            rest_task = {
                "Task Number": len(unified_tasks) + 1,
                "Stop Number": i,
                "Task Type": "Rest Stop (To Pickup)",
                "From Location": "En Route",
                "To Location": "En Route",
                "Cargo Type": cargo['Cargo_Type'],
                "Distance (km)": "0.0",
                "Start Time": rest['time'].strftime('%Y-%m-%d %H:%M'),
                "End Time": (rest['time'] + pd.Timedelta(hours=rest['duration'])).strftime('%Y-%m-%d %H:%M'),
                "Duration (h)": f"{rest['duration']:.1f}",
                "Waiting Time (h)": "0.0",
                "Premium": "",
                "Profit": ""
            }
            unified_tasks.append(rest_task)

        # 3. Add delivery route as a task
        from_location = cargo['Origin']
        to_location = cargo['Delivery_Location']

        delivery_task = {
            "Task Number": len(unified_tasks) + 1,
            "Stop Number": i,
            "Task Type": "Delivery Route",
            "From Location": from_location,
            "To Location": to_location,
            "Cargo Type": cargo['Cargo_Type'],
            "Distance (km)": f"{route['distance_to_delivery']:.1f}",
            "Start Time": route['pickup_time'].strftime('%Y-%m-%d %H:%M'),
            "End Time": route['delivery_time'].strftime('%Y-%m-%d %H:%M') if not route[
                'rest_stops_to_delivery'] else None,
            "Duration (h)": f"{route['distance_to_delivery'] / st.session_state.standard_speed:.1f}",
            "Waiting Time (h)": "0.0",
            "Premium": f"{cargo['Premium']:.2f}",
            "Profit": f"{route['profit']:.2f}"
        }
        unified_tasks.append(delivery_task)

        # 4. Add rest stops during delivery route
        for rest_idx, rest in enumerate(route['rest_stops_to_delivery']):
            rest_task = {
                "Task Number": len(unified_tasks) + 1,
                "Stop Number": i,
                "Task Type": "Rest Stop (To Delivery)",
                "From Location": "En Route",
                "To Location": "En Route",
                "Cargo Type": cargo['Cargo_Type'],
                "Distance (km)": "0.0",
                "Start Time": rest['time'].strftime('%Y-%m-%d %H:%M'),
                "End Time": (rest['time'] + pd.Timedelta(hours=rest['duration'])).strftime('%Y-%m-%d %H:%M'),
                "Duration (h)": f"{rest['duration']:.1f}",
                "Waiting Time (h)": "0.0",
                "Premium": "",
                "Profit": ""
            }
            unified_tasks.append(rest_task)

        # Update current position for next route
        current_position = cargo['Delivery_Location']
        current_coordinates = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])
        current_time = route['delivery_time']

    # Create unified dataframe and display
    unified_df = pd.DataFrame(unified_tasks)
    st.write("### Unified Task Table")
    st.dataframe(unified_df, use_container_width=True)

    # Add profit breakdown
    st.write("### Profit Breakdown")
    profit_breakdown = []

    for i, route in enumerate(chain, 1):
        cargo = route['cargo']
        breakdown = route.get('profit_breakdown', {})

        profit_breakdown.append({
            "Stop": i,
            "Cargo Type": cargo['Cargo_Type'],
            "Premium": f"{cargo['Premium']:.2f}",
            "Distance Cost": f"{breakdown.get('distance_cost', 0):.2f}",
            "Time Value": f"{breakdown.get('time_cost', 0):.2f}",
            "Total Cost": f"{breakdown.get('total_cost', 0):.2f}",
            "Profit": f"{route['profit']:.2f}"
        })

    profit_df = pd.DataFrame(profit_breakdown)
    st.dataframe(profit_df, use_container_width=True)

    # Add a simple profit vs. cost chart
    st.write("### Profit vs. Cost Analysis")

    try:
        # Create data for the chart
        fig, ax = plt.subplots(figsize=(10, 5))
        stops = list(range(1, len(chain) + 1))
        premiums = [route['cargo']['Premium'] for route in chain]
        costs = [route.get('profit_breakdown', {}).get('total_cost', 0) for route in chain]
        profits = [route['profit'] for route in chain]

        x = np.arange(len(stops))  # the locations for the groups
        width = 0.25  # the width of the bars

        ax.bar(x - width, premiums, width, label='Premium')
        ax.bar(x, costs, width, label='Cost')
        ax.bar(x + width, profits, width, label='Profit')

        ax.set_xticks(x)
        ax.set_xticklabels([f'Stop {i}' for i in stops])
        ax.set_ylabel('EUR')
        ax.set_title(f'Profit Analysis for Truck {truck["truck_id"]}')
        ax.legend()

        st.pyplot(fig)
    except Exception as e:
        if st.session_state.show_debug:
            st.error(f"Error creating profit chart: {str(e)}")
        else:
            st.warning("Could not create profit chart. Enable debug mode for details.")

    # Preserve the original route information in an expander for reference
    with st.expander("View Original Route Summary"):
        # Format route data for original view
        route_data = []

        for i, route in enumerate(chain, 1):
            rest_stops_count = len(route['rest_stops_to_pickup']) + len(route['rest_stops_to_delivery'])
            total_rest_duration = sum(
                stop['duration'] for stop in route['rest_stops_to_pickup'] + route['rest_stops_to_delivery']
            ) if rest_stops_count > 0 else 0

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
                "Waiting Time (h)": f"{route['waiting_time']:.1f}",
                "Premium": f"{cargo['Premium']:.2f}",
                "Profit": f"{route['profit']:.2f}"
            })

        # Create a dataframe for display
        route_df = pd.DataFrame(route_data)

        # Simplified table for the main view
        simplified_view = route_df[['Stop', 'Cargo Type', 'Pickup Location', 'Delivery Location',
                                    'Pickup Time', 'Delivery Time', 'Total Distance (km)',
                                    'Waiting Time (h)', 'Premium', 'Profit']]
        st.dataframe(simplified_view, use_container_width=True)


def load_sample_data():
    """Load sample data from files included in the project"""
    try:
        # Get the current directory where streamlit_app.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the sample CSV files
        trucks_df = pd.read_csv(os.path.join(current_dir, 'trucks.csv'))
        cargo_df = pd.read_csv(os.path.join(current_dir, 'cargos.csv'))

        # Add Premium column if not present in sample data
        if 'Premium' not in cargo_df.columns:
            # Generate random premiums based on distance
            if 'Distance' in cargo_df.columns:
                # Use existing Distance column if available
                cargo_df['Premium'] = cargo_df['Distance'] * np.random.uniform(1.5, 2.5, len(cargo_df))
            else:
                # Calculate distance between origin and delivery
                distances = []
                for _, cargo in cargo_df.iterrows():
                    origin = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])
                    delivery = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])
                    distance = geodesic(origin, delivery).kilometers
                    distances.append(distance)

                # Generate premiums based on distances
                cargo_df['Premium'] = [dist * np.random.uniform(1.5, 2.5) for dist in distances]

        # Round premium values for cleaner display
        cargo_df['Premium'] = cargo_df['Premium'].round(2)

        return trucks_df, cargo_df
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None, None


def main():
    st.set_page_config(
        page_title="Truck-Cargo Profit Maximizer",
        page_icon="🚚",
        layout="wide"
    )

    # Initialize session state for persistent settings
    if 'standard_speed' not in st.session_state:
        st.session_state.standard_speed = 73
    if 'max_waiting_hours' not in st.session_state:
        st.session_state.max_waiting_hours = 24
    if 'max_distance' not in st.session_state:
        st.session_state.max_distance = 250
    if 'price_per_km' not in st.session_state:
        st.session_state.price_per_km = 0.39
    if 'price_per_hour' not in st.session_state:
        st.session_state.price_per_hour = 10.0
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'trucks_df' not in st.session_state:
        st.session_state.trucks_df = None
    if 'cargo_df' not in st.session_state:
        st.session_state.cargo_df = None
    if 'route_chains' not in st.session_state:
        st.session_state.route_chains = {}
    if 'total_profit' not in st.session_state:
        st.session_state.total_profit = 0
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    if 'selected_trucks_for_map' not in st.session_state:
        st.session_state.selected_trucks_for_map = []
    if 'operation_end_time' not in st.session_state:
        # Default end time is 7 days from now at 18:00
        st.session_state.operation_end_time = (datetime.now() + timedelta(days=7)).replace(
            hour=18, minute=0, second=0, microsecond=0)

    st.title("🚚 Truck-Cargo Fleet Profit Maximizer")

    # Create a debug logger that captures print statements
    class StreamlitLogger:
        def __init__(self):
            self.logs = []
            self.original_stdout = sys.stdout

        def write(self, text):
            self.logs.append(text)
            self.original_stdout.write(text)

        def flush(self):
            self.original_stdout.flush()

        def get_logs(self):
            return ''.join(self.logs)

    # Create logger
    logger = StreamlitLogger()
    # Redirect stdout to logger
    sys.stdout = logger

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

        # Optimization settings
        st.header("⚙️ Settings")

        # Distance and timing settings
        st.subheader("Distance & Timing")
        st.session_state.standard_speed = st.number_input(
            "Standard Speed (km/h)",
            value=st.session_state.standard_speed,
            min_value=1,
            help="Speed used for travel time calculations"
        )

        st.session_state.max_waiting_hours = st.number_input(
            "Maximum Waiting Hours",
            value=st.session_state.max_waiting_hours,
            min_value=1,
            help="Maximum allowed waiting time at pickup location"
        )

        st.session_state.max_distance = st.number_input(
            "Maximum Distance (km)",
            value=st.session_state.max_distance,
            min_value=1,
            help="Maximum allowed distance between truck position and pickup location"
        )

        # Cost settings
        st.subheader("Cost & Profit")
        st.session_state.price_per_km = st.number_input(
            "Price per km (EUR)",
            value=st.session_state.price_per_km,
            min_value=0.01,
            step=0.01,
            format="%.2f",
            help="Price per kilometer for distance cost calculation"
        )

        st.session_state.price_per_hour = st.number_input(
            "Price per hour (EUR)",
            value=st.session_state.price_per_hour,
            min_value=0.1,
            step=0.1,
            format="%.1f",
            help="Price per hour for time cost calculation (subtracted from total cost)"
        )

        # Operating period settings
        st.subheader("Operation Period")
        operation_end_date = st.date_input(
            "Operation End Date",
            value=st.session_state.operation_end_time.date(),
            help="Global end date for all operations"
        )

        end_time_hours = st.slider(
            "End Time Hour",
            min_value=0,
            max_value=23,
            value=st.session_state.operation_end_time.hour,
            help="Hour of the day for operation end time"
        )

        end_time_minutes = st.slider(
            "End Time Minute",
            min_value=0,
            max_value=59,
            value=st.session_state.operation_end_time.minute,
            help="Minute of the hour for operation end time",
            step=5
        )

        # Create datetime object for operation end time
        st.session_state.operation_end_time = datetime.combine(
            operation_end_date,
            time(end_time_hours, end_time_minutes)
        )

        st.write(f"Operations will end at: {st.session_state.operation_end_time.strftime('%Y-%m-%d %H:%M')}")

        # Debug option
        st.session_state.show_debug = st.checkbox(
            "Show Debug Information",
            value=st.session_state.show_debug,
            help="Display additional debug information"
        )

    # Load sample data if button is clicked
    if use_sample_data:
        with st.spinner('Loading sample data...'):
            trucks_df, cargo_df = load_sample_data()
            if trucks_df is not None and cargo_df is not None:
                st.session_state.trucks_df = trucks_df
                st.session_state.cargo_df = cargo_df
                st.session_state.data_loaded = True
                st.success("Sample data loaded successfully!")
    # Load user-uploaded data if available
    elif trucks_file is not None and cargo_file is not None:
        try:
            trucks_df = pd.read_csv(trucks_file)
            cargo_df = pd.read_csv(cargo_file)
            st.session_state.trucks_df = trucks_df
            st.session_state.cargo_df = cargo_df
            st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Error loading uploaded files: {str(e)}")
            st.info("Please make sure your CSV files have the correct format.")
    else:
        # Use data from session state if already loaded
        trucks_df = st.session_state.trucks_df
        cargo_df = st.session_state.cargo_df

    # Proceed if we have data (either sample or uploaded)
    if st.session_state.data_loaded and trucks_df is not None and cargo_df is not None:
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

            # Add Premium column with default value if missing
            if 'Premium' not in cargo_df.columns:
                st.warning("Premium column not found in cargo data. Generating random values.")
                # Calculate distance between origin and delivery
                distances = []
                for _, cargo in cargo_df.iterrows():
                    origin = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])
                    delivery = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])
                    distance = geodesic(origin, delivery).kilometers
                    distances.append(distance)

                # Generate premiums based on distances
                cargo_df['Premium'] = [dist * np.random.uniform(1.5, 2.5) for dist in distances]
                cargo_df['Premium'] = cargo_df['Premium'].round(2)

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

                filtered_trucks_df = trucks_df.copy()

                if truck_filter_method == "Select by ID":
                    selected_trucks = st.multiselect(
                        "Choose Trucks",
                        options=sorted(trucks_df['truck_id'].unique()),
                        default=sorted(trucks_df['truck_id'].unique())
                    )
                    filtered_trucks_df = trucks_df[trucks_df['truck_id'].isin(selected_trucks)]
                elif truck_filter_method == "Select by Type":
                    selected_truck_types = st.multiselect(
                        "Choose Truck Types",
                        options=sorted(trucks_df['truck type'].unique()),
                        default=sorted(trucks_df['truck type'].unique())
                    )
                    filtered_trucks_df = trucks_df[trucks_df['truck type'].isin(selected_truck_types)]

                # Cargo selection
                st.subheader("Select Cargo")
                cargo_filter_method = st.radio(
                    "Cargo Filter Method",
                    ["All Cargo", "Select by Type", "Select by Location"],
                    key="cargo_filter"
                )

                filtered_cargo_df = cargo_df.copy()

                if cargo_filter_method == "Select by Type":
                    selected_cargo_types = st.multiselect(
                        "Choose Cargo Types",
                        options=sorted(cargo_df['Cargo_Type'].unique()),
                        default=sorted(cargo_df['Cargo_Type'].unique())
                    )
                    filtered_cargo_df = cargo_df[cargo_df['Cargo_Type'].isin(selected_cargo_types)]
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
                    filtered_cargo_df = cargo_df[
                        cargo_df['Origin'].isin(selected_origins) &
                        cargo_df['Delivery_Location'].isin(selected_destinations)
                        ]

                st.markdown("---")

            # Display filtered data preview
            with st.expander("📊 Preview Filtered Data"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Selected Trucks ({len(filtered_trucks_df)} total)")
                    st.dataframe(filtered_trucks_df)
                with col2:
                    st.subheader(f"Selected Cargo ({len(filtered_cargo_df)} total)")
                    # Show Premium column
                    if 'Premium' in filtered_cargo_df.columns:
                        st.dataframe(filtered_cargo_df)
                    else:
                        st.dataframe(filtered_cargo_df)
                        st.warning("No Premium column found in cargo data.")

            # Proceed with optimization only if there are trucks and cargo
            if len(filtered_trucks_df) == 0:
                st.error("No trucks selected! Please adjust your truck filters.")
                return

            if len(filtered_cargo_df) == 0:
                st.error("No cargo selected! Please adjust your cargo filters.")
                return

            # Check if user clicked "Maximize Fleet Profit" or if routes already exist
            calculate_routes = st.button("Maximize Fleet Profit")

            if calculate_routes or not st.session_state.route_chains:
                try:
                    # Initialize the profit optimizer with settings from session state
                    optimizer = FleetProfitOptimizer(
                        max_distance_km=st.session_state.max_distance,
                        max_waiting_hours=st.session_state.max_waiting_hours,
                        price_per_km=st.session_state.price_per_km,
                        price_per_hour=st.session_state.price_per_hour,
                        standard_speed_kmh=st.session_state.standard_speed
                    )

                    # Clear the debug logs before optimization
                    logger.logs = []

                    with st.spinner('Maximizing fleet profit...'):
                        # Optimize fleet profit
                        route_chains, total_profit, rejection_stats, rejection_summary = optimizer.optimize_fleet_profit(
                            filtered_trucks_df,
                            filtered_cargo_df,
                            st.session_state.operation_end_time
                        )

                        # Store in session state
                        st.session_state.route_chains = route_chains
                        st.session_state.total_profit = total_profit
                        st.session_state.rejection_stats = rejection_stats
                        st.session_state.rejection_summary = rejection_summary
                        st.session_state.debug_logs = logger.logs

                except Exception as e:
                    st.error(f"Error during profit optimization: {str(e)}")
                    if st.session_state.show_debug:
                        st.exception(e)
                        # Show logs for debugging
                        st.text_area("Debug Logs", ''.join(logger.logs), height=400)
                    st.info("Please check your input data and try again.")
                    return
            else:
                # Use existing route chains from session state
                route_chains = st.session_state.route_chains
                total_profit = st.session_state.total_profit if 'total_profit' in st.session_state else 0
                rejection_stats = st.session_state.rejection_stats if 'rejection_stats' in st.session_state else {}
                rejection_summary = st.session_state.rejection_summary if 'rejection_summary' in st.session_state else {}

            if route_chains:
                # Display results using the display_route_results function
                display_route_results(route_chains, filtered_trucks_df)

                # Add debug log display if debug mode is enabled
                if st.session_state.show_debug and hasattr(st.session_state,
                                                           'debug_logs') and st.session_state.debug_logs:
                    with st.expander("🔍 Debug Logs"):
                        st.text_area("Optimization Logs", ''.join(st.session_state.debug_logs), height=400)

                # Truck selection for map visualization
                st.header("🗺️ Route Visualization")

                # Let user select which trucks to visualize
                available_truck_ids = [filtered_trucks_df.iloc[idx]['truck_id'] for idx in route_chains.keys()]

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.subheader("Select trucks to visualize:")
                    show_all = st.checkbox("Show all trucks", value=True)

                    if not show_all:
                        st.session_state.selected_trucks_for_map = st.multiselect(
                            "Choose specific trucks to display on map:",
                            options=available_truck_ids,
                            default=[available_truck_ids[0]] if available_truck_ids else []
                        )
                    else:
                        st.session_state.selected_trucks_for_map = available_truck_ids

                with col2:
                    # Filter route chains based on selection
                    if show_all:
                        filtered_route_chains = route_chains
                    else:
                        # Get indices of selected trucks
                        selected_indices = [idx for idx in route_chains.keys()
                                            if filtered_trucks_df.iloc[idx][
                                                'truck_id'] in st.session_state.selected_trucks_for_map]
                        filtered_route_chains = {idx: route_chains[idx] for idx in selected_indices}

                    st.info(
                        "Use the layer controls to show/hide different elements on the map. Unassigned cargo is shown in gray.")

                    # Create the map with selected trucks only
                    map_obj = create_map(filtered_trucks_df, filtered_cargo_df, filtered_route_chains)

                    # Display the map with folium_static
                    folium_static(map_obj, width=1000, height=600)

                # Export functionality
                st.subheader("Export Results")
                export_button = st.download_button(
                    label="📥 Download Route Summary (CSV)",
                    data=generate_route_summary_csv(route_chains, filtered_trucks_df),
                    file_name="profit_route_summary.csv",
                    mime="text/csv"
                )

            else:
                st.warning(
                    "No valid routes found with the current selection. Try adjusting your filters or increasing the maximum distance setting."
                )

                # Display debug information about rejected assignments
                if 'rejection_summary' in st.session_state and st.session_state.rejection_summary:
                    with st.expander("📊 Diagnosis Information (Why no routes were found)"):
                        rejection_summary = st.session_state.rejection_summary

                        st.subheader("Assignment Statistics")
                        st.write(f"Total cargo items: {rejection_summary['total_cargo']}")
                        st.write(f"Assigned cargo items: {rejection_summary['total_assigned']}")
                        st.write(f"Rejected cargo items: {rejection_summary['total_rejected']}")
                        st.write(f"Assignment rate: {rejection_summary['assignment_rate'] * 100:.1f}%")

                        st.subheader("Rejection Reasons Breakdown")
                        if rejection_summary['by_reason']:
                            reasons_df = pd.DataFrame({
                                'Reason': list(rejection_summary['by_reason'].keys()),
                                'Count': list(rejection_summary['by_reason'].values())
                            }).sort_values('Count', ascending=False)

                            st.bar_chart(reasons_df.set_index('Reason'))

                            # Display detailed reasons
                            st.dataframe(reasons_df)
                        else:
                            st.write("No specific rejection reasons found.")

                        if rejection_summary['cargo_with_no_truck']:
                            st.subheader("Cargo with No Compatible Trucks")
                            st.write(
                                f"Number of cargo items with no compatible trucks: {len(rejection_summary['cargo_with_no_truck'])}")

                        # Detailed rejection stats for each cargo
                        st.subheader("Detailed Rejection Statistics")
                        if st.checkbox("Show detailed rejection statistics for each cargo"):
                            for cargo_idx, stats in st.session_state.rejection_stats.items():
                                if cargo_idx not in st.session_state.route_chains.keys() and stats['reasons']:
                                    cargo = stats['cargo']
                                    st.write(
                                        f"Cargo at {cargo['Origin']} to {cargo['Delivery_Location']} (Type: {cargo['Cargo_Type']})")

                                    st.write(f"Attempted with {len(set(stats['attempted_trucks']))} trucks")
                                    st.write("Sample rejection reasons:")
                                    for i, reason in enumerate(stats['reasons'][:5]):  # Show only first 5 reasons
                                        st.write(f"  {i + 1}. Truck {reason['truck_id']}: {reason['reason']}")
                                    st.write("---")

                        # Show full debug logs if debug mode is enabled
                        if st.session_state.show_debug and hasattr(st.session_state,
                                                                   'debug_logs') and st.session_state.debug_logs:
                            st.subheader("Debug Logs")
                            st.text_area("Full Optimization Logs", ''.join(st.session_state.debug_logs), height=400)

                        # Suggest solutions
                        st.subheader("Suggested Solutions")

                        most_common_reasons = sorted(rejection_summary['by_reason'].items(),
                                                     key=lambda x: x[1], reverse=True)

                        if most_common_reasons:
                            top_reason, _ = most_common_reasons[0]

                            if "Distance" in top_reason:
                                st.write("📏 Most rejections are due to distance constraints. Try these solutions:")
                                st.write("1. Increase the Maximum Distance parameter in the sidebar")
                                st.write("2. Filter cargo to locations closer to truck starting positions")
                                st.write("3. Add more trucks in different locations")

                            elif "Type mismatch" in top_reason:
                                st.write(
                                    "🔄 Most rejections are due to truck-cargo type mismatches. Try these solutions:")
                                st.write("1. Make sure you have trucks of the same types as your cargo")
                                st.write("2. Filter to show only matching truck and cargo types")

                            elif "Waiting time" in top_reason:
                                st.write("⏱️ Most rejections are due to waiting time constraints. Try these solutions:")
                                st.write("1. Increase the Maximum Waiting Hours parameter in the sidebar")
                                st.write("2. Filter cargo with time windows closer to truck availability times")

                            elif "after cargo window ends" in top_reason:
                                st.write(
                                    "🕒 Most rejections are because trucks arrive after cargo availability window. Try these solutions:")
                                st.write("1. Filter cargo with later time windows")
                                st.write("2. Use trucks that are available earlier")

                            elif "operation end time" in top_reason:
                                st.write(
                                    "⏳ Most rejections are because deliveries would complete after operation end time. Try these solutions:")
                                st.write("1. Extend the Operation End Time in the sidebar")
                                st.write("2. Filter to cargo with earlier availability times")

                            elif "Unprofitable" in top_reason:
                                st.write(
                                    "💰 Most rejections are because deliveries would be unprofitable. Try these solutions:")
                                st.write("1. Adjust Price per km or Price per hour in the sidebar")
                                st.write("2. Filter to cargo with higher Premium values")

                            else:
                                st.write(
                                    "Try adjusting the parameters in the sidebar to address the rejection reasons above")

                        # Sample diagnostic values
                        with st.expander("Current parameter values"):
                            st.write(f"Price per km: {st.session_state.price_per_km} EUR")
                            st.write(f"Price per hour: {st.session_state.price_per_hour} EUR")
                            st.write(f"Maximum Distance: {st.session_state.max_distance} km")
                            st.write(f"Maximum Waiting Hours: {st.session_state.max_waiting_hours} hours")
                            st.write(f"Standard Speed: {st.session_state.standard_speed} km/h")
                            st.write(
                                f"Operation End Time: {st.session_state.operation_end_time.strftime('%Y-%m-%d %H:%M')}")

                            # Truck types summary
                            truck_types = filtered_trucks_df['truck type'].value_counts().to_dict()
                            st.write("Truck types available:")
                            for t_type, count in truck_types.items():
                                st.write(f"  - {t_type}: {count} trucks")

                            # Cargo types summary
                            cargo_types = filtered_cargo_df['Cargo_Type'].value_counts().to_dict()
                            st.write("Cargo types to deliver:")
                            for c_type, count in cargo_types.items():
                                st.write(f"  - {c_type}: {count} items")

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            if st.session_state.show_debug:
                st.exception(e)
            st.info("Please check your input data and try again.")
    else:
        show_welcome_message()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Restore original stdout when app exits
        sys.stdout = sys.__stdout__