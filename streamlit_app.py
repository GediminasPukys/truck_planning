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
from utils.extended_profit_optimizer import ExtendedFleetProfitOptimizer
from utils.time_horizon_planner import TimeHorizonPlanner, MultiDayRoutePlanner
from utils.visualization import create_map
from utils.route_planner import RoutePlanner


def calculate_suggested_operation_period(cargo_df):
    """Calculate suggested operation period based on cargo data"""
    if cargo_df is None or 'Available_To' not in cargo_df.columns:
        return None, None

    try:
        cargo_df['Available_From'] = pd.to_datetime(cargo_df['Available_From'])
        cargo_df['Available_To'] = pd.to_datetime(cargo_df['Available_To'])

        earliest_cargo = cargo_df['Available_From'].min()
        latest_cargo = cargo_df['Available_To'].max()

        # Add buffer for delivery completion
        suggested_end = latest_cargo + timedelta(days=3)

        return earliest_cargo, suggested_end
    except:
        return None, None


def analyze_time_distribution(cargo_df):
    """Analyze cargo distribution across time"""
    if cargo_df is None:
        return {}

    try:
        cargo_df['Available_From'] = pd.to_datetime(cargo_df['Available_From'])
        cargo_df['Available_To'] = pd.to_datetime(cargo_df['Available_To'])

        # Group by week
        cargo_df['Week'] = cargo_df['Available_From'].dt.isocalendar().week
        weekly_distribution = cargo_df.groupby('Week').size().to_dict()

        # Group by day
        cargo_df['Date'] = cargo_df['Available_From'].dt.date
        daily_distribution = cargo_df.groupby('Date').size().to_dict()

        return {
            'weekly': weekly_distribution,
            'daily': daily_distribution,
            'earliest': cargo_df['Available_From'].min(),
            'latest': cargo_df['Available_To'].max(),
            'total_span_days': (cargo_df['Available_To'].max() - cargo_df['Available_From'].min()).days
        }
    except:
        return {}


def show_welcome_message():
    """Display welcome message and instructions"""
    st.markdown("""
    ## 👋 Welcome to the Extended Truck-Cargo Assignment Profit Optimizer!

    This enhanced application helps optimize the assignment of trucks to cargo loads across **extended time horizons** by:
    - Maximizing total profit across the fleet over weeks or months
    - Planning multi-day routes with proper rest periods
    - Respecting time windows for pickup and delivery across extended periods
    - Matching truck and cargo types
    - Ensuring maximum distance and waiting time constraints are met
    - Optimizing fleet utilization across the entire operating period

    ### ✨ New Extended Features:
    - **Multi-Day Planning**: Plan routes across weeks and months
    - **Automatic Time Horizon Detection**: Automatically sets operation period based on cargo data
    - **Time Distribution Analysis**: Shows cargo distribution across time periods
    - **Extended Route Planning**: Handles overnight stops and multi-day routes
    - **Improved Optimization**: Better algorithms for longer planning horizons

    ### 📝 How to use:
    1. Upload your trucks and cargo CSV files using the sidebar
    2. Or click "Use Sample Data" to try the app with example data
    3. The system will automatically detect and suggest the optimal operation period
    4. Set parameters like maximum distance, waiting time, prices
    5. Choose between Quick Optimization (original) or Extended Optimization (new)
    6. Click "Maximize Fleet Profit" to run the optimization
    7. Review the profit results and route assignments across the entire time horizon
    8. Explore the interactive map visualization
    9. Filter trucks and routes using the map controls

    ### 📄 Required file formats:

    **Trucks CSV:**
    - truck_id, truck type, Address (drop off)
    - Latitude (dropoff), Longitude (dropoff), Timestamp (dropoff)
    - avg moving speed, km/h, price per km, Eur, waiting time price per h, EUR

    **Cargo CSV:**
    - Origin, Origin_Latitude, Origin_Longitude
    - Available_From, Available_To, Delivery_Location
    - Delivery_Latitude, Delivery_Longitude, Cargo_Type
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
                "Rest Stops": len(route.get('rest_stops_to_pickup', [])) + len(route.get('rest_stops_to_delivery', [])),
                "Premium (EUR)": f"{cargo['Premium']:.2f}",
                "Profit (EUR)": f"{route['profit']:.2f}"
            })

    return pd.DataFrame(data).to_csv(index=False).encode('utf-8')


def display_time_horizon_analysis(cargo_df):
    """Display time horizon analysis"""
    st.subheader("📅 Time Horizon Analysis")

    time_analysis = analyze_time_distribution(cargo_df)

    if time_analysis:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Time Span", f"{time_analysis['total_span_days']} days")
            st.metric("Earliest Cargo", time_analysis['earliest'].strftime('%Y-%m-%d'))

        with col2:
            st.metric("Latest Cargo", time_analysis['latest'].strftime('%Y-%m-%d'))
            st.metric("Total Cargo Items", len(cargo_df))

        with col3:
            avg_per_day = len(cargo_df) / max(time_analysis['total_span_days'], 1)
            st.metric("Avg Cargo/Day", f"{avg_per_day:.1f}")

        # Show weekly distribution
        if time_analysis['weekly']:
            st.write("**Weekly Distribution:**")
            weekly_df = pd.DataFrame(
                list(time_analysis['weekly'].items()),
                columns=['Week', 'Cargo Count']
            )
            st.bar_chart(weekly_df.set_index('Week'))

        # Show daily distribution (sample)
        if time_analysis['daily'] and len(time_analysis['daily']) <= 31:
            st.write("**Daily Distribution:**")
            daily_df = pd.DataFrame([
                {'Date': str(date), 'Count': count}
                for date, count in sorted(time_analysis['daily'].items())
            ])
            st.line_chart(daily_df.set_index('Date'))


def display_route_results(route_chains, trucks_df):
    """Display route planning results with profit information in Streamlit"""
    # Summary metrics
    total_deliveries = sum(len(chain) for chain in route_chains.values())
    total_distance = sum(
        sum(route['total_distance'] for route in chain)
        for chain in route_chains.values()
    )
    total_rest_stops = sum(
        sum(len(route.get('rest_stops_to_pickup', [])) + len(route.get('rest_stops_to_delivery', []))
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

    # Calculate time span of operations
    all_pickup_times = []
    all_delivery_times = []
    for chain in route_chains.values():
        for route in chain:
            all_pickup_times.append(route['pickup_time'])
            all_delivery_times.append(route['delivery_time'])

    if all_pickup_times and all_delivery_times:
        operation_start = min(all_pickup_times)
        operation_end = max(all_delivery_times)
        operation_span = (operation_end - operation_start).days
    else:
        operation_span = 0

    # Display in a nice grid of metrics
    st.header("📊 Extended Optimization Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Deliveries", total_deliveries)
        st.metric("Total Distance", f"{total_distance:.1f} km")
    with col2:
        st.metric("Total Waiting Time", f"{total_waiting_hours:.1f} h")
        st.metric("Required Rest Stops", total_rest_stops)
    with col3:
        st.metric("Total Premium", f"{total_premium:.2f} EUR")
        st.metric("Total Profit", f"{total_profit:.2f} EUR")
    with col4:
        st.metric("Operation Span", f"{operation_span} days")
        if total_deliveries > 0:
            st.metric("Profit per Delivery", f"{total_profit / total_deliveries:.2f} EUR")

    # Display profit by truck
    st.subheader("Profit by Truck")

    profit_by_truck = {}
    for truck_idx, chain in route_chains.items():
        truck_id = trucks_df.iloc[truck_idx]['truck_id']
        profit = sum(route['profit'] for route in chain)
        profit_by_truck[truck_id] = profit

    # Create a simple bar chart of profit by truck
    if profit_by_truck:
        fig, ax = plt.subplots(figsize=(12, 6))
        trucks = list(profit_by_truck.keys())
        profits = list(profit_by_truck.values())

        bars = ax.bar(trucks, profits)
        ax.set_xlabel('Truck ID')
        ax.set_ylabel('Profit (EUR)')
        ax.set_title('Profit by Truck - Extended Time Horizon')

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # Timeline visualization
    if all_pickup_times and all_delivery_times:
        st.subheader("📅 Operation Timeline")
        timeline_data = []

        for truck_idx, chain in route_chains.items():
            truck_id = trucks_df.iloc[truck_idx]['truck_id']
            for i, route in enumerate(chain):
                timeline_data.append({
                    'Truck': f"Truck {truck_id}",
                    'Route': i + 1,
                    'Start': route['pickup_time'].date(),
                    'End': route['delivery_time'].date(),
                    'Profit': route['profit']
                })

        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)

            # Group by date to show daily activity
            daily_activity = timeline_df.groupby('Start').agg({
                'Truck': 'count',
                'Profit': 'sum'
            }).rename(columns={'Truck': 'Active Deliveries'})

            st.write("**Daily Activity Overview:**")
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(daily_activity['Active Deliveries'])
            with col2:
                st.line_chart(daily_activity['Profit'])

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

    # Calculate time span for this truck
    if chain:
        start_time = min(route['pickup_time'] for route in chain)
        end_time = max(route['delivery_time'] for route in chain)
        time_span = (end_time - start_time).days
    else:
        time_span = 0

    # Show truck info with profit metrics
    st.write(
        f"**Type:** {truck['truck type']} • **Speed:** {truck['avg moving speed, km/h']} km/h • "
        f"**Price/km:** €{truck['price per km, Eur']} • **Waiting price/h:** €{truck['waiting time price per h, EUR']}"
    )

    # Show profit summary
    st.write(
        f"**Total Profit:** €{total_profit:.2f} • **Total Premium:** €{total_premium:.2f} • "
        f"**Total Distance:** {total_distance:.1f} km • **Operation Span:** {time_span} days"
    )

    # Starting position info
    st.write("**Starting Position:**")
    st.write(
        f"Location: {truck['Address (drop off)']} • Coordinates: ({truck['Latitude (dropoff)']:.4f}, {truck['Longitude (dropoff)']:.4f}) • "
        f"Available from: {pd.to_datetime(truck['Timestamp (dropoff)']).strftime('%Y-%m-%d %H:%M')}"
    )

    # Create route summary table
    route_data = []
    for i, route in enumerate(chain, 1):
        cargo = route['cargo']
        rest_stops_count = len(route.get('rest_stops_to_pickup', [])) + len(route.get('rest_stops_to_delivery', []))

        route_data.append({
            "Stop": i,
            "Cargo Type": cargo['Cargo_Type'],
            "Origin": cargo['Origin'],
            "Destination": cargo['Delivery_Location'],
            "Pickup Date": route['pickup_time'].strftime('%Y-%m-%d'),
            "Pickup Time": route['pickup_time'].strftime('%H:%M'),
            "Delivery Date": route['delivery_time'].strftime('%Y-%m-%d'),
            "Delivery Time": route['delivery_time'].strftime('%H:%M'),
            "Distance (km)": f"{route['total_distance']:.1f}",
            "Waiting (h)": f"{route['waiting_time']:.1f}",
            "Rest Stops": rest_stops_count,
            "Premium (€)": f"{cargo['Premium']:.2f}",
            "Profit (€)": f"{route['profit']:.2f}"
        })

    # Display route summary
    if route_data:
        route_df = pd.DataFrame(route_data)
        st.write("### Route Summary")
        st.dataframe(route_df, use_container_width=True)

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
            "Time Cost": f"{breakdown.get('time_cost', 0):.2f}",
            "Total Cost": f"{breakdown.get('total_cost', 0):.2f}",
            "Profit": f"{route['profit']:.2f}"
        })

    profit_df = pd.DataFrame(profit_breakdown)
    st.dataframe(profit_df, use_container_width=True)


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
            st.info("Adding Premium column to sample data...")

            # Generate random premiums based on distance
            if 'Distance' in cargo_df.columns:
                # Use existing Distance column if available
                cargo_df['Premium'] = cargo_df['Distance'] * np.random.uniform(1.5, 2.5, len(cargo_df))
            else:
                # Calculate distance between origin and delivery
                distances = []
                for _, cargo in cargo_df.iterrows():
                    try:
                        origin = (float(cargo['Origin_Latitude']), float(cargo['Origin_Longitude']))
                        delivery = (float(cargo['Delivery_Latitude']), float(cargo['Delivery_Longitude']))
                        distance = geodesic(origin, delivery).kilometers
                        distances.append(distance)
                    except Exception as e:
                        st.warning(f"Error calculating distance for row {cargo.name}: {str(e)}")
                        distances.append(200)  # Default distance

                # Generate premiums based on distances
                cargo_df['Premium'] = [dist * np.random.uniform(1.5, 2.5) for dist in distances]

        # Round premium values for cleaner display and ensure they're positive
        cargo_df['Premium'] = np.abs(cargo_df['Premium'].fillna(300)).round(2)

        # Ensure data types are correct
        try:
            trucks_df['Latitude (dropoff)'] = pd.to_numeric(trucks_df['Latitude (dropoff)'], errors='coerce')
            trucks_df['Longitude (dropoff)'] = pd.to_numeric(trucks_df['Longitude (dropoff)'], errors='coerce')
            cargo_df['Origin_Latitude'] = pd.to_numeric(cargo_df['Origin_Latitude'], errors='coerce')
            cargo_df['Origin_Longitude'] = pd.to_numeric(cargo_df['Origin_Longitude'], errors='coerce')
            cargo_df['Delivery_Latitude'] = pd.to_numeric(cargo_df['Delivery_Latitude'], errors='coerce')
            cargo_df['Delivery_Longitude'] = pd.to_numeric(cargo_df['Delivery_Longitude'], errors='coerce')
        except Exception as e:
            st.warning(f"Warning converting coordinates to numeric: {str(e)}")

        return trucks_df, cargo_df

    except FileNotFoundError as e:
        st.error(f"Sample data files not found: {str(e)}")
        st.info("Please make sure 'trucks.csv' and 'cargos.csv' are in the same directory as the app.")
        return None, None
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        st.info("Please check the format of your sample data files.")
        return None, None


def main():
    st.set_page_config(
        page_title="Extended Truck-Cargo Profit Maximizer",
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
    if 'selected_trucks_for_map' not in st.session_state:
        st.session_state.selected_trucks_for_map = []
    if 'optimization_mode' not in st.session_state:
        st.session_state.optimization_mode = "Extended"

    # Initialize operation_end_time differently - will be set based on data
    if 'operation_end_time' not in st.session_state:
        st.session_state.operation_end_time = None

    st.title("🚚 Extended Truck-Cargo Fleet Profit Maximizer")

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

        # Optimization Mode Selection
        st.subheader("Optimization Mode")
        st.session_state.optimization_mode = st.selectbox(
            "Choose Optimization Mode",
            ["Extended", "Quick"],
            index=0,
            help="Extended: Multi-day planning with automatic time horizon detection. Quick: Original single-period optimization."
        )

        if st.session_state.optimization_mode == "Extended":
            st.info(
                "🌟 Extended mode automatically detects the optimal operation period from your cargo data and plans multi-day routes.")

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
            help="Price per hour for time cost calculation"
        )

        # Extended settings for Extended mode
        if st.session_state.optimization_mode == "Extended":
            st.subheader("Extended Planning Settings")

            max_planning_days = st.number_input(
                "Maximum Planning Horizon (days)",
                value=60,
                min_value=7,
                max_value=365,
                help="Maximum number of days to plan ahead"
            )

            planning_window_days = st.number_input(
                "Planning Window Size (days)",
                value=7,
                min_value=1,
                max_value=14,
                help="Size of each planning window for optimization"
            )

        # Operating period settings (only for Quick mode or manual override)
        if st.session_state.optimization_mode == "Quick":
            st.subheader("Operation Period")

            # Set default operation end time
            if st.session_state.operation_end_time is None:
                st.session_state.operation_end_time = (datetime.now() + timedelta(days=30)).replace(
                    hour=18, minute=0, second=0, microsecond=0)

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

            # Auto-detect operation period for Extended Mode
            if st.session_state.optimization_mode == "Extended":
                earliest_cargo, suggested_end = calculate_suggested_operation_period(cargo_df)

                if earliest_cargo and suggested_end:
                    st.session_state.operation_end_time = suggested_end

                    # Display auto-detected period
                    with st.expander("🔍 Auto-Detected Operation Period", expanded=True):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Earliest Cargo Available", earliest_cargo.strftime('%Y-%m-%d'))
                        with col2:
                            st.metric("Latest Cargo Window Ends",
                                      pd.to_datetime(cargo_df['Available_To']).max().strftime('%Y-%m-%d'))
                        with col3:
                            st.metric("Suggested Operation End", suggested_end.strftime('%Y-%m-%d'))

                        st.info(
                            f"✅ Extended mode automatically set operation period from {earliest_cargo.strftime('%Y-%m-%d')} to {suggested_end.strftime('%Y-%m-%d')}")
                else:
                    st.warning("Could not auto-detect operation period. Using default 30-day period.")
                    st.session_state.operation_end_time = (datetime.now() + timedelta(days=30)).replace(
                        hour=18, minute=0, second=0, microsecond=0)

            # Display time horizon analysis
            display_time_horizon_analysis(cargo_df)

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
                    ["All Cargo", "Select by Type", "Select by Location", "Select by Time Period"],
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
                elif cargo_filter_method == "Select by Time Period":
                    # Time period selection
                    cargo_df['Available_From'] = pd.to_datetime(cargo_df['Available_From'])
                    cargo_df['Available_To'] = pd.to_datetime(cargo_df['Available_To'])

                    min_date = cargo_df['Available_From'].min().date()
                    max_date = cargo_df['Available_To'].max().date()

                    selected_date_range = st.date_input(
                        "Select Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )

                    if len(selected_date_range) == 2:
                        start_date, end_date = selected_date_range
                        # Filter cargo that overlaps with selected date range
                        filtered_cargo_df = cargo_df[
                            (cargo_df['Available_From'].dt.date <= end_date) &
                            (cargo_df['Available_To'].dt.date >= start_date)
                            ]

                st.markdown("---")

            # Display filtered data preview
            with st.expander("📊 Preview Filtered Data"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Selected Trucks ({len(filtered_trucks_df)} total)")
                    st.dataframe(filtered_trucks_df.head())
                with col2:
                    st.subheader(f"Selected Cargo ({len(filtered_cargo_df)} total)")
                    st.dataframe(filtered_cargo_df.head())

            # Proceed with optimization only if there are trucks and cargo
            if len(filtered_trucks_df) == 0:
                st.error("No trucks selected! Please adjust your truck filters.")
                return

            if len(filtered_cargo_df) == 0:
                st.error("No cargo selected! Please adjust your cargo filters.")
                return

            # Optimization button with mode-specific text
            if st.session_state.optimization_mode == "Extended":
                button_text = "🚀 Maximize Fleet Profit (Extended)"
                button_help = "Run extended optimization with multi-day planning across the detected time horizon"
            else:
                button_text = "⚡ Maximize Fleet Profit (Quick)"
                button_help = "Run quick optimization for the specified time period"

            calculate_routes = st.button(button_text, help=button_help)

            if calculate_routes or not st.session_state.route_chains:
                try:
                    if st.session_state.optimization_mode == "Extended":
                        # Use Extended Optimizer
                        optimizer = ExtendedFleetProfitOptimizer(
                            max_distance_km=st.session_state.max_distance,
                            max_waiting_hours=st.session_state.max_waiting_hours,
                            price_per_km=st.session_state.price_per_km,
                            price_per_hour=st.session_state.price_per_hour,
                            standard_speed_kmh=st.session_state.standard_speed,
                            max_planning_days=max_planning_days,
                            planning_window_days=planning_window_days
                        )

                        with st.spinner('🚀 Running extended optimization across time horizon...'):
                            route_chains, total_profit, rejection_stats, rejection_summary = optimizer.optimize_fleet_profit_extended(
                                filtered_trucks_df,
                                filtered_cargo_df
                            )
                    else:
                        # Use Quick Optimizer
                        optimizer = FleetProfitOptimizer(
                            max_distance_km=st.session_state.max_distance,
                            max_waiting_hours=st.session_state.max_waiting_hours,
                            price_per_km=st.session_state.price_per_km,
                            price_per_hour=st.session_state.price_per_hour,
                            standard_speed_kmh=st.session_state.standard_speed
                        )

                        with st.spinner('⚡ Running quick optimization...'):
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

                except Exception as e:
                    st.error(f"Error during profit optimization: {str(e)}")
                    if st.session_state.show_debug:
                        st.exception(e)
                    st.info("Please check your input data and try again.")
                    return
            else:
                # Use existing route chains from session state
                route_chains = st.session_state.route_chains
                total_profit = st.session_state.total_profit if 'total_profit' in st.session_state else 0
                rejection_stats = st.session_state.rejection_stats if 'rejection_stats' in st.session_state else {}
                rejection_summary = st.session_state.rejection_summary if 'rejection_summary' in st.session_state else {}

            if route_chains:
                # Display results using the enhanced display function
                display_route_results(route_chains, filtered_trucks_df)

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
                    file_name=f"extended_route_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

            else:
                st.warning(
                    "No valid routes found with the current selection. Try adjusting your filters or settings."
                )

                # Display enhanced debug information
                if 'rejection_summary' in st.session_state and st.session_state.rejection_summary:
                    with st.expander("📊 Comprehensive Diagnosis (Why no routes were found)", expanded=True):
                        rejection_summary = st.session_state.rejection_summary

                        # Enhanced assignment statistics
                        st.subheader("📈 Assignment Statistics")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Cargo", rejection_summary['total_cargo'])
                        with col2:
                            st.metric("Assigned", rejection_summary['total_assigned'])
                        with col3:
                            st.metric("Rejected", rejection_summary['total_rejected'])
                        with col4:
                            st.metric("Success Rate", f"{rejection_summary['assignment_rate'] * 100:.1f}%")

                        # Time horizon analysis for failures
                        if st.session_state.optimization_mode == "Extended":
                            st.subheader("⏰ Time Horizon Analysis")
                            time_analysis = analyze_time_distribution(filtered_cargo_df)
                            if time_analysis:
                                st.write(f"**Cargo Time Span**: {time_analysis['total_span_days']} days")
                                st.write(
                                    f"**Operation End Time**: {st.session_state.operation_end_time.strftime('%Y-%m-%d %H:%M')}")

                                cargo_end = time_analysis['latest']
                                if st.session_state.operation_end_time and cargo_end > st.session_state.operation_end_time:
                                    st.error(f"⚠️ Some cargo availability extends beyond operation end time!")
                                    st.write(f"Latest cargo ends: {cargo_end.strftime('%Y-%m-%d')}")
                                    st.write(
                                        f"Operation ends: {st.session_state.operation_end_time.strftime('%Y-%m-%d')}")

                        # Enhanced rejection analysis
                        st.subheader("🔍 Rejection Reasons Breakdown")
                        if rejection_summary['by_reason']:
                            reasons_df = pd.DataFrame({
                                'Reason': list(rejection_summary['by_reason'].keys()),
                                'Count': list(rejection_summary['by_reason'].values()),
                                'Percentage': [count / rejection_summary['total_rejected'] * 100
                                               for count in rejection_summary['by_reason'].values()]
                            }).sort_values('Count', ascending=False)

                            st.dataframe(reasons_df)
                            st.bar_chart(reasons_df.set_index('Reason')['Count'])

                        # Enhanced recommendations
                        st.subheader("💡 Smart Recommendations")

                        most_common_reasons = sorted(rejection_summary['by_reason'].items(),
                                                     key=lambda x: x[1], reverse=True)

                        if most_common_reasons:
                            top_reason, count = most_common_reasons[0]
                            percentage = count / rejection_summary['total_rejected'] * 100

                            st.write(f"**Primary Issue:** {top_reason} ({percentage:.1f}% of rejections)")

                            if "Distance" in top_reason:
                                st.info("📏 **Distance Solutions:**")
                                st.write("1. Increase Maximum Distance parameter")
                                st.write(f"2. Current limit: {st.session_state.max_distance} km")
                                st.write("3. Consider regional clustering of trucks")

                            elif "operation end time" in top_reason:
                                st.info("⏳ **Time Horizon Solutions:**")
                                if st.session_state.optimization_mode == "Extended":
                                    st.write("1. Extended mode should auto-handle this - check data quality")
                                    st.write("2. Verify cargo time windows are reasonable")
                                else:
                                    st.write("1. Switch to Extended optimization mode")
                                    st.write("2. Or extend Operation End Time manually")

                            elif "Type mismatch" in top_reason:
                                st.info("🔄 **Type Compatibility Solutions:**")
                                st.write("1. Check truck and cargo type compatibility")
                                st.write("2. Use type filters to match available combinations")

                            elif "Unprofitable" in top_reason:
                                st.info("💰 **Profitability Solutions:**")
                                st.write("1. Check Premium values in cargo data")
                                st.write("2. Adjust cost parameters (price per km/hour)")
                                st.write("3. Verify cost calculation is correct")

                        # Data quality insights
                        st.subheader("🔧 Data Quality Check")

                        # Check Premium values
                        if 'Premium' in filtered_cargo_df.columns:
                            premium_stats = filtered_cargo_df['Premium'].describe()
                            if premium_stats['min'] <= 0:
                                st.warning("⚠️ Some cargo has zero or negative Premium values")
                            st.write(f"Premium range: €{premium_stats['min']:.2f} - €{premium_stats['max']:.2f}")

                        # Check time windows
                        time_analysis = analyze_time_distribution(filtered_cargo_df)
                        if time_analysis and time_analysis['total_span_days'] > 60:
                            st.info(f"ℹ️ Long planning horizon detected: {time_analysis['total_span_days']} days")
                            if st.session_state.optimization_mode == "Quick":
                                st.write("💡 Consider switching to Extended mode for better handling")

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            if st.session_state.show_debug:
                st.exception(e)
            st.info("Please check your input data and try again.")
    else:
        show_welcome_message()


if __name__ == "__main__":
    main()