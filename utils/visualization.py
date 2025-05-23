# utils/visualization.py (Enhanced for extended time horizons and profit display)
import folium
import pandas as pd
from datetime import datetime, timedelta
from folium import plugins
import numpy as np


def create_extended_map(trucks_df, cargo_df, route_chains, show_timeline=True, show_profit=True):
    """Create an interactive map with enhanced features for extended time horizons"""
    # Calculate map center
    all_lats = pd.concat([
        trucks_df['Latitude (dropoff)'],
        cargo_df['Origin_Latitude'],
        cargo_df['Delivery_Latitude']
    ])
    all_lons = pd.concat([
        trucks_df['Longitude (dropoff)'],
        cargo_df['Origin_Longitude'],
        cargo_df['Delivery_Longitude']
    ])

    # Create base map with better default settings
    m = folium.Map(
        location=[all_lats.mean(), all_lons.mean()],
        zoom_start=6,
        tiles='cartodbpositron'
    )

    # Add fullscreen control
    plugins.Fullscreen().add_to(m)

    # Add measurement tool
    plugins.MeasureControl().add_to(m)

    # Create feature groups for filtering
    trucks_group = folium.FeatureGroup(name='🚚 Trucks')
    pickups_group = folium.FeatureGroup(name='📦 Pickup Points')
    deliveries_group = folium.FeatureGroup(name='🎯 Delivery Points')
    rest_stops_group = folium.FeatureGroup(name='🛌 Rest Stops')
    routes_group = folium.FeatureGroup(name='🛣️ Routes')
    unassigned_group = folium.FeatureGroup(name='❌ Unassigned Cargo')
    timeline_group = folium.FeatureGroup(name='📅 Timeline Markers')

    # Define colors for trucks (more variety for extended fleets)
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'darkblue',
              'darkgreen', 'cadetblue', 'pink', 'lightblue', 'lightgreen', 'gray', 'black']

    # Keep track of assigned cargo and time spans
    assigned_cargo_indices = set()
    all_times = []

    for _, chain in route_chains.items():
        for route in chain:
            assigned_cargo_indices.add(route['cargo'].name)
            all_times.extend([route['pickup_time'], route['delivery_time']])

    # Calculate time span for color coding
    if all_times:
        min_time = min(all_times)
        max_time = max(all_times)
        time_span = (max_time - min_time).total_seconds()
    else:
        min_time = max_time = datetime.now()
        time_span = 1

    # Add data to map
    for truck_idx, route_chain in route_chains.items():
        truck = trucks_df.iloc[truck_idx]
        color = colors[truck_idx % len(colors)]

        # Calculate total metrics for this truck
        total_profit = sum(route['profit'] for route in route_chain)
        total_distance = sum(route['total_distance'] for route in route_chain)
        total_deliveries = len(route_chain)

        # Calculate operation span for this truck
        if route_chain:
            truck_start = min(route['pickup_time'] for route in route_chain)
            truck_end = max(route['delivery_time'] for route in route_chain)
            operation_days = (truck_end - truck_start).days
        else:
            operation_days = 0

        # Enhanced truck marker
        truck_popup = f"""
            <div style='min-width: 250px; font-family: Arial, sans-serif;'>
                <h4 style='color: {color}; margin-bottom: 10px;'>🚚 Truck {truck['truck_id']}</h4>
                <table style='width: 100%; font-size: 12px;'>
                    <tr><td><b>Type:</b></td><td>{truck['truck type']}</td></tr>
                    <tr><td><b>Location:</b></td><td>{truck['Address (drop off)']}</td></tr>
                    <tr><td><b>Routes:</b></td><td>{total_deliveries}</td></tr>
                    <tr><td><b>Total Distance:</b></td><td>{total_distance:.1f} km</td></tr>
                    <tr><td><b>Total Profit:</b></td><td>€{total_profit:.2f}</td></tr>
                    <tr><td><b>Operation Days:</b></td><td>{operation_days}</td></tr>
                    <tr><td><b>Profit/Day:</b></td><td>€{(total_profit / max(operation_days, 1)):.2f}</td></tr>
                </table>
            </div>
        """

        # Add truck marker with enhanced styling
        folium.Marker(
            [truck['Latitude (dropoff)'], truck['Longitude (dropoff)']],
            popup=truck_popup,
            icon=folium.Icon(color=color, icon='truck', prefix='fa'),
            tooltip=f"Truck {truck['truck_id']}: €{total_profit:.0f} profit, {operation_days}d"
        ).add_to(trucks_group)

        # Starting position
        current_pos = (truck['Latitude (dropoff)'], truck['Longitude (dropoff)'])

        # Add routes with enhanced information
        for i, route in enumerate(route_chain):
            cargo = route['cargo']
            profit = route['profit']

            # Color intensity based on time (earlier = lighter, later = darker)
            if time_span > 0:
                time_ratio = (route['pickup_time'] - min_time).total_seconds() / time_span
                opacity = 0.4 + (time_ratio * 0.6)  # Range from 0.4 to 1.0
            else:
                opacity = 0.7

            # Pickup location
            pickup_coords = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])

            # Draw route to pickup with time-based styling
            folium.PolyLine(
                [current_pos, pickup_coords],
                color=color,
                weight=4,
                opacity=opacity,
                tooltip=f"Pickup Route {i + 1}: {route['distance_to_pickup']:.1f} km, €{profit:.0f}"
            ).add_to(routes_group)

            # Enhanced pickup marker
            pickup_popup = f"""
                <div style='min-width: 220px; font-family: Arial, sans-serif;'>
                    <h4 style='color: {color}; margin-bottom: 10px;'>📦 Pickup {i + 1}</h4>
                    <table style='width: 100%; font-size: 12px;'>
                        <tr><td><b>Truck:</b></td><td>{truck['truck_id']}</td></tr>
                        <tr><td><b>Cargo:</b></td><td>{cargo['Cargo_Type']}</td></tr>
                        <tr><td><b>Location:</b></td><td>{cargo['Origin']}</td></tr>
                        <tr><td><b>Date:</b></td><td>{route['pickup_time'].strftime('%Y-%m-%d')}</td></tr>
                        <tr><td><b>Time:</b></td><td>{route['pickup_time'].strftime('%H:%M')}</td></tr>
                        <tr><td><b>Premium:</b></td><td>€{cargo['Premium']:.2f}</td></tr>
                        <tr><td><b>Profit:</b></td><td>€{profit:.2f}</td></tr>
                        <tr><td><b>Distance:</b></td><td>{route['distance_to_pickup']:.1f} km</td></tr>
                        {f"<tr><td><b>Waiting:</b></td><td>{route['waiting_time']:.1f}h</td></tr>" if route.get('waiting_time', 0) > 0 else ""}
                    </table>
                </div>
            """

            # Pickup marker with profit-based icon
            icon_color = 'green' if profit > 100 else 'orange' if profit > 0 else 'red'
            folium.Marker(
                pickup_coords,
                popup=pickup_popup,
                icon=folium.Icon(color=icon_color, icon='play', prefix='fa'),
                tooltip=f"P{i + 1}: {route['pickup_time'].strftime('%m-%d %H:%M')} €{profit:.0f}"
            ).add_to(pickups_group)

            # Add timeline marker if enabled
            if show_timeline:
                folium.CircleMarker(
                    pickup_coords,
                    radius=8,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.3,
                    popup=f"Day {(route['pickup_time'] - min_time).days + 1}",
                    tooltip=f"Day {(route['pickup_time'] - min_time).days + 1}"
                ).add_to(timeline_group)

            # Add rest stops on way to pickup
            for rest_idx, rest in enumerate(route.get('rest_stops_to_pickup', [])):
                folium.Marker(
                    rest['location'],
                    popup=f"Rest Stop: {rest['duration'] * 60:.0f} min<br>Type: {rest.get('type', 'break')}",
                    icon=folium.Icon(color='red', icon='bed', prefix='fa', size=(20, 20))
                ).add_to(rest_stops_group)

            # Delivery location
            delivery_coords = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])

            # Draw route from pickup to delivery
            folium.PolyLine(
                [pickup_coords, delivery_coords],
                color=color,
                weight=4,
                opacity=opacity,
                tooltip=f"Delivery Route {i + 1}: {route['distance_to_delivery']:.1f} km"
            ).add_to(routes_group)

            # Enhanced delivery marker
            delivery_popup = f"""
                <div style='min-width: 220px; font-family: Arial, sans-serif;'>
                    <h4 style='color: {color}; margin-bottom: 10px;'>🎯 Delivery {i + 1}</h4>
                    <table style='width: 100%; font-size: 12px;'>
                        <tr><td><b>Truck:</b></td><td>{truck['truck_id']}</td></tr>
                        <tr><td><b>Cargo:</b></td><td>{cargo['Cargo_Type']}</td></tr>
                        <tr><td><b>Location:</b></td><td>{cargo['Delivery_Location']}</td></tr>
                        <tr><td><b>Date:</b></td><td>{route['delivery_time'].strftime('%Y-%m-%d')}</td></tr>
                        <tr><td><b>Time:</b></td><td>{route['delivery_time'].strftime('%H:%M')}</td></tr>
                        <tr><td><b>Distance:</b></td><td>{route['distance_to_delivery']:.1f} km</td></tr>
                        <tr><td><b>Total Route:</b></td><td>{route['total_distance']:.1f} km</td></tr>
                    </table>
                </div>
            """

            folium.Marker(
                delivery_coords,
                popup=delivery_popup,
                icon=folium.Icon(color=icon_color, icon='stop', prefix='fa'),
                tooltip=f"D{i + 1}: {route['delivery_time'].strftime('%m-%d %H:%M')}"
            ).add_to(deliveries_group)

            # Add rest stops on way to delivery
            for rest in route.get('rest_stops_to_delivery', []):
                folium.Marker(
                    rest['location'],
                    popup=f"Rest Stop: {rest['duration'] * 60:.0f} min<br>Type: {rest.get('type', 'break')}",
                    icon=folium.Icon(color='red', icon='bed', prefix='fa')
                ).add_to(rest_stops_group)

            # Update current position for next route
            current_pos = delivery_coords

    # Add unassigned cargo to the map with enhanced information
    for idx, cargo in cargo_df.iterrows():
        if idx not in assigned_cargo_indices:
            # Enhanced popup for unassigned cargo
            premium_text = f"<tr><td><b>Premium:</b></td><td>€{cargo['Premium']:.2f}</td></tr>" if 'Premium' in cargo else ""

            # Calculate distance for context
            origin = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])
            delivery = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])

            try:
                from geopy.distance import geodesic
                distance = geodesic(origin, delivery).kilometers
                distance_text = f"<tr><td><b>Distance:</b></td><td>{distance:.1f} km</td></tr>"
            except:
                distance_text = ""

            unassigned_popup = f"""
                <div style='min-width: 200px; font-family: Arial, sans-serif;'>
                    <h4 style='color: gray; margin-bottom: 10px;'>❌ Unassigned Cargo</h4>
                    <table style='width: 100%; font-size: 12px;'>
                        <tr><td><b>Type:</b></td><td>{cargo['Cargo_Type']}</td></tr>
                        <tr><td><b>Origin:</b></td><td>{cargo['Origin']}</td></tr>
                        <tr><td><b>Destination:</b></td><td>{cargo['Delivery_Location']}</td></tr>
                        <tr><td><b>Available:</b></td><td>{pd.to_datetime(cargo['Available_From']).strftime('%m-%d %H:%M')} to {pd.to_datetime(cargo['Available_To']).strftime('%m-%d %H:%M')}</td></tr>
                        {premium_text}
                        {distance_text}
                    </table>
                </div>
            """

            # Add marker at origin
            folium.Marker(
                [cargo['Origin_Latitude'], cargo['Origin_Longitude']],
                popup=unassigned_popup,
                icon=folium.Icon(color='gray', icon='circle', prefix='fa'),
                tooltip=f"Unassigned {cargo['Cargo_Type']}: €{cargo.get('Premium', 0):.0f}"
            ).add_to(unassigned_group)

            # Show potential route with dotted line
            folium.PolyLine(
                [(cargo['Origin_Latitude'], cargo['Origin_Longitude']),
                 (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])],
                color='gray',
                weight=2,
                opacity=0.5,
                dash_array='10, 10',
                tooltip=f"Unassigned potential route"
            ).add_to(unassigned_group)

            # Add small marker at destination
            folium.CircleMarker(
                [cargo['Delivery_Latitude'], cargo['Delivery_Longitude']],
                radius=5,
                color='gray',
                fill=True,
                fill_color='gray',
                fillOpacity=0.5,
                tooltip=f"Destination: {cargo['Delivery_Location']}"
            ).add_to(unassigned_group)

    # Add all feature groups to the map
    trucks_group.add_to(m)
    pickups_group.add_to(m)
    deliveries_group.add_to(m)
    rest_stops_group.add_to(m)
    routes_group.add_to(m)
    unassigned_group.add_to(m)

    if show_timeline:
        timeline_group.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Add enhanced legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 280px; 
                border:2px solid grey; z-index:9999; font-size:12px;
                background-color:white; padding: 10px;
                border-radius: 5px; font-family: Arial, sans-serif;">
        <div style="font-weight: bold; font-size: 14px; margin-bottom: 8px;">🗺️ Map Legend</div>

        <div style="margin: 4px 0;">
            <i class="fa fa-truck" style="color:blue;"></i> <b>Truck Starting Points</b>
        </div>
        <div style="margin: 4px 0;">
            <i class="fa fa-play" style="color:green;"></i> Profitable Pickup (€100+)
        </div>
        <div style="margin: 4px 0;">
            <i class="fa fa-play" style="color:orange;"></i> Low Profit Pickup (€0-100)
        </div>
        <div style="margin: 4px 0;">
            <i class="fa fa-stop" style="color:green;"></i> Delivery Points
        </div>
        <div style="margin: 4px 0;">
            <i class="fa fa-bed" style="color:red;"></i> Mandatory Rest Stops
        </div>
        <div style="margin: 4px 0;">
            <i class="fa fa-circle" style="color:gray;"></i> Unassigned Cargo
        </div>

        <hr style="margin: 8px 0;">

        <div style="font-size: 11px;">
            <div><b>Route Lines:</b></div>
            <div>• Solid: Active routes</div>
            <div>• Opacity shows timing</div>
            <div>• Dotted: Unassigned</div>
        </div>

        <hr style="margin: 8px 0;">

        <div style="font-size: 11px;">
            <div><b>Time Span:</b></div>
            <div>{min_time.strftime('%Y-%m-%d') if all_times else 'N/A'} to</div>
            <div>{max_time.strftime('%Y-%m-%d') if all_times else 'N/A'}</div>
            <div>({(max_time - min_time).days if all_times else 0} days)</div>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add time controls if timeline is enabled
    if show_timeline and all_times:
        time_control_html = f'''
        <div style="position: fixed; 
                    top: 100px; right: 20px; width: 250px; height: 120px;
                    border:2px solid grey; z-index:9999; font-size:12px;
                    background-color:white; padding: 10px;
                    border-radius: 5px; font-family: Arial, sans-serif;">
            <div style="font-weight: bold; margin-bottom: 8px;">📅 Timeline Info</div>
            <div><b>Operation Start:</b> {min_time.strftime('%Y-%m-%d %H:%M')}</div>
            <div><b>Operation End:</b> {max_time.strftime('%Y-%m-%d %H:%M')}</div>
            <div><b>Total Duration:</b> {(max_time - min_time).days} days</div>
            <div><b>Active Routes:</b> {sum(len(chain) for chain in route_chains.values())}</div>
            <div style="margin-top: 8px; font-size: 10px; color: #666;">
                Route opacity indicates timing:<br>
                Lighter = Earlier, Darker = Later
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(time_control_html))

    return m


# Maintain backward compatibility
def create_map(trucks_df, cargo_df, route_chains, show_profit=True):
    """Backward compatible function that calls the enhanced version"""
    return create_extended_map(trucks_df, cargo_df, route_chains, show_timeline=True, show_profit=show_profit)