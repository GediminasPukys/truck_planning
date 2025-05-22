# utils/visualization.py (Updated for profit display)
import folium
import pandas as pd
from datetime import datetime
from folium import plugins


def create_map(trucks_df, cargo_df, route_chains, show_profit=True):
    """Create an interactive map with truck routes, cargo locations, and profit information"""
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

    # Create base map
    m = folium.Map(
        location=[all_lats.mean(), all_lons.mean()],
        zoom_start=6,
        tiles='cartodbpositron'
    )

    # Add fullscreen control
    plugins.Fullscreen().add_to(m)

    # Create feature groups for filtering
    trucks_group = folium.FeatureGroup(name='Trucks')
    pickups_group = folium.FeatureGroup(name='Pickup Points')
    deliveries_group = folium.FeatureGroup(name='Delivery Points')
    rest_stops_group = folium.FeatureGroup(name='Rest Stops')
    routes_group = folium.FeatureGroup(name='Routes')
    unassigned_group = folium.FeatureGroup(name='Unassigned Cargo')

    # Define colors for trucks
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']

    # Keep track of assigned cargo
    assigned_cargo_indices = set()
    for _, chain in route_chains.items():
        for route in chain:
            assigned_cargo_indices.add(route['cargo'].name)

    # Add data to map
    for truck_idx, route_chain in route_chains.items():
        truck = trucks_df.iloc[truck_idx]
        color = colors[truck_idx % len(colors)]
        
        # Calculate total profit for this truck
        total_profit = sum(route['profit'] for route in route_chain)
        
        # Add truck marker
        truck_popup = f"""
            <div style='min-width: 200px'>
                <h4>Truck {truck['truck_id']}</h4>
                <b>Type:</b> {truck['truck type']}<br>
                <b>Start Location:</b> {truck['Address (drop off)']}<br>
                <b>Routes:</b> {len(route_chain)}<br>
                <b>Total Profit:</b> {total_profit:.2f} EUR
            </div>
        """

        # Add truck marker
        folium.Marker(
            [truck['Latitude (dropoff)'], truck['Longitude (dropoff)']],
            popup=truck_popup,
            icon=folium.Icon(color=color, icon='truck', prefix='fa'),
            tooltip=f"Truck {truck['truck_id']} (Profit: {total_profit:.2f} EUR)"
        ).add_to(trucks_group)

        # Starting position
        current_pos = (truck['Latitude (dropoff)'], truck['Longitude (dropoff)'])

        # Add routes
        for i, route in enumerate(route_chain):
            cargo = route['cargo']
            profit = route['profit']

            # Pickup location
            pickup_coords = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])

            # Draw route to pickup
            folium.PolyLine(
                [current_pos, pickup_coords],
                color=color,
                weight=3,
                opacity=0.7,
                tooltip=f"To pickup: {route['distance_to_pickup']:.1f} km"
            ).add_to(routes_group)

            # Add pickup marker
            pickup_popup = f"""
                <div style='min-width: 200px'>
                    <h4>Pickup Point {i + 1}</h4>
                    <b>Truck:</b> {truck['truck_id']}<br>
                    <b>Cargo Type:</b> {cargo['Cargo_Type']}<br>
                    <b>Location:</b> {cargo['Origin']}<br>
                    <b>Time:</b> {route['pickup_time'].strftime('%Y-%m-%d %H:%M')}<br>
                    <b>Premium:</b> {cargo['Premium']:.2f} EUR<br>
                    <b>Profit:</b> {profit:.2f} EUR
                </div>
            """

            folium.Marker(
                pickup_coords,
                popup=pickup_popup,
                icon=folium.Icon(color=color, icon='play', prefix='fa'),
                tooltip=f"Pickup {i + 1}: Truck {truck['truck_id']}"
            ).add_to(pickups_group)

            # Add rest stops on way to pickup
            for rest in route['rest_stops_to_pickup']:
                folium.Marker(
                    rest['location'],
                    popup=f"Rest: {rest['duration'] * 60:.0f} min",
                    icon=folium.Icon(color='red', icon='bed', prefix='fa')
                ).add_to(rest_stops_group)

            # Delivery location
            delivery_coords = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])

            # Draw route from pickup to delivery
            folium.PolyLine(
                [pickup_coords, delivery_coords],
                color=color,
                weight=3,
                opacity=0.7,
                tooltip=f"To delivery: {route['distance_to_delivery']:.1f} km"
            ).add_to(routes_group)

            # Add delivery marker
            delivery_popup = f"""
                <div style='min-width: 200px'>
                    <h4>Delivery Point {i + 1}</h4>
                    <b>Truck:</b> {truck['truck_id']}<br>
                    <b>Cargo Type:</b> {cargo['Cargo_Type']}<br>
                    <b>Location:</b> {cargo['Delivery_Location']}<br>
                    <b>Time:</b> {route['delivery_time'].strftime('%Y-%m-%d %H:%M')}<br>
                    <b>Premium:</b> {cargo['Premium']:.2f} EUR<br>
                    <b>Profit:</b> {profit:.2f} EUR
                </div>
            """

            folium.Marker(
                delivery_coords,
                popup=delivery_popup,
                icon=folium.Icon(color=color, icon='stop', prefix='fa'),
                tooltip=f"Delivery {i + 1}: Truck {truck['truck_id']}"
            ).add_to(deliveries_group)

            # Add rest stops on way to delivery
            for rest in route['rest_stops_to_delivery']:
                folium.Marker(
                    rest['location'],
                    popup=f"Rest: {rest['duration'] * 60:.0f} min",
                    icon=folium.Icon(color='red', icon='bed', prefix='fa')
                ).add_to(rest_stops_group)

            # Update current position for next route
            current_pos = delivery_coords

    # Add unassigned cargo to the map
    for idx, cargo in cargo_df.iterrows():
        if idx not in assigned_cargo_indices:
            # Create popup for unassigned cargo
            premium_text = f"<b>Premium:</b> {cargo['Premium']:.2f} EUR<br>" if 'Premium' in cargo else ""
            
            unassigned_popup = f"""
                <div style='min-width: 200px'>
                    <h4>Unassigned Cargo</h4>
                    <b>Type:</b> {cargo['Cargo_Type']}<br>
                    <b>Origin:</b> {cargo['Origin']}<br>
                    <b>Destination:</b> {cargo['Delivery_Location']}<br>
                    <b>Available:</b> {pd.to_datetime(cargo['Available_From']).strftime('%Y-%m-%d %H:%M')} to 
                                    {pd.to_datetime(cargo['Available_To']).strftime('%Y-%m-%d %H:%M')}<br>
                    {premium_text}
                </div>
            """

            # Add marker at origin
            folium.Marker(
                [cargo['Origin_Latitude'], cargo['Origin_Longitude']],
                popup=unassigned_popup,
                icon=folium.Icon(color='gray', icon='circle', prefix='fa'),
                tooltip=f"Unassigned: {cargo['Cargo_Type']}"
            ).add_to(unassigned_group)

            # Add small line to show destination
            folium.PolyLine(
                [(cargo['Origin_Latitude'], cargo['Origin_Longitude']),
                 (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])],
                color='gray',
                weight=2,
                opacity=0.5,
                tooltip=f"Unassigned route"
            ).add_to(unassigned_group)

            # Add small marker at destination
            folium.CircleMarker(
                [cargo['Delivery_Latitude'], cargo['Delivery_Longitude']],
                radius=5,
                color='gray',
                fill=True,
                fill_color='gray',
                tooltip=f"Unassigned destination: {cargo['Delivery_Location']}"
            ).add_to(unassigned_group)

    # Add all feature groups to the map
    trucks_group.add_to(m)
    pickups_group.add_to(m)
    deliveries_group.add_to(m)
    rest_stops_group.add_to(m)
    routes_group.add_to(m)
    unassigned_group.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 170px; height: 210px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding: 10px;
                border-radius: 5px;">
        <div style="font-weight: bold;">Legend</div>
        <div style="margin-top: 5px;">
            <i class="fa fa-truck" style="color:blue;"></i> Truck
        </div>
        <div>
            <i class="fa fa-play" style="color:green;"></i> Pickup
        </div>
        <div>
            <i class="fa fa-stop" style="color:red;"></i> Delivery
        </div>
        <div>
            <i class="fa fa-bed" style="color:red;"></i> Rest Stop
        </div>
        <div>
            <i class="fa fa-circle" style="color:gray;"></i> Unassigned
        </div>
        <div style="margin-top: 5px;">
            <hr style="margin: 2px;">
            <div>Lines show routes</div>
            <div>Tooltips show profit</div>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m