# utils/visualization.py
import folium
import pandas as pd
from datetime import datetime
from folium import plugins


def create_map(trucks_df, cargo_df, route_chains):
    """Create an interactive map with filters and full-screen mode"""
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

    m = folium.Map(
        location=[all_lats.mean(), all_lons.mean()],
        zoom_start=6,
        tiles='cartodbpositron'
    )

    # Add fullscreen control
    plugins.Fullscreen(
        position='topleft',
        title='Expand map',
        title_cancel='Exit full screen',
        force_separate_button=True
    ).add_to(m)

    # Create feature groups for filtering
    trucks_group = folium.FeatureGroup(name='Trucks')
    pickup_routes_group = folium.FeatureGroup(name='Pickup Routes')
    delivery_routes_group = folium.FeatureGroup(name='Delivery Routes')
    pickup_points_group = folium.FeatureGroup(name='Pickup Points')
    delivery_points_group = folium.FeatureGroup(name='Delivery Points')
    rest_stops_group = folium.FeatureGroup(name='Rest Stops')

    # Add custom layer control CSS
    m.get_root().html.add_child(folium.Element("""
        <style>
            .leaflet-control-layers {
                background: #fff;
                padding: 10px;
                border-radius: 5px;
                max-height: 400px;
                overflow-y: auto;
            }
            .filter-group {
                margin: 5px 0;
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            .filter-title {
                font-weight: bold;
                margin-bottom: 5px;
            }
        </style>
    """))

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']

    for truck_idx, route_chain in route_chains.items():
        truck = trucks_df.iloc[truck_idx]
        color = colors[truck_idx % len(colors)]

        # Create truck marker
        truck_popup = f"""
            <div style='min-width: 200px'>
                <h4>Truck {truck['truck_id']}</h4>
                <b>Type:</b> {truck['truck type']}<br>
                <b>Start Location:</b> {truck['Address (drop off)']}<br>
                <b>Routes:</b> {len(route_chain)}
            </div>
        """

        # Add truck marker to trucks group
        folium.Marker(
            [truck['Latitude (dropoff)'], truck['Longitude (dropoff)']],
            popup=truck_popup,
            icon=folium.Icon(color=color, icon='truck', prefix='fa'),
            tooltip=f"Truck {truck['truck_id']}",
        ).add_to(trucks_group)

        current_pos = (truck['Latitude (dropoff)'], truck['Longitude (dropoff)'])

        for i, route in enumerate(route_chain):
            cargo = route['cargo']

            # Draw route to pickup with rest stops
            pickup_coords = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])
            last_pos = current_pos

            # Add rest stops on way to pickup
            for rest in route['rest_stops_to_pickup']:
                # Draw route segment to rest stop
                folium.PolyLine(
                    locations=[last_pos, rest['location']],
                    weight=2,
                    color=color,
                    opacity=0.8,
                    tooltip=f"To rest stop: Truck {truck['truck_id']}"
                ).add_to(pickup_routes_group)

                # Add rest stop marker
                rest_popup = f"""
                    <div style='min-width: 150px'>
                        <h4>Rest Stop</h4>
                        <b>Truck:</b> {truck['truck_id']}<br>
                        <b>Time:</b> {rest['time']}<br>
                        <b>Duration:</b> {rest['duration']}h<br>
                        <b>Type:</b> {rest['type']}
                    </div>
                """

                folium.Marker(
                    rest['location'],
                    popup=rest_popup,
                    icon=folium.Icon(color='red', icon='bed', prefix='fa'),
                    tooltip=f"Rest: {rest['duration']}h"
                ).add_to(rest_stops_group)

                last_pos = rest['location']

            # Complete route to pickup
            folium.PolyLine(
                locations=[last_pos, pickup_coords],
                weight=2,
                color=color,
                opacity=0.8,
                tooltip=f"To pickup: Truck {truck['truck_id']}"
            ).add_to(pickup_routes_group)

            # Add pickup marker
            pickup_popup = f"""
                <div style='min-width: 200px'>
                    <h4>Pickup Point {i + 1}</h4>
                    <b>Truck:</b> {truck['truck_id']}<br>
                    <b>Cargo Type:</b> {cargo['Cargo_Type']}<br>
                    <b>Location:</b> {cargo['Origin']}<br>
                    <b>Time:</b> {route['pickup_time']}
                </div>
            """

            folium.Marker(
                pickup_coords,
                popup=pickup_popup,
                icon=folium.Icon(color=color, icon='play', prefix='fa'),
                tooltip=f"Pickup {i + 1}: Truck {truck['truck_id']}"
            ).add_to(pickup_points_group)

            # Draw delivery route
            last_pos = pickup_coords
            delivery_coords = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])

            # Add rest stops on way to delivery
            for rest in route['rest_stops_to_delivery']:
                folium.PolyLine(
                    locations=[last_pos, rest['location']],
                    weight=2,
                    color=color,
                    opacity=0.8,
                    tooltip=f"To rest stop: Truck {truck['truck_id']}"
                ).add_to(delivery_routes_group)

                rest_popup = f"""
                    <div style='min-width: 150px'>
                        <h4>Rest Stop</h4>
                        <b>Truck:</b> {truck['truck_id']}<br>
                        <b>Time:</b> {rest['time']}<br>
                        <b>Duration:</b> {rest['duration']}h<br>
                        <b>Type:</b> {rest['type']}
                    </div>
                """

                folium.Marker(
                    rest['location'],
                    popup=rest_popup,
                    icon=folium.Icon(color='red', icon='bed', prefix='fa'),
                    tooltip=f"Rest: {rest['duration']}h"
                ).add_to(rest_stops_group)

                last_pos = rest['location']

            # Complete route to delivery
            folium.PolyLine(
                locations=[last_pos, delivery_coords],
                weight=2,
                color=color,
                opacity=0.8,
                tooltip=f"To delivery: Truck {truck['truck_id']}"
            ).add_to(delivery_routes_group)

            # Add delivery marker
            delivery_popup = f"""
                <div style='min-width: 200px'>
                    <h4>Delivery Point {i + 1}</h4>
                    <b>Truck:</b> {truck['truck_id']}<br>
                    <b>Cargo Type:</b> {cargo['Cargo_Type']}<br>
                    <b>Location:</b> {cargo['Delivery_Location']}<br>
                    <b>Time:</b> {route['delivery_time']}
                </div>
            """

            folium.Marker(
                delivery_coords,
                popup=delivery_popup,
                icon=folium.Icon(color=color, icon='stop', prefix='fa'),
                tooltip=f"Delivery {i + 1}: Truck {truck['truck_id']}"
            ).add_to(delivery_points_group)

            current_pos = delivery_coords

    # Add all feature groups to map
    trucks_group.add_to(m)
    pickup_routes_group.add_to(m)
    delivery_routes_group.add_to(m)
    pickup_points_group.add_to(m)
    delivery_points_group.add_to(m)
    rest_stops_group.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Add search functionality
    search_trucks = list(trucks_df['truck_id'].unique())
    search_cargo = list(cargo_df['Cargo_Type'].unique())

    search_js = f"""
        <script>
        var searchTrucks = {search_trucks};
        var searchCargo = {search_cargo};

        function filterMap() {{
            var truckInput = document.getElementById('truckFilter').value.toLowerCase();
            var cargoInput = document.getElementById('cargoFilter').value.toLowerCase();

            // Filter markers and routes based on input
            // Implementation would go here
        }}
        </script>
    """

    # Add custom filter controls
    filter_html = f"""
        <div class='leaflet-control leaflet-control-layers' style='position: absolute; top: 10px; right: 10px;'>
            <div class='filter-group'>
                <div class='filter-title'>🚚 Filter by Truck</div>
                <select id='truckFilter' onchange='filterMap()'>
                    <option value=''>All Trucks</option>
                    {"".join(f"<option value='{t}'>{t}</option>" for t in search_trucks)}
                </select>
            </div>
            <div class='filter-group'>
                <div class='filter-title'>📦 Filter by Cargo</div>
                <select id='cargoFilter' onchange='filterMap()'>
                    <option value=''>All Cargo</option>
                    {"".join(f"<option value='{c}'>{c}</option>" for c in search_cargo)}
                </select>
            </div>
        </div>
    """

    m.get_root().html.add_child(folium.Element(search_js + filter_html))

    return m