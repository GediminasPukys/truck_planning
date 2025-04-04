# route_planner.py
from datetime import datetime, timedelta
import pandas as pd
from geopy.distance import geodesic


class RoutePlanner:
    def __init__(self, standard_speed_kmh=73):
        self.standard_speed_kmh = standard_speed_kmh
        # Set default constraints
        self.max_distance_km = 250
        self.max_waiting_hours = 24
        # EU regulations constants
        self.DAILY_DRIVE_LIMIT = 9  # hours
        self.CONTINUOUS_DRIVE_LIMIT = 4.5  # hours
        self.BREAK_DURATION = 0.75  # 45 minutes in hours
        self.DAILY_REST_PERIOD = 11  # hours

    def calculate_travel_time(self, origin, destination):
        """Calculate travel time between two points"""
        distance = geodesic(origin, destination).kilometers
        return distance / self.standard_speed_kmh, distance

    def find_rest_point(self, start_pos, end_pos, distance_covered, total_distance):
        """Calculate rest point location as a fraction along the route"""
        fraction = distance_covered / total_distance
        rest_lat = start_pos[0] + (end_pos[0] - start_pos[0]) * fraction
        rest_lon = start_pos[1] + (end_pos[1] - start_pos[1]) * fraction
        return (rest_lat, rest_lon)

    def calculate_route_with_rests(self, start_pos, end_pos, start_time, cumulative_drive_time):
        """Calculate route including any necessary rest stops"""
        travel_time, distance = self.calculate_travel_time(start_pos, end_pos)
        current_time = start_time
        rest_stops = []
        remaining_distance = distance
        distance_covered = 0

        # Check if continuous drive limit will be exceeded
        while cumulative_drive_time + (remaining_distance / self.standard_speed_kmh) > self.CONTINUOUS_DRIVE_LIMIT:
            # Calculate distance until rest needed
            distance_until_rest = (self.CONTINUOUS_DRIVE_LIMIT - cumulative_drive_time) * self.standard_speed_kmh
            distance_covered += distance_until_rest
            remaining_distance -= distance_until_rest

            # Add rest stop
            rest_point = self.find_rest_point(start_pos, end_pos, distance_covered, distance)
            current_time += timedelta(hours=distance_until_rest / self.standard_speed_kmh + self.BREAK_DURATION)

            rest_stops.append({
                'location': rest_point,
                'time': current_time - timedelta(hours=self.BREAK_DURATION),
                'duration': self.BREAK_DURATION,
                'type': 'break'
            })

            cumulative_drive_time = 0

        # Add remaining drive time
        cumulative_drive_time += remaining_distance / self.standard_speed_kmh
        current_time += timedelta(hours=remaining_distance / self.standard_speed_kmh)

        return {
            'arrival_time': current_time,
            'rest_stops': rest_stops,
            'updated_drive_time': cumulative_drive_time,
            'total_distance': distance
        }

    def plan_route_chain(self, truck, available_cargo, current_time, last_rest_time):
        """Plan optimal chain of cargo deliveries for a truck"""
        route_chain = []
        current_pos = (truck['Latitude (dropoff)'], truck['Longitude (dropoff)'])
        current_time = pd.to_datetime(current_time)
        cumulative_drive_time = 0

        # Sort cargo by earliest available time
        available_cargo = available_cargo.sort_values('Available_From')

        while True:
            best_next_cargo = None
            best_score = float('inf')

            for _, cargo in available_cargo.iterrows():
                if cargo['Cargo_Type'].lower() != truck['truck type'].lower():
                    continue

                # Calculate distance to pickup
                _, distance_to_pickup = self.calculate_travel_time(
                    current_pos,
                    (cargo['Origin_Latitude'], cargo['Origin_Longitude'])
                )

                # ENFORCE MAXIMUM DISTANCE CHECK
                if distance_to_pickup > self.max_distance_km:
                    continue

                # Calculate route to pickup
                pickup_route = self.calculate_route_with_rests(
                    current_pos,
                    (cargo['Origin_Latitude'], cargo['Origin_Longitude']),
                    current_time,
                    cumulative_drive_time
                )

                earliest_pickup = max(
                    pickup_route['arrival_time'],
                    pd.to_datetime(cargo['Available_From'])
                )

                # Check if waiting time exceeds maximum
                waiting_time = (
                                       earliest_pickup - pickup_route['arrival_time']
                               ).total_seconds() / 3600 if earliest_pickup > pickup_route['arrival_time'] else 0

                if waiting_time > self.max_waiting_hours:
                    continue

                if earliest_pickup > pd.to_datetime(cargo['Available_To']):
                    continue

                # Calculate route from pickup to delivery
                delivery_route = self.calculate_route_with_rests(
                    (cargo['Origin_Latitude'], cargo['Origin_Longitude']),
                    (cargo['Delivery_Latitude'], cargo['Delivery_Longitude']),
                    earliest_pickup,
                    pickup_route['updated_drive_time']
                )

                # Use the distance from the route calculation
                distance_to_pickup = pickup_route['total_distance']
                distance_to_delivery = delivery_route['total_distance']
                total_distance = distance_to_pickup + distance_to_delivery

                # Calculate score
                score = waiting_time * float(truck['waiting time price per h, EUR']) + \
                        total_distance * float(truck['price per km, Eur'])

                if score < best_score:
                    best_score = score
                    best_next_cargo = {
                        'cargo': cargo,
                        'pickup_time': earliest_pickup,
                        'delivery_time': delivery_route['arrival_time'],
                        'distance_to_pickup': distance_to_pickup,
                        'distance_to_delivery': distance_to_delivery,
                        'total_distance': total_distance,
                        'waiting_time': waiting_time,
                        'score': score,
                        'rest_stops_to_pickup': pickup_route['rest_stops'],
                        'rest_stops_to_delivery': delivery_route['rest_stops']
                    }

            if best_next_cargo is None:
                break

            route_chain.append(best_next_cargo)

            # Update current position and time for next iteration
            current_pos = (
                best_next_cargo['cargo']['Delivery_Latitude'],
                best_next_cargo['cargo']['Delivery_Longitude']
            )
            current_time = best_next_cargo['delivery_time']
            cumulative_drive_time = delivery_route['updated_drive_time']

            # Remove assigned cargo from available pool
            available_cargo = available_cargo[
                available_cargo.index != best_next_cargo['cargo'].name
                ]

        return route_chain