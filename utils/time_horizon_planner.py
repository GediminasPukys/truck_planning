# utils/time_horizon_planner.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from geopy.distance import geodesic
from typing import Dict, List, Tuple, Optional
import math


class TimeHorizonPlanner:
    """Class to plan operations across extended time horizons"""

    def __init__(self, planning_horizon_days=30, window_size_days=7):
        self.planning_horizon_days = planning_horizon_days
        self.window_size_days = window_size_days

    def analyze_cargo_time_distribution(self, cargo_df):
        """Analyze how cargo is distributed across time"""
        try:
            cargo_df['Available_From'] = pd.to_datetime(cargo_df['Available_From'])
            cargo_df['Available_To'] = pd.to_datetime(cargo_df['Available_To'])

            analysis = {
                'earliest_cargo': cargo_df['Available_From'].min(),
                'latest_cargo': cargo_df['Available_To'].max(),
                'total_span_days': (cargo_df['Available_To'].max() - cargo_df['Available_From'].min()).days,
                'cargo_per_day': {},
                'cargo_per_week': {},
                'peak_periods': [],
                'low_periods': []
            }

            # Daily distribution
            cargo_df['date'] = cargo_df['Available_From'].dt.date
            daily_counts = cargo_df.groupby('date').size()
            analysis['cargo_per_day'] = daily_counts.to_dict()

            # Weekly distribution
            cargo_df['week'] = cargo_df['Available_From'].dt.isocalendar().week
            weekly_counts = cargo_df.groupby('week').size()
            analysis['cargo_per_week'] = weekly_counts.to_dict()

            # Identify peak and low periods
            daily_mean = daily_counts.mean()
            daily_std = daily_counts.std()

            peak_threshold = daily_mean + daily_std
            low_threshold = daily_mean - daily_std

            for date, count in daily_counts.items():
                if count > peak_threshold:
                    analysis['peak_periods'].append({'date': date, 'count': count})
                elif count < low_threshold:
                    analysis['low_periods'].append({'date': date, 'count': count})

            return analysis

        except Exception as e:
            print(f"Error analyzing cargo time distribution: {str(e)}")
            return {}

    def create_planning_windows(self, start_date, end_date, window_size_days=None):
        """Create time windows for planning"""
        if window_size_days is None:
            window_size_days = self.window_size_days

        windows = []
        current_start = start_date
        window_id = 0

        while current_start < end_date:
            window_end = min(current_start + timedelta(days=window_size_days), end_date)

            windows.append({
                'id': window_id,
                'start': current_start,
                'end': window_end,
                'duration_days': (window_end - current_start).days,
                'cargo_count': 0,
                'planned_assignments': [],
                'utilization_score': 0
            })

            current_start = window_end
            window_id += 1

        return windows

    def assign_cargo_to_windows(self, cargo_df, windows):
        """Assign cargo items to appropriate time windows"""
        try:
            cargo_df['Available_From'] = pd.to_datetime(cargo_df['Available_From'])
            cargo_df['Available_To'] = pd.to_datetime(cargo_df['Available_To'])

            for window in windows:
                # Find cargo that can be picked up in this window
                window_cargo = cargo_df[
                    (cargo_df['Available_From'] <= window['end']) &
                    (cargo_df['Available_To'] >= window['start'])
                    ]

                window['cargo_count'] = len(window_cargo)
                window['cargo_indices'] = list(window_cargo.index)

                # Calculate utilization score based on cargo density
                if window['duration_days'] > 0:
                    window['utilization_score'] = len(window_cargo) / window['duration_days']

            return windows

        except Exception as e:
            print(f"Error assigning cargo to windows: {str(e)}")
            return windows

    def optimize_window_sequencing(self, windows):
        """Optimize the sequence of planning windows based on cargo density and dependencies"""
        # Sort windows by utilization score (descending) and then by start time
        sorted_windows = sorted(windows, key=lambda w: (-w['utilization_score'], w['start']))

        # Re-assign IDs to maintain chronological order for dependencies
        for i, window in enumerate(sorted(windows, key=lambda w: w['start'])):
            window['planning_priority'] = next(j for j, w in enumerate(sorted_windows) if w['id'] == window['id'])

        return sorted_windows


class MultiDayRoutePlanner:
    """Class to handle multi-day route planning with rest periods"""

    def __init__(self, daily_drive_limit=9, continuous_drive_limit=4.5,
                 break_duration=0.75, daily_rest_hours=11, standard_speed_kmh=73):
        self.daily_drive_limit = daily_drive_limit
        self.continuous_drive_limit = continuous_drive_limit
        self.break_duration = break_duration
        self.daily_rest_hours = daily_rest_hours
        self.standard_speed_kmh = standard_speed_kmh

    def calculate_rest_requirements(self, current_time, cumulative_drive_time, daily_drive_time):
        """Calculate what rest is needed before continuing"""
        required_rests = []

        # Check if continuous drive limit exceeded
        if cumulative_drive_time >= self.continuous_drive_limit:
            required_rests.append({
                'type': 'break',
                'duration_hours': self.break_duration,
                'reason': 'Continuous drive limit reached'
            })

        # Check if daily drive limit exceeded or if it's late
        current_hour = current_time.hour
        if daily_drive_time >= self.daily_drive_limit or current_hour >= 20:
            required_rests.append({
                'type': 'daily_rest',
                'duration_hours': self.daily_rest_hours,
                'reason': 'Daily drive limit or late hour'
            })

        return required_rests

    def plan_rest_stops(self, current_pos, destination_pos, current_time,
                        cumulative_drive_time, daily_drive_time):
        """Plan necessary rest stops for a route"""
        distance = geodesic(current_pos, destination_pos).kilometers
        travel_time_hours = distance / self.standard_speed_kmh

        rest_stops = []
        current_drive_time = cumulative_drive_time
        current_daily_time = daily_drive_time
        remaining_distance = distance
        covered_distance = 0

        while remaining_distance > 0:
            # Calculate how far we can drive before needing rest
            time_until_break = max(0, self.continuous_drive_limit - current_drive_time)
            time_until_daily_limit = max(0, self.daily_drive_limit - current_daily_time)

            # Take the more restrictive limit
            max_drive_time = min(time_until_break, time_until_daily_limit)

            if max_drive_time <= 0:
                # Need immediate rest
                required_rests = self.calculate_rest_requirements(
                    current_time, current_drive_time, current_daily_time
                )

                for rest in required_rests:
                    # Find rest location (interpolate along route)
                    rest_fraction = covered_distance / distance if distance > 0 else 0
                    rest_location = self.interpolate_position(
                        current_pos, destination_pos, rest_fraction
                    )

                    rest_stops.append({
                        'location': rest_location,
                        'time': current_time,
                        'duration': rest['duration_hours'],
                        'type': rest['type'],
                        'reason': rest['reason']
                    })

                    # Update time and reset counters
                    current_time += timedelta(hours=rest['duration_hours'])
                    if rest['type'] == 'daily_rest':
                        current_daily_time = 0
                        current_drive_time = 0
                    elif rest['type'] == 'break':
                        current_drive_time = 0

            # Calculate how much we can drive in the remaining time
            max_distance = max_drive_time * self.standard_speed_kmh
            drive_distance = min(remaining_distance, max_distance)

            if drive_distance > 0:
                drive_time = drive_distance / self.standard_speed_kmh

                # Update positions and times
                covered_distance += drive_distance
                remaining_distance -= drive_distance
                current_time += timedelta(hours=drive_time)
                current_drive_time += drive_time
                current_daily_time += drive_time

        return rest_stops, current_time, current_drive_time, current_daily_time

    def interpolate_position(self, start_pos, end_pos, fraction):
        """Interpolate position along a route"""
        lat = start_pos[0] + (end_pos[0] - start_pos[0]) * fraction
        lon = start_pos[1] + (end_pos[1] - start_pos[1]) * fraction
        return (lat, lon)

    def plan_extended_route(self, truck_state, cargo_list, max_days=7):
        """Plan an extended route for a truck across multiple days"""
        route_plan = {
            'truck_id': truck_state['truck']['truck_id'],
            'routes': [],
            'total_distance': 0,
            'total_profit': 0,
            'total_days': 0,
            'rest_stops': []
        }

        current_pos = truck_state['current_pos']
        current_time = truck_state['current_time']
        cumulative_drive_time = truck_state.get('cumulative_drive_time', 0)
        daily_drive_time = truck_state.get('daily_drive_time', 0)

        start_time = current_time

        for cargo in cargo_list:
            # Check if we've exceeded maximum planning days
            if (current_time - start_time).days >= max_days:
                break

            # Plan route to pickup
            pickup_pos = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])
            pickup_rest_stops, pickup_arrival_time, updated_drive_time, updated_daily_time = \
                self.plan_rest_stops(current_pos, pickup_pos, current_time,
                                     cumulative_drive_time, daily_drive_time)

            # Add pickup rest stops
            route_plan['rest_stops'].extend(pickup_rest_stops)

            # Plan route to delivery
            delivery_pos = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])
            delivery_rest_stops, delivery_arrival_time, final_drive_time, final_daily_time = \
                self.plan_rest_stops(pickup_pos, delivery_pos, pickup_arrival_time,
                                     updated_drive_time, updated_daily_time)

            # Add delivery rest stops
            route_plan['rest_stops'].extend(delivery_rest_stops)

            # Calculate distances
            distance_to_pickup = geodesic(current_pos, pickup_pos).kilometers
            distance_to_delivery = geodesic(pickup_pos, delivery_pos).kilometers
            total_route_distance = distance_to_pickup + distance_to_delivery

            # Add route to plan
            route_info = {
                'cargo': cargo,
                'pickup_time': pickup_arrival_time,
                'delivery_time': delivery_arrival_time,
                'distance_to_pickup': distance_to_pickup,
                'distance_to_delivery': distance_to_delivery,
                'total_distance': total_route_distance,
                'rest_stops_to_pickup': pickup_rest_stops,
                'rest_stops_to_delivery': delivery_rest_stops
            }

            route_plan['routes'].append(route_info)
            route_plan['total_distance'] += total_route_distance

            # Update current state
            current_pos = delivery_pos
            current_time = delivery_arrival_time
            cumulative_drive_time = final_drive_time
            daily_drive_time = final_daily_time

        route_plan['total_days'] = (current_time - start_time).days

        return route_plan

    def validate_route_legality(self, route_plan):
        """Validate that a route plan complies with driving regulations"""
        violations = []

        for i, route in enumerate(route_plan['routes']):
            # Check if daily limits are respected
            pickup_time = route['pickup_time']
            delivery_time = route['delivery_time']

            # Simple check: ensure no single day has excessive driving
            same_day_routes = [r for r in route_plan['routes']
                               if r['pickup_time'].date() == pickup_time.date()]

            daily_distance = sum(r['total_distance'] for r in same_day_routes)
            daily_drive_time = daily_distance / self.standard_speed_kmh

            if daily_drive_time > self.daily_drive_limit:
                violations.append({
                    'type': 'daily_drive_limit',
                    'date': pickup_time.date(),
                    'actual_hours': daily_drive_time,
                    'limit_hours': self.daily_drive_limit
                })

        return violations


class RestLocationOptimizer:
    """Class to optimize rest stop locations"""

    def __init__(self):
        self.preferred_rest_locations = [
            # This could be populated with known truck stops, rest areas, etc.
            # For now, we'll use interpolated positions
        ]

    def find_optimal_rest_location(self, start_pos, end_pos, target_distance_fraction):
        """Find optimal rest location along a route"""
        # For now, simple interpolation
        # In practice, this could consider:
        # - Truck stops and rest areas
        # - Fuel stations
        # - Safe parking areas
        # - Local regulations

        lat = start_pos[0] + (end_pos[0] - start_pos[0]) * target_distance_fraction
        lon = start_pos[1] + (end_pos[1] - start_pos[1]) * target_distance_fraction

        return (lat, lon)

    def evaluate_rest_location(self, location, criteria=None):
        """Evaluate the quality of a rest location"""
        # This could consider factors like:
        # - Safety
        # - Facilities available
        # - Cost
        # - Regulatory compliance

        # For now, return a basic score
        return {
            'safety_score': 0.8,
            'facilities_score': 0.7,
            'cost_score': 0.9,
            'overall_score': 0.8
        }

