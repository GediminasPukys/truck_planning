# utils/profit_optimizer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
from typing import Dict, List, Tuple, Set, Optional


class ProfitCalculator:
    """Class to calculate profit based on cargo premium and route costs"""

    def __init__(self, price_per_km=0.39, price_per_hour=10, standard_speed_kmh=73):
        self.price_per_km = price_per_km
        self.price_per_hour = price_per_hour
        self.standard_speed_kmh = standard_speed_kmh

    def calculate_travel_time(self, distance_km):
        """Calculate travel time in hours based on distance"""
        return distance_km / self.standard_speed_kmh

    def calculate_profit(self, cargo, truck, route_info):
        """
        Calculate profit for a cargo delivery

        Args:
            cargo: Cargo data (contains Premium)
            truck: Truck data (contains costs)
            route_info: Dictionary with route details

        Returns:
            profit: Calculated profit value
        """
        # Get premium from cargo
        try:
            premium = float(cargo['Premium'])
            if pd.isna(premium) or premium <= 0:
                premium = 0
                print(f"Warning: Invalid premium value {cargo.get('Premium')} for cargo. Using 0 instead.")
        except (KeyError, ValueError, TypeError) as e:
            premium = 0
            print(f"Warning: Error getting premium value: {str(e)}. Using 0 instead.")

        # Calculate distance cost
        try:
            total_distance = float(route_info['total_distance'])
            distance_cost = total_distance * self.price_per_km
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error calculating distance cost: {str(e)}")
            distance_cost = 0

        # Calculate time cost (pickup + waiting + delivery time)
        try:
            pickup_time = float(route_info.get('travel_to_cargo_hours', 0))
            delivery_time = float(route_info.get('travel_to_delivery_hours', 0))
            waiting_time = float(route_info.get('waiting_hours', 0))

            # Note: time_cost is SUBTRACTED from the overall cost per requirements
            time_cost = (pickup_time + waiting_time + delivery_time) * self.price_per_hour
        except (ValueError, TypeError) as e:
            print(f"Error calculating time cost: {str(e)}")
            time_cost = 0

        # Total cost is distance_cost MINUS time_cost
        total_cost = distance_cost - time_cost

        # Profit is premium minus total_cost
        profit = premium - total_cost

        # Sanity check for unusually high/low profit values
        if profit > 10000:
            print(f"Warning: Unusually high profit: {profit:.2f} EUR. Premium: {premium:.2f}, Cost: {total_cost:.2f}")
        elif profit < -10000:
            print(f"Warning: Unusually low profit: {profit:.2f} EUR. Premium: {premium:.2f}, Cost: {total_cost:.2f}")

        # Detailed debug output to help diagnose issues
        if profit <= 0:
            print(f"Unprofitable delivery detected:")
            print(f"  Premium: {premium:.2f} EUR")
            print(f"  Distance: {total_distance:.2f} km at {self.price_per_km:.2f} EUR/km = {distance_cost:.2f} EUR")
            print(
                f"  Time: {(pickup_time + waiting_time + delivery_time):.2f} h at {self.price_per_hour:.2f} EUR/h = {-time_cost:.2f} EUR (credit)")
            print(f"  Total cost: {total_cost:.2f} EUR")
            print(f"  Profit: {profit:.2f} EUR")

        # Store all calculations for reference
        profit_breakdown = {
            'premium': premium,
            'distance_cost': distance_cost,
            'time_cost': -time_cost,  # Negative to show it reduces cost
            'total_cost': total_cost,
            'profit': profit
        }

        return profit, profit_breakdown


class FleetProfitOptimizer:
    """Class to optimize fleet profit over an operation period"""

    def __init__(self, max_distance_km=250, max_waiting_hours=24,
                 price_per_km=0.39, price_per_hour=10, standard_speed_kmh=73):
        self.max_distance_km = max_distance_km
        self.max_waiting_hours = max_waiting_hours
        self.profit_calculator = ProfitCalculator(
            price_per_km=price_per_km,
            price_per_hour=price_per_hour,
            standard_speed_kmh=standard_speed_kmh
        )
        self.standard_speed_kmh = standard_speed_kmh

        # EU regulations constants for rest periods
        self.DAILY_DRIVE_LIMIT = 9  # hours
        self.CONTINUOUS_DRIVE_LIMIT = 4.5  # hours
        self.BREAK_DURATION = 0.75  # 45 minutes in hours
        self.DAILY_REST_PERIOD = 11  # hours

    def find_rest_point(self, start_pos, end_pos, distance_covered, total_distance):
        """Calculate rest point location as a fraction along the route"""
        fraction = distance_covered / total_distance
        rest_lat = start_pos[0] + (end_pos[0] - start_pos[0]) * fraction
        rest_lon = start_pos[1] + (end_pos[1] - start_pos[1]) * fraction
        return (rest_lat, rest_lon)

    def calculate_route_with_rests(self, start_pos, end_pos, start_time, cumulative_drive_time):
        """Calculate route including any necessary rest stops"""
        # Calculate distance and travel time
        distance = geodesic(start_pos, end_pos).kilometers
        travel_time = distance / self.standard_speed_kmh

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
            'total_distance': distance,
            'travel_time_hours': distance / self.standard_speed_kmh
        }

    def evaluate_cargo_assignment(self, truck_info, cargo, operation_end_time):
        """
        Evaluate if a cargo assignment is feasible and calculate profit

        Args:
            truck_info: Dictionary with truck data and current status
            cargo: Cargo data row
            operation_end_time: Global end time for operations

        Returns:
            (profit, feasible, route_details, rejection_reason): Profit value, feasibility, and details
        """
        # Extract truck and status info
        truck = truck_info['truck']
        current_pos = truck_info['current_pos']
        current_time = truck_info['current_time']
        cumulative_drive_time = truck_info.get('cumulative_drive_time', 0)

        rejection_reason = None

        # Check if truck type matches cargo type (case-insensitive)
        if truck['truck type'].lower() != cargo['Cargo_Type'].lower():
            rejection_reason = f"Type mismatch: Truck type '{truck['truck type']}' doesn't match cargo type '{cargo['Cargo_Type']}'"
            return -float('inf'), False, {}, rejection_reason

        # Calculate distance to pickup
        pickup_pos = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])
        distance_to_pickup = geodesic(current_pos, pickup_pos).kilometers

        # Check if distance exceeds maximum allowed
        if distance_to_pickup > self.max_distance_km:
            rejection_reason = f"Distance to pickup ({distance_to_pickup:.1f} km) exceeds maximum allowed ({self.max_distance_km} km)"
            return -float('inf'), False, {}, rejection_reason

        # Calculate route to pickup (including rest stops)
        pickup_route = self.calculate_route_with_rests(
            current_pos,
            pickup_pos,
            current_time,
            cumulative_drive_time
        )

        # Earliest arrival time at pickup
        earliest_pickup_time = pickup_route['arrival_time']

        # Check if cargo pickup window is compatible
        cargo_available_from = pd.to_datetime(cargo['Available_From'])
        cargo_available_to = pd.to_datetime(cargo['Available_To'])

        # If truck arrives before cargo is available, calculate waiting time
        if earliest_pickup_time < cargo_available_from:
            waiting_hours = (cargo_available_from - earliest_pickup_time).total_seconds() / 3600
            actual_pickup_time = cargo_available_from

            # Check if waiting time exceeds maximum allowed
            if waiting_hours > self.max_waiting_hours:
                rejection_reason = f"Waiting time ({waiting_hours:.1f} h) exceeds maximum allowed ({self.max_waiting_hours} h)"
                return -float('inf'), False, {}, rejection_reason
        else:
            waiting_hours = 0
            actual_pickup_time = earliest_pickup_time

            # Check if truck arrives after cargo available window ends
            if earliest_pickup_time > cargo_available_to:
                rejection_reason = f"Truck arrives at {earliest_pickup_time.strftime('%Y-%m-%d %H:%M')} after cargo window ends ({cargo_available_to.strftime('%Y-%m-%d %H:%M')})"
                return -float('inf'), False, {}, rejection_reason

        # Now calculate delivery route
        delivery_pos = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])

        # Route from pickup to delivery (including rest stops)
        delivery_route = self.calculate_route_with_rests(
            pickup_pos,
            delivery_pos,
            actual_pickup_time,
            pickup_route['updated_drive_time']
        )

        # Check if delivery will complete before operation end time
        delivery_time = delivery_route['arrival_time']
        if delivery_time > operation_end_time:
            rejection_reason = f"Delivery time ({delivery_time.strftime('%Y-%m-%d %H:%M')}) is after operation end time ({operation_end_time.strftime('%Y-%m-%d %H:%M')})"
            return -float('inf'), False, {}, rejection_reason

        # Calculate profit
        route_info = {
            'distance_to_pickup': distance_to_pickup,
            'distance_to_delivery': delivery_route['total_distance'],
            'total_distance': distance_to_pickup + delivery_route['total_distance'],
            'travel_to_cargo_hours': pickup_route['travel_time_hours'],
            'travel_to_delivery_hours': delivery_route['travel_time_hours'],
            'waiting_hours': waiting_hours,
            'pickup_time': actual_pickup_time,
            'delivery_time': delivery_time,
            'rest_stops_to_pickup': pickup_route['rest_stops'],
            'rest_stops_to_delivery': delivery_route['rest_stops']
        }

        try:
            profit, profit_breakdown = self.profit_calculator.calculate_profit(cargo, truck, route_info)

            # Check if profit is negative (unprofitable assignment)
            if profit <= 0:
                rejection_reason = f"Unprofitable assignment: profit ({profit:.2f}) is <= 0"
                return profit, False, {}, rejection_reason

        except Exception as e:
            rejection_reason = f"Error calculating profit: {str(e)}"
            return -float('inf'), False, {}, rejection_reason

        # Return complete information
        route_details = {
            **route_info,
            'profit': profit,
            'profit_breakdown': profit_breakdown,
            'cumulative_drive_time': delivery_route['updated_drive_time']
        }

        return profit, True, route_details, None

    def optimize_fleet_profit(self, trucks_df, cargo_df, operation_end_time):
        """
        Optimize entire fleet to maximize profit within operating period

        Args:
            trucks_df: DataFrame with truck data
            cargo_df: DataFrame with cargo data
            operation_end_time: Global end time for operations

        Returns:
            (route_chains, total_profit, rejection_stats): Optimized routes, total profit, and rejection statistics
        """
        print(f"Starting fleet profit optimization with {len(trucks_df)} trucks and {len(cargo_df)} cargo items")
        print(f"Operation end time: {operation_end_time}")

        # Check if Premium column exists in cargo_df
        if 'Premium' not in cargo_df.columns:
            print("WARNING: No Premium column found in cargo data. All profit calculations will be negative.")
        else:
            premium_stats = cargo_df['Premium'].describe()
            print(
                f"Premium statistics: min={premium_stats['min']}, max={premium_stats['max']}, mean={premium_stats['mean']}")

        # Initialize tracking variables
        total_profit = 0
        assigned_cargo = set()
        route_chains = {}
        rejection_stats = {}

        # Create a list of available trucks
        available_trucks = [
            {
                'idx': idx,
                'truck': truck,
                'current_pos': (truck['Latitude (dropoff)'], truck['Longitude (dropoff)']),
                'current_time': pd.to_datetime(truck['Timestamp (dropoff)']),
                'cumulative_drive_time': 0,
                'routes': []
            }
            for idx, truck in trucks_df.iterrows()
        ]

        # Initialize rejection statistics for each cargo
        for cargo_idx in cargo_df.index:
            rejection_stats[cargo_idx] = {
                'cargo': cargo_df.loc[cargo_idx],
                'reasons': [],
                'attempted_trucks': []
            }

        # Print truck types and cargo types
        truck_types = trucks_df['truck type'].str.lower().value_counts().to_dict()
        cargo_types = cargo_df['Cargo_Type'].str.lower().value_counts().to_dict()

        print(f"Truck types: {truck_types}")
        print(f"Cargo types: {cargo_types}")

        # Check for type compatibility
        truck_type_set = set(truck_types.keys())
        cargo_type_set = set(cargo_types.keys())
        matching_types = truck_type_set.intersection(cargo_type_set)

        if len(matching_types) == 0:
            print("ERROR: No matching truck and cargo types found!")
            print(f"Truck types: {truck_type_set}")
            print(f"Cargo types: {cargo_type_set}")
            return {}, 0, rejection_stats, {
                'total_rejected': len(cargo_df),
                'total_cargo': len(cargo_df),
                'total_assigned': 0,
                'assignment_rate': 0,
                'by_reason': {'Type mismatch': len(cargo_df)},
                'cargo_with_no_truck': list(cargo_df.index)
            }

        # Main assignment loop - continue until no more profitable assignments possible
        iteration = 0
        max_iterations = len(cargo_df) * 10  # Safety to prevent infinite loops

        while available_trucks and len(assigned_cargo) < len(cargo_df) and iteration < max_iterations:
            iteration += 1

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: {len(assigned_cargo)}/{len(cargo_df)} cargo items assigned")

            best_assignment = None
            best_profit = -float('inf')
            best_truck_idx = -1
            best_cargo_idx = -1

            # For each available truck, find the most profitable cargo
            for truck_idx, truck_info in enumerate(available_trucks):
                for cargo_idx, cargo in cargo_df.iterrows():
                    if cargo_idx in assigned_cargo:
                        continue

                    # Calculate profit & check constraints
                    profit, feasible, route_details, rejection_reason = self.evaluate_cargo_assignment(
                        truck_info,
                        cargo,
                        operation_end_time
                    )

                    # Record rejection reason for debugging
                    if not feasible and rejection_reason:
                        truck_id = truck_info['truck']['truck_id']
                        rejection_stats[cargo_idx]['reasons'].append({
                            'truck_id': truck_id,
                            'reason': rejection_reason,
                            'profit': profit
                        })
                        rejection_stats[cargo_idx]['attempted_trucks'].append(truck_id)

                    if feasible and profit > best_profit:
                        best_profit = profit
                        best_assignment = (truck_idx, cargo_idx, route_details)
                        best_truck_idx = truck_idx
                        best_cargo_idx = cargo_idx

            # If found profitable assignment, update truck status and continue
            if best_assignment:
                truck_idx, cargo_idx, route_details = best_assignment
                truck_info = available_trucks[truck_idx]
                cargo = cargo_df.loc[cargo_idx]

                # Update truck position, time, and cumulative drive time
                truck_info['current_pos'] = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])
                truck_info['current_time'] = route_details['delivery_time']
                truck_info['cumulative_drive_time'] = route_details['cumulative_drive_time']

                # Record assignment
                assigned_cargo.add(cargo_idx)
                total_profit += best_profit

                print(
                    f"Assigned cargo {cargo_idx} to truck {truck_info['truck']['truck_id']} with profit {best_profit:.2f} EUR")

                # Add to route chains (for visualization and output)
                if truck_info['idx'] not in route_chains:
                    route_chains[truck_info['idx']] = []

                route_chains[truck_info['idx']].append({
                    'cargo': cargo,
                    'profit': best_profit,
                    'profit_breakdown': route_details['profit_breakdown'],
                    'pickup_time': route_details['pickup_time'],
                    'delivery_time': route_details['delivery_time'],
                    'distance_to_pickup': route_details['distance_to_pickup'],
                    'distance_to_delivery': route_details['distance_to_delivery'],
                    'total_distance': route_details['total_distance'],
                    'waiting_time': route_details['waiting_hours'],
                    'rest_stops_to_pickup': route_details['rest_stops_to_pickup'],
                    'rest_stops_to_delivery': route_details['rest_stops_to_delivery']
                })
            else:
                print(f"No more profitable assignments found after {iteration} iterations")
                # No more profitable assignments possible
                break

        print(f"Optimization completed with {len(assigned_cargo)}/{len(cargo_df)} cargo items assigned")
        print(f"Total profit: {total_profit:.2f} EUR")

        # Calculate rejection statistics summary
        rejection_summary = {
            'total_rejected': len(cargo_df) - len(assigned_cargo),
            'total_cargo': len(cargo_df),
            'total_assigned': len(assigned_cargo),
            'assignment_rate': len(assigned_cargo) / len(cargo_df) if len(cargo_df) > 0 else 0,
            'by_reason': {},
            'cargo_with_no_truck': []
        }

        # Count reasons
        for cargo_idx, stats in rejection_stats.items():
            if cargo_idx not in assigned_cargo:
                # Count the most common rejection reason for this cargo
                if stats['reasons']:
                    most_common_reason = max(set(r['reason'] for r in stats['reasons']),
                                             key=[r['reason'] for r in stats['reasons']].count)
                    rejection_summary['by_reason'][most_common_reason] = \
                        rejection_summary['by_reason'].get(most_common_reason, 0) + 1
                else:
                    # No truck attempted this cargo
                    rejection_summary['cargo_with_no_truck'].append(cargo_idx)

        # Print rejection summary
        print(f"Rejection summary:")
        print(f"  Total rejected: {rejection_summary['total_rejected']}")
        print(f"  Assignment rate: {rejection_summary['assignment_rate'] * 100:.1f}%")

        if rejection_summary['by_reason']:
            print(f"  Rejection reasons:")
            for reason, count in rejection_summary['by_reason'].items():
                print(f"    {reason}: {count}")

        return route_chains, total_profit, rejection_stats, rejection_summary