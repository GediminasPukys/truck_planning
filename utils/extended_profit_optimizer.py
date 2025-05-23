# utils/extended_profit_optimizer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
from typing import Dict, List, Tuple, Set, Optional
from utils.profit_optimizer import ProfitCalculator, FleetProfitOptimizer


class ExtendedFleetProfitOptimizer(FleetProfitOptimizer):
    """Extended Fleet Profit Optimizer with multi-day planning capabilities"""

    def __init__(self, max_distance_km=250, max_waiting_hours=24,
                 price_per_km=0.39, price_per_hour=10, standard_speed_kmh=73,
                 max_planning_days=60, planning_window_days=7):

        super().__init__(max_distance_km, max_waiting_hours, price_per_km, price_per_hour, standard_speed_kmh)

        self.max_planning_days = max_planning_days
        self.planning_window_days = planning_window_days

        print(f"Extended optimizer initialized:")
        print(f"  Max planning horizon: {max_planning_days} days")
        print(f"  Planning window size: {planning_window_days} days")

    def calculate_operation_period(self, trucks_df, cargo_df):
        """Calculate optimal operation period based on truck availability and cargo windows"""
        try:
            # Get truck availability times
            trucks_df['Timestamp (dropoff)'] = pd.to_datetime(trucks_df['Timestamp (dropoff)'])
            earliest_truck = trucks_df['Timestamp (dropoff)'].min()

            # Get cargo time windows
            cargo_df['Available_From'] = pd.to_datetime(cargo_df['Available_From'])
            cargo_df['Available_To'] = pd.to_datetime(cargo_df['Available_To'])
            earliest_cargo = cargo_df['Available_From'].min()
            latest_cargo = cargo_df['Available_To'].max()

            # Operation starts when first truck is available or first cargo is available
            operation_start = max(earliest_truck, earliest_cargo - timedelta(days=1))

            # Operation ends when last cargo window closes + buffer for delivery
            operation_end = latest_cargo + timedelta(days=3)

            # Limit to maximum planning horizon
            max_end = operation_start + timedelta(days=self.max_planning_days)
            operation_end = min(operation_end, max_end)

            print(f"Calculated operation period:")
            print(f"  Start: {operation_start.strftime('%Y-%m-%d %H:%M')}")
            print(f"  End: {operation_end.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Duration: {(operation_end - operation_start).days} days")

            return operation_start, operation_end

        except Exception as e:
            print(f"Error calculating operation period: {str(e)}")
            # Fallback to default period
            now = datetime.now()
            return now, now + timedelta(days=30)

    def create_time_windows(self, operation_start, operation_end):
        """Divide operation period into manageable time windows"""
        windows = []
        current = operation_start
        window_id = 0

        while current < operation_end:
            window_end = min(current + timedelta(days=self.planning_window_days), operation_end)

            windows.append({
                'id': window_id,
                'start': current,
                'end': window_end,
                'duration_days': (window_end - current).days,
                'cargo_assignments': [],
                'truck_states': {}
            })

            current = window_end
            window_id += 1

        print(f"Created {len(windows)} time windows of {self.planning_window_days} days each")
        return windows

    def filter_cargo_for_window(self, cargo_df, window_start, window_end):
        """Filter cargo that can be picked up within the time window"""
        try:
            cargo_df['Available_From'] = pd.to_datetime(cargo_df['Available_From'])
            cargo_df['Available_To'] = pd.to_datetime(cargo_df['Available_To'])

            # Cargo that overlaps with this time window
            # (cargo available period overlaps with window period)
            overlapping_cargo = cargo_df[
                (cargo_df['Available_From'] <= window_end) &
                (cargo_df['Available_To'] >= window_start)
                ].copy()

            print(
                f"Window {window_start.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}: {len(overlapping_cargo)} cargo items")

            return overlapping_cargo.sort_values('Available_From')

        except Exception as e:
            print(f"Error filtering cargo for window: {str(e)}")
            return cargo_df.copy()

    def initialize_truck_states(self, trucks_df, operation_start):
        """Initialize truck states at the beginning of operation period"""
        truck_states = {}

        for idx, truck in trucks_df.iterrows():
            truck_available_time = pd.to_datetime(truck['Timestamp (dropoff)'])

            # If truck is available before operation starts, use operation start time
            current_time = max(truck_available_time, operation_start)

            truck_states[idx] = {
                'idx': idx,
                'truck': truck,
                'current_pos': (truck['Latitude (dropoff)'], truck['Longitude (dropoff)']),
                'current_time': current_time,
                'cumulative_drive_time': 0,
                'routes': [],
                'last_rest_time': current_time,
                'daily_drive_time': 0,
                'total_profit': 0
            }

        print(f"Initialized {len(truck_states)} truck states")
        return truck_states

    def plan_overnight_rest(self, truck_state, current_time):
        """Plan overnight rest for a truck if needed"""
        # Check if truck needs daily rest (after 9 hours of driving or late hour)
        current_hour = current_time.hour

        # If it's late (after 18:00) or truck has driven 8+ hours, plan overnight rest
        if current_hour >= 18 or truck_state['daily_drive_time'] >= 8:
            # Find next suitable rest time (next day 06:00)
            next_day = current_time.date() + timedelta(days=1)
            rest_end_time = datetime.combine(next_day, datetime.min.time().replace(hour=6))

            # Update truck state
            truck_state['current_time'] = rest_end_time
            truck_state['daily_drive_time'] = 0
            truck_state['cumulative_drive_time'] = 0
            truck_state['last_rest_time'] = rest_end_time

            print(
                f"Planned overnight rest for truck {truck_state['truck']['truck_id']} until {rest_end_time.strftime('%Y-%m-%d %H:%M')}")

            return True

        return False

    def optimize_window(self, window, cargo_window_df, truck_states, global_assigned_cargo):
        """Optimize assignments within a single time window"""
        print(f"Optimizing window {window['start'].strftime('%Y-%m-%d')} to {window['end'].strftime('%Y-%m-%d')}")

        window_assignments = []
        window_profit = 0

        # Filter out already assigned cargo
        available_cargo = cargo_window_df[~cargo_window_df.index.isin(global_assigned_cargo)].copy()

        if len(available_cargo) == 0:
            print("  No available cargo in this window")
            return window_assignments, window_profit

        # Optimize within this window using greedy approach
        max_iterations = len(available_cargo) * len(truck_states) * 2
        iteration = 0
        assigned_in_window = set()

        while len(assigned_in_window) < len(available_cargo) and iteration < max_iterations:
            iteration += 1

            best_assignment = None
            best_profit = -float('inf')
            best_truck_idx = None
            best_cargo_idx = None

            # Find best assignment in this window
            for truck_idx, truck_state in truck_states.items():
                # Check if truck needs overnight rest
                if self.plan_overnight_rest(truck_state, truck_state['current_time']):
                    continue

                # Skip if truck's current time is beyond window end
                if truck_state['current_time'] > window['end']:
                    continue

                for cargo_idx, cargo in available_cargo.iterrows():
                    if cargo_idx in assigned_in_window:
                        continue

                    # Evaluate assignment
                    profit, feasible, route_details, rejection_reason = self.evaluate_cargo_assignment(
                        truck_state,
                        cargo,
                        window['end']  # Use window end as operation end for this window
                    )

                    if feasible and profit > best_profit:
                        best_profit = profit
                        best_assignment = (truck_idx, cargo_idx, route_details)
                        best_truck_idx = truck_idx
                        best_cargo_idx = cargo_idx

            # Apply best assignment if found
            if best_assignment:
                truck_idx, cargo_idx, route_details = best_assignment
                truck_state = truck_states[truck_idx]
                cargo = available_cargo.loc[cargo_idx]

                # Update truck state
                truck_state['current_pos'] = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])
                truck_state['current_time'] = route_details['delivery_time']
                truck_state['cumulative_drive_time'] = route_details.get('cumulative_drive_time', 0)
                truck_state['total_profit'] += best_profit

                # Track daily drive time
                travel_time = route_details.get('travel_to_cargo_hours', 0) + route_details.get(
                    'travel_to_delivery_hours', 0)
                truck_state['daily_drive_time'] += travel_time

                # Record assignment
                assigned_in_window.add(cargo_idx)
                global_assigned_cargo.add(cargo_idx)
                window_profit += best_profit

                # Add to truck's route chain
                route_info = {
                    'cargo': cargo,
                    'profit': best_profit,
                    'profit_breakdown': route_details.get('profit_breakdown', {}),
                    'pickup_time': route_details['pickup_time'],
                    'delivery_time': route_details['delivery_time'],
                    'distance_to_pickup': route_details['distance_to_pickup'],
                    'distance_to_delivery': route_details['distance_to_delivery'],
                    'total_distance': route_details['total_distance'],
                    'waiting_time': route_details.get('waiting_hours', 0),
                    'rest_stops_to_pickup': route_details.get('rest_stops_to_pickup', []),
                    'rest_stops_to_delivery': route_details.get('rest_stops_to_delivery', []),
                    'window_id': window['id']
                }

                truck_state['routes'].append(route_info)
                window_assignments.append((truck_idx, cargo_idx, route_info))

                print(
                    f"  Assigned cargo {cargo_idx} to truck {truck_state['truck']['truck_id']} with profit {best_profit:.2f}")
            else:
                # No more feasible assignments in this window
                break

        print(f"  Window completed: {len(assigned_in_window)} assignments, profit: {window_profit:.2f}")
        return window_assignments, window_profit

    def optimize_fleet_profit_extended(self, trucks_df, cargo_df):
        """
        Extended optimization across multiple time windows with multi-day planning

        Args:
            trucks_df: DataFrame with truck data
            cargo_df: DataFrame with cargo data

        Returns:
            (route_chains, total_profit, rejection_stats, rejection_summary): Extended optimization results
        """
        print(
            f"Starting extended fleet profit optimization with {len(trucks_df)} trucks and {len(cargo_df)} cargo items")

        # Fix the cost calculation (distance_cost + time_cost, not subtract)
        # This is handled in the parent class's profit calculator

        # Calculate operation period
        operation_start, operation_end = self.calculate_operation_period(trucks_df, cargo_df)

        # Create time windows
        time_windows = self.create_time_windows(operation_start, operation_end)

        # Initialize truck states
        truck_states = self.initialize_truck_states(trucks_df, operation_start)

        # Track global assignments
        global_assigned_cargo = set()
        total_profit = 0
        all_window_assignments = []

        # Process each time window
        for window in time_windows:
            print(f"\n=== Processing Window {window['id']} ===")

            # Filter cargo for this window
            cargo_window_df = self.filter_cargo_for_window(cargo_df, window['start'], window['end'])

            if len(cargo_window_df) == 0:
                print(f"No cargo available in window {window['id']}")
                continue

            # Optimize this window
            window_assignments, window_profit = self.optimize_window(
                window, cargo_window_df, truck_states, global_assigned_cargo
            )

            all_window_assignments.extend(window_assignments)
            total_profit += window_profit

            print(
                f"Window {window['id']} completed: {len(window_assignments)} assignments, profit: {window_profit:.2f}")

        # Convert truck states back to route chains format
        route_chains = {}
        for truck_idx, truck_state in truck_states.items():
            if truck_state['routes']:
                route_chains[truck_idx] = truck_state['routes']

        # Calculate rejection statistics
        total_assigned = len(global_assigned_cargo)
        total_cargo = len(cargo_df)
        total_rejected = total_cargo - total_assigned
        assignment_rate = total_assigned / total_cargo if total_cargo > 0 else 0

        rejection_stats = {}
        rejection_summary = {
            'total_rejected': total_rejected,
            'total_cargo': total_cargo,
            'total_assigned': total_assigned,
            'assignment_rate': assignment_rate,
            'by_reason': {},
            'cargo_with_no_truck': []
        }

        # Identify unassigned cargo reasons (simplified for extended mode)
        unassigned_cargo_indices = set(cargo_df.index) - global_assigned_cargo
        if unassigned_cargo_indices:
            # For extended mode, main reasons are usually time window issues or unprofitability
            rejection_summary['by_reason']['Time window or profitability constraints'] = len(unassigned_cargo_indices)

        print(f"\n=== Extended Optimization Complete ===")
        print(f"Total assignments: {total_assigned}/{total_cargo} ({assignment_rate * 100:.1f}%)")
        print(f"Total profit: {total_profit:.2f} EUR")
        print(f"Operation period: {operation_start.strftime('%Y-%m-%d')} to {operation_end.strftime('%Y-%m-%d')}")
        print(f"Planning windows used: {len(time_windows)}")

        return route_chains, total_profit, rejection_stats, rejection_summary

    def evaluate_cargo_assignment(self, truck_info, cargo, operation_end_time):
        """
        Enhanced cargo assignment evaluation for extended planning

        Args:
            truck_info: Dictionary with truck data and current status
            cargo: Cargo data row
            operation_end_time: End time for this planning window

        Returns:
            (profit, feasible, route_details, rejection_reason): Enhanced evaluation results
        """
        # Use parent class method but with enhanced error handling
        try:
            result = super().evaluate_cargo_assignment(truck_info, cargo, operation_end_time)

            # Fix the cost calculation bug here by recalculating profit correctly
            if result[1]:  # if feasible
                route_details = result[2]

                # Recalculate with correct cost formula
                route_info = {
                    'total_distance': route_details['total_distance'],
                    'travel_to_cargo_hours': route_details.get('travel_to_cargo_hours', 0),
                    'travel_to_delivery_hours': route_details.get('travel_to_delivery_hours', 0),
                    'waiting_hours': route_details.get('waiting_hours', 0)
                }

                # Use corrected profit calculation
                corrected_profit, corrected_breakdown = self.calculate_profit_corrected(cargo, truck_info['truck'],
                                                                                        route_info)

                # Update the route details with corrected values
                route_details['profit'] = corrected_profit
                route_details['profit_breakdown'] = corrected_breakdown

                # Check if still profitable after correction
                if corrected_profit <= 0:
                    return corrected_profit, False, {}, "Unprofitable assignment after cost correction"

                return corrected_profit, True, route_details, None

            return result

        except Exception as e:
            return -float('inf'), False, {}, f"Error in extended evaluation: {str(e)}"

    def calculate_profit_corrected(self, cargo, truck, route_info):
        """
        Calculate profit with CORRECTED cost formula: total_cost = distance_cost + time_cost
        """
        # Get premium from cargo
        try:
            premium = float(cargo['Premium'])
            if pd.isna(premium) or premium <= 0:
                premium = 0
        except (KeyError, ValueError, TypeError):
            premium = 0

        # Calculate distance cost
        try:
            total_distance = float(route_info['total_distance'])
            distance_cost = total_distance * self.profit_calculator.price_per_km
        except (KeyError, ValueError, TypeError):
            distance_cost = 0

        # Calculate time cost
        try:
            pickup_time = float(route_info.get('travel_to_cargo_hours', 0))
            delivery_time = float(route_info.get('travel_to_delivery_hours', 0))
            waiting_time = float(route_info.get('waiting_hours', 0))

            time_cost = (pickup_time + waiting_time + delivery_time) * self.profit_calculator.price_per_hour
        except (ValueError, TypeError):
            time_cost = 0

        # CORRECTED FORMULA: Add both distance and time costs
        total_cost = distance_cost + time_cost

        # Calculate profit
        profit = premium - total_cost

        # Store breakdown
        profit_breakdown = {
            'premium': premium,
            'distance_cost': distance_cost,
            'time_cost': time_cost,
            'total_cost': total_cost,
            'profit': profit
        }

        return profit, profit_breakdown
