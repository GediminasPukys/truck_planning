# tests/test_profit_calculator.py
import sys
import os
import unittest
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.profit_optimizer import ProfitCalculator, FleetProfitOptimizer


class TestProfitCalculator(unittest.TestCase):
    """Test cases for the ProfitCalculator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.calculator = ProfitCalculator(
            price_per_km=0.39,
            price_per_hour=10,
            standard_speed_kmh=73
        )
        
        # Sample cargo data
        self.cargo = {
            'Premium': 500,
            'Origin': 'Location A',
            'Origin_Latitude': 50.0,
            'Origin_Longitude': 10.0,
            'Delivery_Location': 'Location B',
            'Delivery_Latitude': 51.0,
            'Delivery_Longitude': 11.0,
            'Cargo_Type': 'General'
        }
        
        # Sample truck data
        self.truck = {
            'truck_id': 1,
            'truck type': 'General',
            'price per km, Eur': 1.0,
            'waiting time price per h, EUR': 10.0
        }
        
        # Sample route info
        self.route_info = {
            'total_distance': 100,
            'travel_to_cargo_hours': 1.0,
            'travel_to_delivery_hours': 0.5,
            'waiting_hours': 0.5
        }

    def test_calculate_travel_time(self):
        """Test travel time calculation"""
        # 73 km/h for 100 km should take 1.37 hours
        self.assertAlmostEqual(self.calculator.calculate_travel_time(100), 1.37, places=2)
        
        # 0 km should take 0 hours
        self.assertEqual(self.calculator.calculate_travel_time(0), 0)

    def test_profit_calculation_basic(self):
        """Test basic profit calculation"""
        profit, breakdown = self.calculator.calculate_profit(
            self.cargo, self.truck, self.route_info
        )
        
        # Calculate expected values
        expected_distance_cost = 100 * 0.39  # 39.0
        expected_time_cost = -(1.0 + 0.5 + 0.5) * 10  # -20.0 (negative because it's subtracted from cost)
        expected_total_cost = expected_distance_cost - expected_time_cost  # 39.0 - (-20.0) = 59.0
        expected_profit = 500 - expected_total_cost  # 500 - 59.0 = 441.0
        
        # Check results
        self.assertAlmostEqual(profit, expected_profit, places=2)
        self.assertAlmostEqual(breakdown['premium'], 500, places=2)
        self.assertAlmostEqual(breakdown['distance_cost'], expected_distance_cost, places=2)
        self.assertAlmostEqual(breakdown['time_cost'], expected_time_cost, places=2)
        self.assertAlmostEqual(breakdown['total_cost'], expected_total_cost, places=2)
        self.assertAlmostEqual(breakdown['profit'], expected_profit, places=2)

    def test_profit_calculation_zero_premium(self):
        """Test profit calculation with zero premium"""
        cargo_zero_premium = self.cargo.copy()
        cargo_zero_premium['Premium'] = 0
        
        profit, breakdown = self.calculator.calculate_profit(
            cargo_zero_premium, self.truck, self.route_info
        )
        
        # With zero premium, profit should be negative
        expected_distance_cost = 100 * 0.39  # 39.0
        expected_time_cost = -(1.0 + 0.5 + 0.5) * 10  # -20.0
        expected_total_cost = expected_distance_cost - expected_time_cost  # 39.0 - (-20.0) = 59.0
        expected_profit = 0 - expected_total_cost  # 0 - 59.0 = -59.0
        
        self.assertAlmostEqual(profit, expected_profit, places=2)
        self.assertAlmostEqual(breakdown['profit'], expected_profit, places=2)

    def test_profit_calculation_zero_distance(self):
        """Test profit calculation with zero distance"""
        route_info_zero_distance = self.route_info.copy()
        route_info_zero_distance['total_distance'] = 0
        
        profit, breakdown = self.calculator.calculate_profit(
            self.cargo, self.truck, route_info_zero_distance
        )
        
        # With zero distance, only time cost applies
        expected_distance_cost = 0
        expected_time_cost = -(1.0 + 0.5 + 0.5) * 10  # -20.0
        expected_total_cost = expected_distance_cost - expected_time_cost  # 0 - (-20.0) = 20.0
        expected_profit = 500 - expected_total_cost  # 500 - 20.0 = 480.0
        
        self.assertAlmostEqual(profit, expected_profit, places=2)
        self.assertAlmostEqual(breakdown['distance_cost'], expected_distance_cost, places=2)
        self.assertAlmostEqual(breakdown['profit'], expected_profit, places=2)
        
    def test_profit_calculation_zero_time(self):
        """Test profit calculation with zero time"""
        route_info_zero_time = self.route_info.copy()
        route_info_zero_time['travel_to_cargo_hours'] = 0
        route_info_zero_time['travel_to_delivery_hours'] = 0
        route_info_zero_time['waiting_hours'] = 0
        
        profit, breakdown = self.calculator.calculate_profit(
            self.cargo, self.truck, route_info_zero_time
        )
        
        # With zero time, only distance cost applies
        expected_distance_cost = 100 * 0.39  # 39.0
        expected_time_cost = 0
        expected_total_cost = expected_distance_cost - expected_time_cost  # 39.0
        expected_profit = 500 - expected_total_cost  # 500 - 39.0 = 461.0
        
        self.assertAlmostEqual(profit, expected_profit, places=2)
        self.assertAlmostEqual(breakdown['time_cost'], expected_time_cost, places=2)
        self.assertAlmostEqual(breakdown['profit'], expected_profit, places=2)


class TestFleetProfitOptimizer(unittest.TestCase):
    """Test cases for the FleetProfitOptimizer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = FleetProfitOptimizer(
            max_distance_km=250,
            max_waiting_hours=24,
            price_per_km=0.39,
            price_per_hour=10,
            standard_speed_kmh=73
        )
        
        # Create sample trucks dataframe
        self.trucks_df = pd.DataFrame([
            {
                'truck_id': 1,
                'truck type': 'General',
                'Address (drop off)': 'Location A',
                'Latitude (dropoff)': 50.0,
                'Longitude (dropoff)': 10.0,
                'Timestamp (dropoff)': '2024-11-24 08:00:00',
                'avg moving speed, km/h': 73,
                'price per km, Eur': 1.0,
                'waiting time price per h, EUR': 10.0
            },
            {
                'truck_id': 2,
                'truck type': 'Frozen',
                'Address (drop off)': 'Location B',
                'Latitude (dropoff)': 51.0,
                'Longitude (dropoff)': 11.0,
                'Timestamp (dropoff)': '2024-11-24 09:00:00',
                'avg moving speed, km/h': 73,
                'price per km, Eur': 1.2,
                'waiting time price per h, EUR': 12.0
            }
        ])
        
        # Create sample cargo dataframe
        self.cargo_df = pd.DataFrame([
            {
                'Origin': 'Origin A',
                'Origin_Latitude': 50.5,
                'Origin_Longitude': 10.5,
                'Available_From': '2024-11-24 10:00:00',
                'Available_To': '2024-11-24 16:00:00',
                'Delivery_Location': 'Delivery A',
                'Delivery_Latitude': 51.5,
                'Delivery_Longitude': 11.5,
                'Cargo_Type': 'General',
                'Premium': 500
            },
            {
                'Origin': 'Origin B',
                'Origin_Latitude': 51.2,
                'Origin_Longitude': 11.2,
                'Available_From': '2024-11-24 11:00:00',
                'Available_To': '2024-11-24 17:00:00',
                'Delivery_Location': 'Delivery B',
                'Delivery_Latitude': 52.2,
                'Delivery_Longitude': 12.2,
                'Cargo_Type': 'Frozen',
                'Premium': 600
            }
        ])
        
        # Set operation end time
        self.operation_end_time = datetime.strptime('2024-11-25 18:00:00', '%Y-%m-%d %H:%M:%S')

    def test_calculate_route_with_rests_no_breaks(self):
        """Test route calculation with no rest breaks needed"""
        # A short route that doesn't need breaks
        start_pos = (50.0, 10.0)
        end_pos = (50.5, 10.5)  # ~56km
        start_time = datetime.strptime('2024-11-24 08:00:00', '%Y-%m-%d %H:%M:%S')
        cumulative_drive_time = 0
        
        route = self.optimizer.calculate_route_with_rests(
            start_pos, end_pos, start_time, cumulative_drive_time
        )
        
        # Check results
        self.assertLess(route['travel_time_hours'], self.optimizer.CONTINUOUS_DRIVE_LIMIT)
        self.assertEqual(len(route['rest_stops']), 0)
        self.assertAlmostEqual(route['updated_drive_time'], route['travel_time_hours'], places=2)
        
        # Check arrival time is correct
        expected_arrival = start_time + timedelta(hours=route['travel_time_hours'])
        self.assertEqual(route['arrival_time'], expected_arrival)

    def test_find_rest_point(self):
        """Test finding a rest point along a route"""
        start_pos = (50.0, 10.0)
        end_pos = (51.0, 11.0)
        distance_covered = 50
        total_distance = 100
        
        rest_point = self.optimizer.find_rest_point(
            start_pos, end_pos, distance_covered, total_distance
        )
        
        # Rest point should be halfway between start and end
        self.assertAlmostEqual(rest_point[0], 50.5, places=1)
        self.assertAlmostEqual(rest_point[1], 10.5, places=1)

    def test_evaluate_cargo_assignment_type_mismatch(self):
        """Test cargo assignment with type mismatch"""
        truck_info = {
            'truck': self.trucks_df.iloc[0],
            'current_pos': (self.trucks_df.iloc[0]['Latitude (dropoff)'], 
                           self.trucks_df.iloc[0]['Longitude (dropoff)']),
            'current_time': pd.to_datetime(self.trucks_df.iloc[0]['Timestamp (dropoff)']),
            'cumulative_drive_time': 0
        }
        
        # Try to assign Frozen cargo to General truck
        cargo = self.cargo_df.iloc[1]  # Frozen cargo
        
        profit, feasible, _ = self.optimizer.evaluate_cargo_assignment(
            truck_info, cargo, self.operation_end_time
        )
        
        # Assignment should not be feasible due to type mismatch
        self.assertFalse(feasible)
        self.assertEqual(profit, -float('inf'))

    def test_evaluate_cargo_assignment_valid(self):
        """Test valid cargo assignment"""
        truck_info = {
            'truck': self.trucks_df.iloc[0],
            'current_pos': (self.trucks_df.iloc[0]['Latitude (dropoff)'], 
                           self.trucks_df.iloc[0]['Longitude (dropoff)']),
            'current_time': pd.to_datetime(self.trucks_df.iloc[0]['Timestamp (dropoff)']),
            'cumulative_drive_time': 0
        }
        
        # Assign General cargo to General truck
        cargo = self.cargo_df.iloc[0]  # General cargo
        
        profit, feasible, route_details = self.optimizer.evaluate_cargo_assignment(
            truck_info, cargo, self.operation_end_time
        )
        
        # Assignment should be feasible
        self.assertTrue(feasible)
        self.assertGreater(profit, 0)
        
        # Check route details
        self.assertIn('pickup_time', route_details)
        self.assertIn('delivery_time', route_details)
        self.assertIn('distance_to_pickup', route_details)
        self.assertIn('distance_to_delivery', route_details)
        self.assertIn('waiting_hours', route_details)
        self.assertIn('profit', route_details)
        self.assertIn('profit_breakdown', route_details)


if __name__ == '__main__':
    unittest.main()