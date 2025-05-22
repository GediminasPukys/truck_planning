# utils/diagnostics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from geopy.distance import geodesic


def analyze_truck_cargo_compatibility(trucks_df, cargo_df):
    """
    Analyze compatibility between trucks and cargo
    
    Args:
        trucks_df: DataFrame with truck data
        cargo_df: DataFrame with cargo data
        
    Returns:
        compatibility_stats: Dictionary with compatibility statistics
    """
    # Get unique truck and cargo types
    truck_types = set(trucks_df['truck type'].str.lower())
    cargo_types = set(cargo_df['Cargo_Type'].str.lower())
    
    # Check for matching types
    matching_types = truck_types.intersection(cargo_types)
    
    # Count trucks and cargo by type
    truck_type_counts = trucks_df['truck type'].str.lower().value_counts().to_dict()
    cargo_type_counts = cargo_df['Cargo_Type'].str.lower().value_counts().to_dict()
    
    # Compile stats
    compatibility_stats = {
        'truck_types': truck_types,
        'cargo_types': cargo_types,
        'matching_types': matching_types,
        'truck_type_counts': truck_type_counts,
        'cargo_type_counts': cargo_type_counts,
        'has_mismatch': len(matching_types) < len(cargo_types),
        'unmatched_cargo_types': cargo_types - matching_types,
        'unmatched_truck_types': truck_types - matching_types
    }
    
    return compatibility_stats


def analyze_distance_constraints(trucks_df, cargo_df, max_distance_km):
    """
    Analyze distance constraints between trucks and cargo
    
    Args:
        trucks_df: DataFrame with truck data
        cargo_df: DataFrame with cargo data
        max_distance_km: Maximum allowed distance
        
    Returns:
        distance_stats: Dictionary with distance statistics
    """
    # Calculate distances from each truck to each cargo
    distances = []
    
    for _, truck in trucks_df.iterrows():
        truck_pos = (truck['Latitude (dropoff)'], truck['Longitude (dropoff)'])
        truck_type = truck['truck type'].lower()
        
        for _, cargo in cargo_df.iterrows():
            cargo_pos = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])
            cargo_type = cargo['Cargo_Type'].lower()
            
            # Only calculate for matching types
            if truck_type == cargo_type:
                distance = geodesic(truck_pos, cargo_pos).kilometers
                distances.append({
                    'truck_id': truck['truck_id'],
                    'cargo_idx': cargo.name,
                    'distance': distance,
                    'within_limit': distance <= max_distance_km
                })
    
    # Create DataFrame for analysis
    distances_df = pd.DataFrame(distances)
    
    # Calculate stats
    if len(distances_df) > 0:
        percent_within_limit = (distances_df['within_limit'].sum() / len(distances_df)) * 100
        
        distance_stats = {
            'min_distance': distances_df['distance'].min() if not distances_df.empty else None,
            'max_distance': distances_df['distance'].max() if not distances_df.empty else None,
            'avg_distance': distances_df['distance'].mean() if not distances_df.empty else None,
            'median_distance': distances_df['distance'].median() if not distances_df.empty else None,
            'percent_within_limit': percent_within_limit,
            'distances_df': distances_df
        }
    else:
        distance_stats = {
            'min_distance': None,
            'max_distance': None,
            'avg_distance': None,
            'median_distance': None,
            'percent_within_limit': 0,
            'distances_df': distances_df
        }
    
    return distance_stats


def analyze_time_windows(trucks_df, cargo_df, max_waiting_hours, standard_speed_kmh):
    """
    Analyze time window compatibility between trucks and cargo
    
    Args:
        trucks_df: DataFrame with truck data
        cargo_df: DataFrame with cargo data
        max_waiting_hours: Maximum allowed waiting time
        standard_speed_kmh: Standard speed for travel time calculation
        
    Returns:
        time_stats: Dictionary with time window statistics
    """
    # Calculate earliest possible arrival for each truck-cargo pair
    time_windows = []
    
    for _, truck in trucks_df.iterrows():
        truck_pos = (truck['Latitude (dropoff)'], truck['Longitude (dropoff)'])
        truck_type = truck['truck type'].lower()
        truck_avail_time = pd.to_datetime(truck['Timestamp (dropoff)'])
        
        for _, cargo in cargo_df.iterrows():
            cargo_pos = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])
            cargo_type = cargo['Cargo_Type'].lower()
            cargo_from = pd.to_datetime(cargo['Available_From'])
            cargo_to = pd.to_datetime(cargo['Available_To'])
            
            # Only calculate for matching types
            if truck_type == cargo_type:
                # Calculate travel time
                distance = geodesic(truck_pos, cargo_pos).kilometers
                travel_time_hours = distance / standard_speed_kmh
                travel_time_delta = timedelta(hours=travel_time_hours)
                
                # Calculate earliest possible arrival
                earliest_arrival = truck_avail_time + travel_time_delta
                
                # Check compatibility
                if earliest_arrival <= cargo_to:
                    # Can arrive before window ends
                    if earliest_arrival < cargo_from:
                        # Need to wait
                        waiting_time = (cargo_from - earliest_arrival).total_seconds() / 3600
                        waiting_ok = waiting_time <= max_waiting_hours
                        actual_pickup = cargo_from
                    else:
                        # No waiting needed
                        waiting_time = 0
                        waiting_ok = True
                        actual_pickup = earliest_arrival
                    
                    compatible = True
                else:
                    # Arrives after window ends
                    waiting_time = 0
                    waiting_ok = False
                    compatible = False
                    actual_pickup = earliest_arrival
                
                time_windows.append({
                    'truck_id': truck['truck_id'],
                    'cargo_idx': cargo.name,
                    'truck_avail_time': truck_avail_time,
                    'cargo_from': cargo_from,
                    'cargo_to': cargo_to,
                    'earliest_arrival': earliest_arrival,
                    'waiting_time': waiting_time,
                    'waiting_ok': waiting_ok,
                    'compatible': compatible,
                    'actual_pickup': actual_pickup
                })
    
    # Create DataFrame for analysis
    time_windows_df = pd.DataFrame(time_windows)
    
    # Calculate stats
    if len(time_windows_df) > 0:
        percent_compatible = (time_windows_df['compatible'].sum() / len(time_windows_df)) * 100
        
        time_stats = {
            'percent_compatible': percent_compatible,
            'avg_waiting_time': time_windows_df['waiting_time'].mean() if not time_windows_df.empty else None,
            'max_waiting_time': time_windows_df['waiting_time'].max() if not time_windows_df.empty else None,
            'pct_excessive_wait': (100 - time_windows_df['waiting_ok'].mean() * 100) if not time_windows_df.empty else None,
            'time_windows_df': time_windows_df
        }
    else:
        time_stats = {
            'percent_compatible': 0,
            'avg_waiting_time': None,
            'max_waiting_time': None,
            'pct_excessive_wait': None,
            'time_windows_df': time_windows_df
        }
    
    return time_stats


def analyze_profit_potential(trucks_df, cargo_df, price_per_km, price_per_hour, standard_speed_kmh):
    """
    Analyze profit potential for each truck-cargo pair
    
    Args:
        trucks_df: DataFrame with truck data
        cargo_df: DataFrame with cargo data
        price_per_km: Price per kilometer
        price_per_hour: Price per hour
        standard_speed_kmh: Standard speed for travel time calculation
        
    Returns:
        profit_stats: Dictionary with profit statistics
    """
    # Calculate profit for each truck-cargo pair
    profits = []
    
    for _, truck in trucks_df.iterrows():
        truck_pos = (truck['Latitude (dropoff)'], truck['Longitude (dropoff)'])
        truck_type = truck['truck type'].lower()
        
        for _, cargo in cargo_df.iterrows():
            cargo_pos = (cargo['Origin_Latitude'], cargo['Origin_Longitude'])
            cargo_type = cargo['Cargo_Type'].lower()
            
            # Only calculate for matching types
            if truck_type == cargo_type:
                # Calculate distances
                distance_to_pickup = geodesic(truck_pos, cargo_pos).kilometers
                delivery_pos = (cargo['Delivery_Latitude'], cargo['Delivery_Longitude'])
                distance_to_delivery = geodesic(cargo_pos, delivery_pos).kilometers
                total_distance = distance_to_pickup + distance_to_delivery
                
                # Calculate times
                pickup_time = distance_to_pickup / standard_speed_kmh
                delivery_time = distance_to_delivery / standard_speed_kmh
                total_time = pickup_time + delivery_time
                
                # Assume zero waiting time for simplicity
                waiting_time = 0
                
                # Calculate costs
                distance_cost = total_distance * price_per_km
                time_value = (total_time + waiting_time) * price_per_hour
                total_cost = distance_cost - time_value  # Time value is subtracted as per requirements
                
                # Calculate profit
                premium = cargo['Premium'] if 'Premium' in cargo else 0
                profit = premium - total_cost
                
                profits.append({
                    'truck_id': truck['truck_id'],
                    'cargo_idx': cargo.name,
                    'distance_to_pickup': distance_to_pickup,
                    'distance_to_delivery': distance_to_delivery,
                    'total_distance': total_distance,
                    'pickup_time': pickup_time,
                    'delivery_time': delivery_time,
                    'total_time': total_time,
                    'distance_cost': distance_cost,
                    'time_value': -time_value,  # Negative to show it reduces cost
                    'total_cost': total_cost,
                    'premium': premium,
                    'profit': profit,
                    'profitable': profit > 0
                })
    
    # Create DataFrame for analysis
    profits_df = pd.DataFrame(profits)
    
    # Calculate stats
    if len(profits_df) > 0:
        percent_profitable = (profits_df['profitable'].sum() / len(profits_df)) * 100
        
        profit_stats = {
            'percent_profitable': percent_profitable,
            'avg_profit': profits_df['profit'].mean() if not profits_df.empty else None,
            'max_profit': profits_df['profit'].max() if not profits_df.empty else None,
            'min_profit': profits_df['profit'].min() if not profits_df.empty else None,
            'median_profit': profits_df['profit'].median() if not profits_df.empty else None,
            'profits_df': profits_df
        }
    else:
        profit_stats = {
            'percent_profitable': 0,
            'avg_profit': None,
            'max_profit': None,
            'min_profit': None,
            'median_profit': None,
            'profits_df': profits_df
        }
    
    return profit_stats


def run_comprehensive_diagnostics(trucks_df, cargo_df, optimization_params):
    """
    Run comprehensive diagnostics on trucks and cargo data
    
    Args:
        trucks_df: DataFrame with truck data
        cargo_df: DataFrame with cargo data
        optimization_params: Dictionary with optimization parameters
        
    Returns:
        diagnostics: Dictionary with comprehensive diagnostic results
    """
    # Extract optimization parameters
    max_distance_km = optimization_params.get('max_distance_km', 250)
    max_waiting_hours = optimization_params.get('max_waiting_hours', 24)
    price_per_km = optimization_params.get('price_per_km', 0.39)
    price_per_hour = optimization_params.get('price_per_hour', 10)
    standard_speed_kmh = optimization_params.get('standard_speed_kmh', 73)
    
    # Run individual diagnostics
    compatibility_stats = analyze_truck_cargo_compatibility(trucks_df, cargo_df)
    distance_stats = analyze_distance_constraints(trucks_df, cargo_df, max_distance_km)
    time_stats = analyze_time_windows(trucks_df, cargo_df, max_waiting_hours, standard_speed_kmh)
    profit_stats = analyze_profit_potential(trucks_df, cargo_df, price_per_km, price_per_hour, standard_speed_kmh)
    
    # Combine results
    diagnostics = {
        'compatibility': compatibility_stats,
        'distance': distance_stats,
        'time_windows': time_stats,
        'profit': profit_stats,
        'parameters': optimization_params
    }
    
    # Calculate overall feasibility
    if (compatibility_stats['has_mismatch'] or 
        distance_stats['percent_within_limit'] < 1.0 or 
        time_stats['percent_compatible'] < 1.0 or 
        profit_stats['percent_profitable'] < 1.0):
        
        # Identify primary constraint
        constraints = {
            'Type Mismatch': 1.0 if not compatibility_stats['has_mismatch'] else 0.0,
            'Distance': distance_stats['percent_within_limit'] / 100,
            'Time Windows': time_stats['percent_compatible'] / 100,
            'Profitability': profit_stats['percent_profitable'] / 100
        }
        
        primary_constraint = min(constraints.items(), key=lambda x: x[1])
        diagnostics['primary_constraint'] = primary_constraint[0]
        diagnostics['feasibility_score'] = min(constraints.values())
    else:
        diagnostics['primary_constraint'] = None
        diagnostics['feasibility_score'] = 1.0
    
    return diagnostics


def generate_diagnostic_report(diagnostics):
    """
    Generate a diagnostic report based on diagnostics results
    
    Args:
        diagnostics: Dictionary with diagnostic results
        
    Returns:
        report: Dictionary with report sections
    """
    report = {
        'summary': "# Diagnostic Report\n\n",
        'compatibility': "## Type Compatibility\n\n",
        'distance': "## Distance Constraints\n\n",
        'time_windows': "## Time Windows\n\n",
        'profit': "## Profit Analysis\n\n",
        'recommendations': "## Recommendations\n\n"
    }
    
    # Summary
    if diagnostics['feasibility_score'] < 1.0:
        report['summary'] += f"Overall feasibility score: {diagnostics['feasibility_score']*100:.1f}%\n\n"
        report['summary'] += f"Primary constraint: **{diagnostics['primary_constraint']}**\n\n"
    else:
        report['summary'] += "All constraints are satisfied. Optimization should be successful.\n\n"
    
    # Compatibility
    compat = diagnostics['compatibility']
    if compat['has_mismatch']:
        report['compatibility'] += "⚠️ **Type Mismatch Detected**\n\n"
        report['compatibility'] += "The following cargo types have no matching trucks:\n\n"
        for c_type in compat['unmatched_cargo_types']:
            report['compatibility'] += f"- {c_type}\n"
    else:
        report['compatibility'] += "✅ All cargo types have matching trucks\n\n"
    
    report['compatibility'] += "\n**Truck Types:**\n\n"
    for t_type, count in compat['truck_type_counts'].items():
        report['compatibility'] += f"- {t_type}: {count} trucks\n"
    
    report['compatibility'] += "\n**Cargo Types:**\n\n"
    for c_type, count in compat['cargo_type_counts'].items():
        report['compatibility'] += f"- {c_type}: {count} items\n"
    
    # Distance
    dist = diagnostics['distance']
    if dist['percent_within_limit'] < 100:
        report['distance'] += f"⚠️ **Distance Issues Detected** - Only {dist['percent_within_limit']:.1f}% of truck-cargo pairs are within the maximum distance\n\n"
    else:
        report['distance'] += "✅ All truck-cargo pairs are within the maximum distance\n\n"
    
    report['distance'] += f"- Maximum allowed distance: {diagnostics['parameters']['max_distance_km']} km\n"
    report['distance'] += f"- Minimum distance: {dist['min_distance']:.1f} km\n" if dist['min_distance'] else ""
    report['distance'] += f"- Maximum distance: {dist['max_distance']:.1f} km\n" if dist['max_distance'] else ""
    report['distance'] += f"- Average distance: {dist['avg_distance']:.1f} km\n" if dist['avg_distance'] else ""
    
    # Time Windows
    time = diagnostics['time_windows']
    if time['percent_compatible'] < 100:
        report['time_windows'] += f"⚠️ **Time Window Issues Detected** - Only {time['percent_compatible']:.1f}% of truck-cargo pairs have compatible time windows\n\n"
    else:
        report['time_windows'] += "✅ All truck-cargo pairs have compatible time windows\n\n"
    
    report['time_windows'] += f"- Maximum allowed waiting time: {diagnostics['parameters']['max_waiting_hours']} hours\n"
    report['time_windows'] += f"- Average waiting time: {time['avg_waiting_time']:.1f} hours\n" if time['avg_waiting_time'] else ""
    report['time_windows'] += f"- Maximum waiting time: {time['max_waiting_time']:.1f} hours\n" if time['max_waiting_time'] else ""
    
    if time['pct_excessive_wait'] and time['pct_excessive_wait'] > 0:
        report['time_windows'] += f"- {time['pct_excessive_wait']:.1f}% of pairs exceed maximum waiting time\n"
    
    # Profit
    profit = diagnostics['profit']
    if profit['percent_profitable'] < 100:
        report['profit'] += f"⚠️ **Profitability Issues Detected** - Only {profit['percent_profitable']:.1f}% of truck-cargo pairs are profitable\n\n"
    else:
        report['profit'] += "✅ All truck-cargo pairs are profitable\n\n"
    
    report['profit'] += f"- Price per km: {diagnostics['parameters']['price_per_km']} EUR\n"
    report['profit'] += f"- Price per hour: {diagnostics['parameters']['price_per_hour']} EUR\n"
    report['profit'] += f"- Average profit: {profit['avg_profit']:.2f} EUR\n" if profit['avg_profit'] else ""
    report['profit'] += f"- Median profit: {profit['median_profit']:.2f} EUR\n" if profit['median_profit'] else ""
    report['profit'] += f"- Maximum profit: {profit['max_profit']:.2f} EUR\n" if profit['max_profit'] else ""
    report['profit'] += f"- Minimum profit: {profit['min_profit']:.2f} EUR\n" if profit['min_profit'] else ""
    
    # Recommendations
    if diagnostics['primary_constraint'] == 'Type Mismatch':
        report['recommendations'] += "### Type Mismatch Recommendations\n\n"
        report['recommendations'] += "1. Add trucks of the missing cargo types\n"
        report['recommendations'] += "2. Filter out cargo types that don't have matching trucks\n"
        report['recommendations'] += "3. Rename cargo types to match available truck types\n"
    
    if diagnostics['primary_constraint'] == 'Distance':
        report['recommendations'] += "### Distance Recommendations\n\n"
        report['recommendations'] += f"1. Increase the Maximum Distance parameter (currently {diagnostics['parameters']['max_distance_km']} km)\n"
        report['recommendations'] += "2. Filter cargo to locations closer to truck starting positions\n"
        report['recommendations'] += "3. Add more trucks in different locations\n"
    
    if diagnostics['primary_constraint'] == 'Time Windows':
        report['recommendations'] += "### Time Window Recommendations\n\n"
        report['recommendations'] += f"1. Increase the Maximum Waiting Hours parameter (currently {diagnostics['parameters']['max_waiting_hours']} hours)\n"
        report['recommendations'] += "2. Filter cargo with time windows closer to truck availability times\n"
        report['recommendations'] += "3. Use trucks that are available earlier\n"
    
    if diagnostics['primary_constraint'] == 'Profitability':
        report['recommendations'] += "### Profitability Recommendations\n\n"
        report['recommendations'] += f"1. Adjust Price per km (currently {diagnostics['parameters']['price_per_km']} EUR)\n"
        report['recommendations'] += f"2. Adjust Price per hour (currently {diagnostics['parameters']['price_per_hour']} EUR)\n"
        report['recommendations'] += "3. Filter to cargo with higher Premium values\n"
    
    return report