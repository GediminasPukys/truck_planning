# Fleet Profit Optimization System - Documentation

## Overview

This document provides detailed information about the Fleet Profit Optimization System implemented in the Truck-Cargo Assignment Application. The system is designed to maximize total profit across the entire fleet during a specified operating period by optimally matching trucks with cargo deliveries.

## Key Concepts

### Profit Calculation

The core of the system is a profit calculation model defined as:

**Profit = Premium - Cost**

Where:
- **Premium**: Revenue earned from delivering a cargo (from cargo data)
- **Cost**: Operational costs calculated as:
  - **Cost = (Distance Cost) - (Time Value)**
  - **Distance Cost** = Total distance (km) * Price per km (€)
  - **Time Value** = Total time (hours) * Price per hour (€)

Note that the time component is *subtracted* from the cost (rather than added), meaning longer operational times can reduce costs or increase profits. This reflects the utilization-focused approach where keeping trucks active generates value.

### Operating Period

The system optimizes profit within a global operating period:
- Starting from each truck's initial availability time
- Ending at a common global end time specified by the user
- The optimization aims to maximize profit across the entire period

### Constraints

The optimizer respects several important constraints:
1. **Truck-Cargo Type Matching**: Trucks can only carry cargo of matching types
2. **Maximum Distance**: Trucks can't travel more than the specified maximum distance to pick up cargo
3. **Maximum Waiting Time**: Trucks can't wait longer than the maximum waiting time at pickup locations
4. **Cargo Time Windows**: Cargo must be picked up within its specified availability window
5. **Operation End Time**: All deliveries must be completed before the operation end time
6. **Rest Requirements**: Driver rest periods are enforced following EU regulations:
   - Maximum continuous driving time: 4.5 hours
   - Required break after continuous driving: 45 minutes
   - Daily driving limit: 9 hours
   - Daily rest period: 11 hours

## Configuration Parameters

### Main Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Price per km | 0.39 € | Cost per kilometer traveled |
| Price per hour | 10.0 € | Value of time (subtracted from cost) |
| Maximum Distance | 250 km | Maximum allowed distance to pickup |
| Maximum Waiting Hours | 24 hours | Maximum allowed waiting time at pickup |
| Standard Speed | 73 km/h | Speed used for travel time calculations |
| Operation End Time | 7 days from now at 18:00 | End time for all operations |

### EU Regulations Parameters (Fixed)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Continuous Drive Limit | 4.5 hours | Maximum time a driver can drive without a break |
| Break Duration | 45 minutes | Mandatory break time |
| Daily Drive Limit | 9 hours | Maximum total driving time per day |
| Daily Rest Period | 11 hours | Mandatory daily rest time |

## Optimization Algorithm

The fleet optimization algorithm follows these steps:

1. **Initialization**:
   - Creates a pool of available trucks from the input data
   - Sets initial positions, times, and drive times for each truck
   - Prepares an empty assignment set

2. **Main Assignment Loop**:
   - For each available truck:
     - Evaluates all unassigned cargo
     - Calculates potential profit for each assignment
     - Checks all constraints (type, distance, time windows, etc.)
     - Finds the most profitable feasible assignment

   - If a profitable assignment is found:
     - Updates truck position, time, and cumulative drive time
     - Records the assignment and adds to route chains
     - Continues to the next iteration
   
   - If no profitable assignment is found:
     - Exits the loop

3. **Route Planning**:
   - For each assignment, plans detailed routes including rest stops
   - Calculates pickup routes, delivery routes, and rest locations
   - Tracks cumulative drive time for EU regulation compliance

4. **Profit Calculation**:
   - For each route, calculates detailed profit breakdowns
   - Aggregates profit across all trucks and routes
   - Summarizes fleet-wide utilization metrics

## Data Requirements

### Truck Data (CSV)

Required columns:
- `truck_id`: Unique identifier
- `truck type`: Type of truck (must match cargo types)
- `Address (drop off)`: Drop-off location name
- `Latitude (dropoff)`: Drop-off latitude
- `Longitude (dropoff)`: Drop-off longitude
- `Timestamp (dropoff)`: Drop-off time (initial availability)
- `avg moving speed, km/h`: Average moving speed
- `price per km, Eur`: Price per kilometer
- `waiting time price per h, EUR`: Waiting time price

### Cargo Data (CSV)

Required columns:
- `Origin`: Origin location name
- `Origin_Latitude`: Origin latitude
- `Origin_Longitude`: Origin longitude
- `Available_From`: Availability start time
- `Available_To`: Availability end time
- `Delivery_Location`: Delivery location name
- `Delivery_Latitude`: Delivery latitude
- `Delivery_Longitude`: Delivery longitude
- `Cargo_Type`: Type of cargo (must match truck types)
- `Premium`: Revenue for delivering cargo

## Using the System

### Step 1: Data Preparation
Ensure your truck and cargo data CSVs have all required columns. The Premium column in the cargo data is especially important for profit calculation.

### Step 2: Configure Settings
Adjust the optimization parameters in the sidebar:
- Set distance and timing constraints
- Configure cost parameters (price per km and price per hour)
- Set the operation end date and time

### Step 3: Run Optimization
Click "Maximize Fleet Profit" to run the optimization algorithm.

### Step 4: Analyze Results
Review the results displayed in several sections:
- Optimization Summary with key metrics
- Profit by Truck bar chart
- Detailed Route Plans for each truck
- Route visualizations on the interactive map

### Step 5: Export Results
Use the "Download Route Summary" button to export results as a CSV file.

## Advanced Features

### Profit Breakdown Analysis
For each truck and route, detailed profit breakdowns show:
- Premium value
- Distance cost
- Time value
- Total cost
- Net profit

### Visualization
The interactive map shows:
- Truck starting positions
- Pickup and delivery points
- Route paths with distance information
- Rest stop locations
- Profit information in popup dialogs

### Time-Space Constraints
The system handles complex time-space constraints:
- Respects cargo availability windows
- Schedules driver rest periods
- Manages waiting time at pickup locations
- Enforces maximum distance constraints

## Technical Implementation

The profit optimization system is implemented across several modules:

- `profit_optimizer.py`: Contains the core profit calculation and fleet optimization logic
- `visualization.py`: Handles map creation and profit visualization
- `streamlit_app.py`: Provides the user interface and application flow
- `data_loader.py`: Manages loading and validation of truck and cargo data

The system uses several key classes:

1. **ProfitCalculator**: Calculates profit based on cargo premium and route costs
2. **FleetProfitOptimizer**: Optimizes fleet-wide profit within an operating period
3. **TimeCostCalculator**: Calculates travel times and manages rest periods

## Performance Considerations

For large datasets (many trucks and cargo items), the optimization algorithm may take significant time to run. Consider these tips:

1. Start with smaller datasets to test configurations
2. Use filters to focus on specific truck types or regions
3. Adjust the maximum distance parameter to reduce the search space
4. Set reasonable operation end times based on your data

## Troubleshooting

Common issues and solutions:

1. **No feasible assignments found**:
   - Check that truck and cargo types match
   - Ensure cargo time windows overlap with truck availability
   - Verify that distance constraints aren't too restrictive

2. **Unexpected profit values**:
   - Verify Premium values in cargo data
   - Check price per km and price per hour settings
   - Review the profit breakdown for detailed calculations

3. **Long computation times**:
   - Reduce dataset size with filters
   - Decrease maximum distance parameter
   - Shorten the operation period

## Future Enhancements

Potential improvements for future versions:

1. **Multi-objective optimization**: Balance profit with other factors like environmental impact
2. **Dynamic pricing**: Adjust premiums based on time sensitivity or demand
3. **Stochastic optimization**: Account for uncertainties in travel times and delays
4. **Multi-day planning**: Extend to handle multi-day schedules with overnight stops
5. **Driver-specific constraints**: Add individual driver constraints and preferences
6. **Real-time updates**: Handle dynamic updates to truck positions and cargo availability