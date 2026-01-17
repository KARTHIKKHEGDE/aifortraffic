"""
Arrival Rate Converter - Production Grade
==========================================
Converts real-world congestion data → RL-compatible arrival rates

Key Features:
- Physics-based conversion (congestion length → vehicle arrivals)
- Direction-aware distribution (NS/EW corridor modeling)
- Temporal smoothing (reduces noise in hourly data)
- Validation & sanity checks
- Multiple output formats for different use cases

Author: Traffic Control RL System
Version: 2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

# ============================================
# CONFIGURATION - CALIBRATED FOR BANGALORE
# ============================================

class ConversionConfig:
    """Physics-based calibration parameters."""
    
    # Vehicle spacing in congestion
    VEHICLE_LENGTH_M = 4.5          # Average car length
    VEHICLE_GAP_M = 2.5             # Gap in congestion
    VEHICLE_SPACE_M = 7.0           # Total space per vehicle
    
    # Infrastructure (Silk Board typical)
    LANES_PER_DIRECTION = 3         # Main corridor lanes
    
    # Traffic dynamics
    CONGESTION_TO_FLOW_RATIO = 0.35 # Queue ≈ 35% of hourly throughput
    PEAK_HOUR_FACTOR = 0.85         # Peak 15-min / hourly average
    
    # Time constants
    SECONDS_PER_HOUR = 3600
    TIMESTEP_SEC = 1.0              # Match environment timestep
    
    # Direction distribution (Silk Board: Hosur Rd dominates)
    NS_CORRIDOR_FRACTION = 0.60     # North-South gets 60% of traffic
    EW_CORRIDOR_FRACTION = 0.40     # East-West gets 40%
    
    # Directional asymmetry (morning: southbound, evening: northbound)
    PEAK_DIRECTION_BIAS = 0.15      # 15% more traffic in peak direction
    
    # Saturation limits (prevent unrealistic values)
    MAX_ARRIVAL_RATE_PER_SEC = 4.0  # Cap at 4 veh/sec/direction
    MIN_ARRIVAL_RATE_PER_SEC = 0.05 # Baseline night traffic


# ============================================
# CORE CONVERSION LOGIC
# ============================================

class ArrivalRateConverter:
    """Converts congestion snapshots to arrival rates."""
    
    def __init__(self, config: ConversionConfig = None):
        self.config = config or ConversionConfig()
    
    def congestion_to_total_lambda(self, congestion_km: float) -> float:
        """
        Convert congestion length → total intersection arrival rate.
        
        Physics:
        1. Congestion length → vehicle count in queue
        2. Queue depth → inferred hourly throughput
        3. Throughput → per-second arrival rate
        
        Args:
            congestion_km: Average congestion length (kilometers)
        
        Returns:
            Total arrival rate (vehicles/second for entire intersection)
        """
        if congestion_km <= 0:
            return self.config.MIN_ARRIVAL_RATE_PER_SEC * 4  # Baseline for 4 directions
        
        # Step 1: Length → vehicles
        congestion_m = congestion_km * 1000
        vehicles_per_lane = congestion_m / self.config.VEHICLE_SPACE_M
        total_vehicles_queued = vehicles_per_lane * self.config.LANES_PER_DIRECTION
        
        # Step 2: Queue → hourly flow estimate
        # Logic: If 35% of hourly flow accumulates as queue, then:
        # hourly_flow = queue / 0.35
        estimated_hourly_flow = total_vehicles_queued / self.config.CONGESTION_TO_FLOW_RATIO
        
        # Step 3: Hourly → per-second
        lambda_per_second = estimated_hourly_flow / self.config.SECONDS_PER_HOUR
        
        # Apply saturation limit (prevent explosion)
        lambda_per_second = min(lambda_per_second, self.config.MAX_ARRIVAL_RATE_PER_SEC * 4)
        
        return lambda_per_second
    
    def distribute_to_directions(
        self, 
        total_lambda: float, 
        hour: int,
        mode: str = 'corridor'
    ) -> Dict[str, float]:
        """
        Distribute total λ across 4 directions.
        
        Modes:
        - 'corridor': NS/EW split with time-of-day bias
        - 'uniform': Equal distribution (conservative)
        - 'random': Random perturbation (training robustness)
        
        Args:
            total_lambda: Total arrival rate (veh/sec)
            hour: Hour of day (0-23) for temporal patterns
            mode: Distribution strategy
        
        Returns:
            {'north': λ_N, 'south': λ_S, 'east': λ_E, 'west': λ_W}
        """
        if mode == 'uniform':
            # Simple equal split
            per_direction = total_lambda / 4
            return {d: per_direction for d in ['north', 'south', 'east', 'west']}
        
        elif mode == 'corridor':
            # Realistic NS/EW split with peak-hour bias
            
            # Base corridor split
            ns_total = total_lambda * self.config.NS_CORRIDOR_FRACTION
            ew_total = total_lambda * self.config.EW_CORRIDOR_FRACTION
            
            # Time-of-day directional bias
            # Morning (6-10): Southbound dominant (toward city center)
            # Evening (17-21): Northbound dominant (toward residences)
            if 6 <= hour <= 10:
                # Morning rush: more southbound
                lambda_north = ns_total * (0.5 - self.config.PEAK_DIRECTION_BIAS)
                lambda_south = ns_total * (0.5 + self.config.PEAK_DIRECTION_BIAS)
            elif 17 <= hour <= 21:
                # Evening rush: more northbound
                lambda_north = ns_total * (0.5 + self.config.PEAK_DIRECTION_BIAS)
                lambda_south = ns_total * (0.5 - self.config.PEAK_DIRECTION_BIAS)
            else:
                # Off-peak: balanced
                lambda_north = lambda_south = ns_total / 2
            
            # EW typically balanced (cross traffic)
            lambda_east = lambda_west = ew_total / 2
            
            return {
                'north': lambda_north,
                'south': lambda_south,
                'east': lambda_east,
                'west': lambda_west
            }
        
        elif mode == 'random':
            # Random perturbation (for training diversity)
            base = total_lambda / 4
            return {
                direction: max(
                    self.config.MIN_ARRIVAL_RATE_PER_SEC,
                    base * np.random.uniform(0.6, 1.4)
                )
                for direction in ['north', 'south', 'east', 'west']
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def smooth_temporal(self, arrival_rates: pd.Series, window: int = 3) -> pd.Series:
        """
        Apply moving average smoothing to reduce hourly noise.
        
        Why: Single-hour snapshots can be noisy; smoothing gives
        more stable RL evaluation scenarios.
        """
        return arrival_rates.rolling(window=window, center=True, min_periods=1).mean()


# ============================================
# CSV PROCESSING PIPELINE
# ============================================

def process_congestion_csv(
    input_csv: str,
    output_csv: str,
    distribution_mode: str = 'corridor',
    smooth_window: int = 3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Full pipeline: CSV → Arrival rates with validation.
    
    Args:
        input_csv: Path to hourly congestion data
        output_csv: Path to save processed arrival rates
        distribution_mode: 'corridor', 'uniform', or 'random'
        smooth_window: Temporal smoothing window (hours)
        verbose: Print diagnostics
    
    Returns:
        DataFrame with arrival rates per direction per hour
    """
    # Load data
    df = pd.read_csv(input_csv)
    required_cols = ['hour', 'avg_congestion_km']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Initialize converter
    converter = ArrivalRateConverter()
    
    # Convert congestion → total λ
    df['total_lambda_raw'] = df['avg_congestion_km'].apply(
        converter.congestion_to_total_lambda
    )
    
    # Smooth temporal noise
    df['total_lambda_smooth'] = converter.smooth_temporal(
        df['total_lambda_raw'], 
        window=smooth_window
    )
    
    # Distribute to directions
    direction_data = []
    for idx, row in df.iterrows():
        rates = converter.distribute_to_directions(
            row['total_lambda_smooth'],
            row['hour'],
            mode=distribution_mode
        )
        direction_data.append(rates)
    
    # Merge direction columns
    direction_df = pd.DataFrame(direction_data)
    result = pd.concat([df, direction_df], axis=1)
    
    # Calculate derived metrics
    result['total_lambda_final'] = result[['north', 'south', 'east', 'west']].sum(axis=1)
    result['lambda_per_hour'] = (result['total_lambda_final'] * 3600).round(0).astype(int)
    
    # Validation checks
    if verbose:
        print("=" * 70)
        print("📊 ARRIVAL RATE CONVERSION SUMMARY")
        print("=" * 70)
        print(f"Input: {input_csv}")
        print(f"Output: {output_csv}")
        print(f"Distribution mode: {distribution_mode}")
        print(f"Smoothing window: {smooth_window} hours")
        print()
        print(f"Total λ range: {result['total_lambda_final'].min():.3f} - {result['total_lambda_final'].max():.3f} veh/sec")
        print(f"Peak hour: {result.loc[result['total_lambda_final'].idxmax(), 'hour']:02d}:00")
        print(f"Min hour: {result.loc[result['total_lambda_final'].idxmin(), 'hour']:02d}:00")
        print()
        print("Direction balance:")
        for direction in ['north', 'south', 'east', 'west']:
            avg_lambda = result[direction].mean()
            print(f"  {direction.capitalize():5s}: {avg_lambda:.3f} veh/sec (avg)")
        print("=" * 70)
    
    # Save
    output_cols = ['hour', 'avg_congestion_km', 'total_lambda_final', 
                   'north', 'south', 'east', 'west', 'lambda_per_hour']
    if 'time_slot' in df.columns:
        output_cols.insert(1, 'time_slot')
    
    result[output_cols].to_csv(output_csv, index=False)
    
    return result


# ============================================
# HELPER: GET RATES FOR SPECIFIC HOUR
# ============================================

def get_hourly_rates(csv_path: str, hour: int) -> Dict[str, float]:
    """
    Quick utility: Get arrival rates for a specific hour.
    
    Usage:
        rates = get_hourly_rates('silk_board_arrival_rates.csv', hour=8)
        env.reset(arrival_rates=rates)
    """
    df = pd.read_csv(csv_path)
    row = df[df['hour'] == hour].iloc[0]
    return {
        'north': row['north'],
        'south': row['south'],
        'east': row['east'],
        'west': row['west']
    }


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Process Silk Board data (updated paths)
    input_file = "data/silk_board/silk_board_hourly_avg.csv"
    output_file = "data/silk_board/silk_board_arrival_rates.csv"
    
    result_df = process_congestion_csv(
        input_csv=input_file,
        output_csv=output_file,
        distribution_mode='corridor',  # Use realistic NS/EW split
        smooth_window=3,               # 3-hour moving average
        verbose=True
    )
    
    # Display final table
    print("\n📋 HOURLY ARRIVAL RATES (vehicles/second per direction)")
    print("=" * 70)
    display_cols = ['hour', 'north', 'south', 'east', 'west', 'total_lambda_final']
    print(result_df[display_cols].to_string(index=False))
    print("=" * 70)
    print("✅ Ready for RL evaluation!")