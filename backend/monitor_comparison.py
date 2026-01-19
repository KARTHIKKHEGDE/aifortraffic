"""
Real-time comparison dashboard for Fixed vs Heuristic traffic control
Shows live metrics and highlights key differences
"""

import requests
import time
import os
from colorama import init, Fore, Back, Style

init(autoreset=True)

API_URL = "http://localhost:8000/api/simulation/status"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print(Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "  🚦 TRAFFIC CONTROL COMPARISON: FIXED vs HEURISTIC 🚦")
    print(Fore.CYAN + "=" * 80)
    print()

def print_metrics_comparison(data):
    """Display side-by-side comparison"""
    
    fixed = data.get('fixed', {})
    heuristic = data.get('heuristic', {})
    
    # Header
    print(f"{'METRIC':<30} {'FIXED (30s)':<20} {'HEURISTIC (Adaptive)':<25} {'DIFFERENCE':<15}")
    print("-" * 90)
    
    # Average Wait Time
    fixed_wait = fixed.get('avg_wait_time', 0)
    heur_wait = heuristic.get('avg_wait_time', 0)
    wait_diff = fixed_wait - heur_wait
    wait_color = Fore.GREEN if wait_diff > 0 else Fore.RED
    
    print(f"{'Avg Wait Time (s)':<30} {fixed_wait:<20.2f} {heur_wait:<25.2f} {wait_color}{wait_diff:>+14.2f}")
    
    # Throughput
    fixed_arrived = fixed.get('total_arrived', 0)
    heur_arrived = heuristic.get('total_arrived', 0)
    throughput_diff = heur_arrived - fixed_arrived
    throughput_color = Fore.GREEN if throughput_diff > 0 else Fore.RED
    
    print(f"{'Total Vehicles Served':<30} {fixed_arrived:<20} {heur_arrived:<25} {throughput_color}{throughput_diff:>+14}")
    
    # Active Vehicles
    fixed_active = fixed.get('active_vehicles', 0)
    heur_active = heuristic.get('active_vehicles', 0)
    active_diff = fixed_active - heur_active
    active_color = Fore.GREEN if active_diff > 0 else Fore.RED
    
    print(f"{'Active Vehicles (in system)':<30} {fixed_active:<20} {heur_active:<25} {active_color}{active_diff:>+14}")
    
    # Phase Switches
    fixed_switches = fixed.get('total_switches', 0)
    heur_switches = heuristic.get('total_switches', 0)
    switch_diff = heur_switches - fixed_switches
    
    print(f"{'Total Phase Switches':<30} {fixed_switches:<20} {heur_switches:<25} {switch_diff:>+14}")
    
    print()
    print(Fore.YELLOW + "=" * 90)
    print(Fore.YELLOW + "  HEURISTIC ADAPTIVE FEATURES (What Makes It Smart!)")
    print(Fore.YELLOW + "=" * 90)
    print()
    
    # Heuristic-specific metrics
    early_term = heuristic.get('early_terminations', 0)
    extended = heuristic.get('extended_phases', 0)
    emergency = heuristic.get('emergency_interventions', 0)
    
    print(f"{Fore.CYAN}⚡ Gap-Out (Early Terminations):{Style.RESET_ALL}    {Fore.GREEN}{early_term:>5} {Style.RESET_ALL}← Wasted no time on empty lanes!")
    print(f"{Fore.CYAN}⏰ Extended Phases (Heavy Traffic):{Style.RESET_ALL} {Fore.YELLOW}{extended:>5} {Style.RESET_ALL}← Adapted to high demand")
    print(f"{Fore.CYAN}🚨 Emergency Interventions:{Style.RESET_ALL}        {Fore.RED}{emergency:>5} {Style.RESET_ALL}← Saved lives!")
    
    print()
    
    # Overall improvement
    improvement = data.get('improvement_percentage', 0)
    if improvement > 0:
        print(Fore.GREEN + f"✅ OVERALL IMPROVEMENT: {improvement:.1f}% reduction in wait time!")
    elif improvement < 0:
        print(Fore.RED + f"⚠️  Currently: {abs(improvement):.1f}% longer wait (early simulation)")
    
    print()
    print(f"Simulation Time: {data.get('time', 0):.1f}s")

def monitor_simulation():
    """Continuously monitor and display simulation stats"""
    
    print_header()
    print(Fore.YELLOW + "Connecting to simulation...")
    print()
    
    iteration = 0
    
    try:
        while True:
            try:
                response = requests.get(API_URL, timeout=2)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if simulation is running
                    if data.get('status') == 'running':
                        comparison = data.get('comparison', {})
                        
                        if iteration % 10 == 0:  # Update display every 10 iterations
                            clear_screen()
                            print_header()
                            print_metrics_comparison(comparison)
                            print(Fore.CYAN + "Press Ctrl+C to exit | Updating every 1 second...")
                        
                        iteration += 1
                    else:
                        print(Fore.YELLOW + f"Simulation status: {data.get('status', 'unknown')}")
                        print("Waiting for simulation to start...")
                
                else:
                    print(Fore.RED + f"Error: API returned status {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print(Fore.RED + "Could not connect to backend. Is it running?")
                print(f"Make sure server is running at {API_URL}")
                break
            except requests.exceptions.Timeout:
                print(Fore.YELLOW + "Request timeout, retrying...")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print()
        print(Fore.CYAN + "Monitoring stopped.")
        print()

if __name__ == "__main__":
    monitor_simulation()
