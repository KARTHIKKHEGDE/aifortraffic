"""
SUMO Utilities Module
Handles SUMO installation verification and binary paths
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from .logger import setup_logger

logger = setup_logger("sumo_utils")


def get_sumo_home() -> Optional[Path]:
    """
    Get SUMO_HOME environment variable
    
    Returns:
        Path to SUMO installation or None if not set
    """
    sumo_home = os.environ.get('SUMO_HOME')
    
    if sumo_home:
        return Path(sumo_home)
    
    # Try common installation paths on Windows
    common_paths = [
        Path("C:/Program Files (x86)/Eclipse/Sumo"),
        Path("C:/Program Files/Eclipse/Sumo"),
        Path("C:/sumo"),
        Path.home() / "sumo",
    ]
    
    for path in common_paths:
        if path.exists() and (path / "bin" / "sumo.exe").exists():
            logger.warning(f"SUMO_HOME not set, but found SUMO at: {path}")
            return path
    
    return None


def get_sumo_binary(gui: bool = False) -> str:
    """
    Get path to SUMO binary
    
    Args:
        gui: Whether to get SUMO-GUI binary
    
    Returns:
        Path to SUMO binary
    """
    sumo_home = get_sumo_home()
    
    if sumo_home is None:
        raise EnvironmentError(
            "SUMO_HOME environment variable is not set. "
            "Please install SUMO and set the SUMO_HOME environment variable. "
            "Download from: https://sumo.dlr.de/docs/Downloads.php"
        )
    
    binary_name = "sumo-gui" if gui else "sumo"
    
    # Windows
    if sys.platform == "win32":
        binary_path = sumo_home / "bin" / f"{binary_name}.exe"
    else:
        binary_path = sumo_home / "bin" / binary_name
    
    if not binary_path.exists():
        raise FileNotFoundError(f"SUMO binary not found: {binary_path}")
    
    return str(binary_path)


def get_netconvert_binary() -> str:
    """Get path to netconvert binary for network conversion"""
    sumo_home = get_sumo_home()
    
    if sumo_home is None:
        raise EnvironmentError("SUMO_HOME not set")
    
    if sys.platform == "win32":
        binary_path = sumo_home / "bin" / "netconvert.exe"
    else:
        binary_path = sumo_home / "bin" / "netconvert"
    
    if not binary_path.exists():
        raise FileNotFoundError(f"netconvert not found: {binary_path}")
    
    return str(binary_path)


def get_random_trips_script() -> str:
    """Get path to randomTrips.py script"""
    sumo_home = get_sumo_home()
    
    if sumo_home is None:
        raise EnvironmentError("SUMO_HOME not set")
    
    script_path = sumo_home / "tools" / "randomTrips.py"
    
    if not script_path.exists():
        raise FileNotFoundError(f"randomTrips.py not found: {script_path}")
    
    return str(script_path)


def check_sumo_installation() -> Tuple[bool, str]:
    """
    Check if SUMO is properly installed
    
    Returns:
        Tuple of (success, message)
    """
    try:
        sumo_home = get_sumo_home()
        
        if sumo_home is None:
            return False, (
                "SUMO_HOME environment variable is not set.\n"
                "Please install SUMO from https://sumo.dlr.de/docs/Downloads.php\n"
                "Then set SUMO_HOME to the installation directory."
            )
        
        # Check for essential binaries
        sumo_bin = get_sumo_binary(gui=False)
        sumo_gui_bin = get_sumo_binary(gui=True)
        netconvert_bin = get_netconvert_binary()
        
        # Try to get version
        result = subprocess.run(
            [sumo_bin, "--version"],
            capture_output=True,
            text=True
        )
        
        version_info = result.stdout.strip().split('\n')[0] if result.stdout else "Unknown"
        
        return True, (
            f"SUMO Installation Found!\n"
            f"  SUMO_HOME: {sumo_home}\n"
            f"  Version: {version_info}\n"
            f"  SUMO Binary: {sumo_bin}\n"
            f"  SUMO-GUI Binary: {sumo_gui_bin}\n"
            f"  netconvert: {netconvert_bin}"
        )
        
    except Exception as e:
        return False, f"SUMO installation check failed: {str(e)}"


def add_sumo_tools_to_path():
    """Add SUMO tools directory to Python path"""
    sumo_home = get_sumo_home()
    
    if sumo_home:
        tools_path = sumo_home / "tools"
        if str(tools_path) not in sys.path:
            sys.path.insert(0, str(tools_path))
            logger.info(f"Added SUMO tools to path: {tools_path}")


def import_traci():
    """
    Import TraCI with proper path setup
    
    Returns:
        traci module
    """
    add_sumo_tools_to_path()
    
    try:
        import traci
        return traci
    except ImportError as e:
        raise ImportError(
            "Failed to import traci. Make sure SUMO is installed and "
            "SUMO_HOME is set correctly. "
            f"Error: {e}"
        )


def import_sumolib():
    """
    Import sumolib with proper path setup
    
    Returns:
        sumolib module
    """
    add_sumo_tools_to_path()
    
    try:
        import sumolib
        return sumolib
    except ImportError as e:
        raise ImportError(
            "Failed to import sumolib. Make sure SUMO is installed. "
            f"Error: {e}"
        )


def create_sumo_config_file(
    net_file: str,
    route_file: str,
    output_dir: str,
    begin: int = 0,
    end: int = 3600,
    step_length: float = 1.0
) -> str:
    """
    Create a SUMO configuration file (.sumocfg)
    
    Args:
        net_file: Path to network file
        route_file: Path to route file
        output_dir: Directory for output files
        begin: Simulation start time
        end: Simulation end time
        step_length: Simulation step length
    
    Returns:
        Path to created configuration file
    """
    config_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_file}"/>
    </input>
    
    <time>
        <begin value="{begin}"/>
        <end value="{end}"/>
        <step-length value="{step_length}"/>
    </time>
    
    <output>
        <tripinfo-output value="{output_dir}/tripinfo.xml"/>
        <summary-output value="{output_dir}/summary.xml"/>
    </output>
    
    <processing>
        <time-to-teleport value="-1"/>
        <ignore-route-errors value="true"/>
    </processing>
    
    <report>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
    
</configuration>
'''
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Write config file
    config_path = Path(output_dir) / "simulation.sumocfg"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Created SUMO config file: {config_path}")
    
    return str(config_path)


if __name__ == "__main__":
    # Test SUMO installation check
    print("Checking SUMO installation...")
    
    success, message = check_sumo_installation()
    
    if success:
        print("✓ SUMO is properly installed!")
        print(message)
    else:
        print("✗ SUMO installation issue:")
        print(message)
    
    # Test TraCI import
    print("\nTesting TraCI import...")
    try:
        traci = import_traci()
        print(f"✓ TraCI imported successfully: {traci}")
    except ImportError as e:
        print(f"✗ TraCI import failed: {e}")
