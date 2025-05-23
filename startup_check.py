#!/usr/bin/env python3
"""
Startup Check and Troubleshooting Script for Extended Fleet Optimization
Run this before starting the main application to identify and fix common issues.
"""

import sys
import os
import importlib
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")

    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True


def check_required_packages():
    """Check if all required packages are installed"""
    print("\n📦 Checking required packages...")

    required_packages = {
        'streamlit': '>=1.28.0',
        'pandas': '>=1.5.3',
        'numpy': '>=1.24.3',
        'folium': '>=0.14.0',
        'streamlit_folium': '>=0.15.0',
        'geopy': '>=2.3.0',
        'matplotlib': '>=3.7.1',
        'scipy': '>=1.10.1'
    }

    missing_packages = []

    for package, version in required_packages.items():
        try:
            module = importlib.import_module(package.replace('-', '_'))

            # Try to get version
            version_str = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version_str}")

        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: NOT INSTALLED")

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("✅ All required packages are installed!")
    return True


def check_file_structure():
    """Check if all required files exist"""
    print("\n📁 Checking file structure...")

    required_files = [
        'streamlit_app.py',
        'utils/__init__.py',
        'utils/data_loader.py',
        'utils/profit_optimizer.py',
        'utils/extended_profit_optimizer.py',
        'utils/time_horizon_planner.py',
        'utils/visualization.py',
        'utils/comprehensive_diagnostics.py'
    ]

    missing_files = []

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")

    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False

    print("✅ All required files are present!")
    return True


def check_sample_data():
    """Check if sample data files exist"""
    print("\n📊 Checking sample data files...")

    sample_files = ['trucks.csv', 'cargos.csv']
    existing_files = []

    for file_path in sample_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
            existing_files.append(file_path)
        else:
            print(f"⚠️  {file_path} (optional)")

    if existing_files:
        print("✅ Sample data files available!")
        return True
    else:
        print("⚠️  No sample data files found (you can upload your own)")
        return True


def test_imports():
    """Test importing the main modules"""
    print("\n🔧 Testing module imports...")

    try:
        print("  Testing basic imports...")
        import pandas as pd
        import numpy as np
        import folium
        import streamlit as st
        print("  ✅ Basic modules imported successfully")

        print("  Testing utility imports...")
        sys.path.append('.')

        from utils.data_loader import DataLoader
        print("  ✅ DataLoader imported")

        from utils.profit_optimizer import ProfitCalculator, FleetProfitOptimizer
        print("  ✅ Profit optimizer imported")

        try:
            from utils.extended_profit_optimizer import ExtendedFleetProfitOptimizer
            print("  ✅ Extended optimizer imported")
        except ImportError as e:
            print(f"  ⚠️  Extended optimizer: {e}")

        try:
            from utils.time_horizon_planner import TimeHorizonPlanner
            print("  ✅ Time horizon planner imported")
        except ImportError as e:
            print(f"  ⚠️  Time horizon planner: {e}")

        try:
            from utils.visualization import create_map
            print("  ✅ Visualization imported")
        except ImportError as e:
            print(f"  ⚠️  Visualization: {e}")

        return True

    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False


def check_folium_functionality():
    """Test folium functionality"""
    print("\n🗺️  Testing folium functionality...")

    try:
        import folium
        from folium import plugins

        # Create a simple test map
        m = folium.Map(location=[50.0, 10.0], zoom_start=6)

        # Test adding a marker
        folium.Marker([50.0, 10.0], popup="Test").add_to(m)

        # Test adding a feature group
        group = folium.FeatureGroup(name="Test Group")
        group.add_to(m)

        # Test layer control
        folium.LayerControl().add_to(m)

        print("✅ Folium functionality test passed!")
        return True

    except Exception as e:
        print(f"❌ Folium test failed: {e}")
        return False


def fix_common_issues():
    """Attempt to fix common issues"""
    print("\n🔧 Attempting to fix common issues...")

    # Create utils __init__.py if missing
    utils_init = Path('utils/__init__.py')
    if not utils_init.exists():
        print("  Creating utils/__init__.py...")
        utils_init.parent.mkdir(exist_ok=True)
        utils_init.write_text("# Utils package\n")
        print("  ✅ Created utils/__init__.py")

    # Add current directory to Python path
    if '.' not in sys.path:
        sys.path.insert(0, '.')
        print("  ✅ Added current directory to Python path")

    return True


def generate_startup_report():
    """Generate a comprehensive startup report"""
    print("\n" + "=" * 60)
    print("🚀 EXTENDED FLEET OPTIMIZATION - STARTUP REPORT")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("File Structure", check_file_structure),
        ("Sample Data", check_sample_data),
        ("Module Imports", test_imports),
        ("Folium Functionality", check_folium_functionality)
    ]

    results = {}

    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results[check_name] = False

    # Fix common issues
    fix_common_issues()

    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check_name}: {status}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! You can start the application with:")
        print("   streamlit run streamlit_app.py")
    else:
        print(f"\n⚠️  {total - passed} issues found. Please fix them before starting the application.")

        if not results.get("Required Packages", True):
            print("\n💡 Quick fix for missing packages:")
            print("   pip install -r requirements.txt")

        if not results.get("Module Imports", True):
            print("\n💡 For import issues, check:")
            print("   - All files are in the correct directories")
            print("   - Python path includes current directory")
            print("   - No syntax errors in the modules")


def main():
    """Main startup check function"""
    print("🔍 Extended Fleet Optimization - Startup Check")
    print("This will check your installation and fix common issues.\n")

    generate_startup_report()


if __name__ == "__main__":
    main()