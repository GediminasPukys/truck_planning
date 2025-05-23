# utils/__init__.py (Updated to handle new modules)

# Import fixes for the extended optimization system

# Ensure backward compatibility
try:
    from .extended_profit_optimizer import ExtendedFleetProfitOptimizer
except ImportError:
    print("Warning: ExtendedFleetProfitOptimizer not available")
    ExtendedFleetProfitOptimizer = None

try:
    from .time_horizon_planner import TimeHorizonPlanner, MultiDayRoutePlanner
except ImportError:
    print("Warning: Time horizon planning modules not available")
    TimeHorizonPlanner = None
    MultiDayRoutePlanner = None

try:
    from .comprehensive_diagnostics import ExtendedDiagnostics, run_extended_diagnostics
except ImportError:
    print("Warning: Extended diagnostics not available")
    ExtendedDiagnostics = None
    run_extended_diagnostics = None

# Re-export main classes for easy importing
__all__ = [
    'ExtendedFleetProfitOptimizer',
    'TimeHorizonPlanner',
    'MultiDayRoutePlanner',
    'ExtendedDiagnostics',
    'run_extended_diagnostics'
]


# Alternative import fix for streamlit_app.py
def safe_import_fallback():
    """Fallback imports for missing modules"""
    import sys

    # Create dummy classes if modules are missing
    if 'utils.extended_profit_optimizer' not in sys.modules:
        class DummyExtendedOptimizer:
            def __init__(self, *args, **kwargs):
                raise ImportError("ExtendedFleetProfitOptimizer not available")

        sys.modules['utils.extended_profit_optimizer'] = type('Module', (), {
            'ExtendedFleetProfitOptimizer': DummyExtendedOptimizer
        })()

    if 'utils.time_horizon_planner' not in sys.modules:
        class DummyPlanner:
            def __init__(self, *args, **kwargs):
                raise ImportError("Time horizon planners not available")

        sys.modules['utils.time_horizon_planner'] = type('Module', (), {
            'TimeHorizonPlanner': DummyPlanner,
            'MultiDayRoutePlanner': DummyPlanner
        })()


# Quick fix for folium issues
def fix_folium_compatibility():
    """Fix common folium compatibility issues"""
    try:
        import folium

        # Check if folium version supports required features
        folium_version = getattr(folium, '__version__', '0.0.0')
        major_version = int(folium_version.split('.')[0])

        if major_version < 0:  # Very old version
            print("Warning: Folium version may be too old for some features")
            return False

        return True

    except ImportError:
        print("Error: Folium not installed. Please install with: pip install folium")
        return False


# Module loading test
def test_module_imports():
    """Test that all required modules can be imported"""
    modules_to_test = [
        'pandas',
        'numpy',
        'folium',
        'streamlit',
        'geopy',
        'matplotlib',
        'datetime'
    ]

    missing_modules = []

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError:
            missing_modules.append(module_name)
            print(f"✗ {module_name}")

    if missing_modules:
        print(f"\nMissing modules: {', '.join(missing_modules)}")
        print("Install missing modules with:")
        print(f"pip install {' '.join(missing_modules)}")
        return False

    print("\nAll required modules are available!")
    return True


if __name__ == "__main__":
    print("Testing module imports...")
    test_module_imports()
    print("\nTesting folium compatibility...")
    fix_folium_compatibility()