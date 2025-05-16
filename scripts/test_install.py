#!/usr/bin/env python
# test_install.py - Test Installation Script
# =========================================

"""
This script tests if ContextWormhole is properly installed
and can be imported correctly.
"""

import sys
import importlib.metadata

def check_package(package_name):
    """Check if a package is installed and get its version."""
    try:
        version = importlib.metadata.version(package_name)
        print(f"✅ {package_name} is installed (version {version})")
        return True
    except importlib.metadata.PackageNotFoundError:
        print(f"❌ {package_name} is not installed")
        return False

def test_imports():
    """Test importing key components from ContextWormhole."""
    try:
        from contextwormhole import (
            ContextWormholeModel,
            ExtendedContextConfig,
            sliding_window,
            hierarchical_context,
            attention_sink,
        )
        print("✅ Successfully imported ContextWormhole components")
        return True
    except ImportError as e:
        print(f"❌ Failed to import ContextWormhole components: {e}")
        return False

def main():
    """Run installation tests."""
    print("ContextWormhole Installation Test")
    print("================================")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # Check required packages
    required_packages = [
        "contextwormhole",
        "torch",
        "transformers",
        "numpy",
    ]
    
    all_installed = all(check_package(pkg) for pkg in required_packages)
    
    # Test imports
    imports_ok = test_imports()
    
    # Summary
    print("\nSummary:")
    if all_installed and imports_ok:
        print("✅ ContextWormhole is correctly installed and ready to use!")
    else:
        print("❌ There were issues with the ContextWormhole installation.")
        print("   Please check the output above for details.")

if __name__ == "__main__":
    main()