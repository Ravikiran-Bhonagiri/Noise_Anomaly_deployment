import sys
import time
import platform
import numpy as np

def check_system_precision():
    """Check numerical and time-related precision metrics"""
    
    # 1. Check floating-point precision (machine epsilon)
    machine_epsilon = sys.float_info.epsilon
    print(f"Floating-point precision (machine epsilon): {machine_epsilon:.3e}")
    
    # 2. Check smallest measurable time difference
    start = time.time()
    end = time.time()
    delta = end - start
    while delta <= 0:  # Ensure we get a measurable difference
        end = time.time()
        delta = end - start
    print(f"Smallest measurable time delta: {delta:.3e} seconds")

    # 3. System/OS information
    print("\nSystem Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Python implementation: {platform.python_implementation()}")
    print(f"Python version: {platform.python_version()}")

if __name__ == "__main__":
    check_system_precision()


    # Check Python float precision (usually FP64)
    print("Python float size:", sys.float_info.dig)  # 15-17 digits (FP64)
