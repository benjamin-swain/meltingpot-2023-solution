from setuptools import setup, Extension
import numpy as np

# Define the extension module
feature_detector_module = Extension(
    'feature_detector_clean_up',   # Name of the module
    sources=['feature_detector.cpp'],  # Source files
    include_dirs=[np.get_include()],  # You can add include directories here
     extra_compile_args=['-std=c++17'],
    # libraries=[],  # Add required libraries here
    # library_dirs=[],  # Add library directories here
)

# Setup script
setup(
    name='feature_detector_clean_up',
    version='1.0',
    description='This is a package for feature detection',
    ext_modules=[feature_detector_module],
)
