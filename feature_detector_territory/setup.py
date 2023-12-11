from setuptools import setup, Extension

# Define the extension module
def build_ext():
    import numpy as np
    return Extension(
        'feature_detector_territory',   # Name of the module
        sources=['feature_detector.cpp'],  # Source files
        include_dirs=[np.get_include()],  # Include directories
        extra_compile_args=['-std=c++17'],
    )

# Setup script
setup(
    name='feature_detector_territory',
    version='1.0',
    description='This is a package for feature detection',
    ext_modules=[build_ext()],
)
