from setuptools import setup, find_packages

# Function to read requirements from requirements-lock.txt
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="robust_raster",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements('requirements-lock.txt'),  # Use the lock file
)