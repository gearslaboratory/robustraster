from setuptools import setup, find_packages
import os

# Function to read requirements from requirements-lock.txt
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="robustraster",
    version="0.1.1",
    author="Adriano Matos",
    author_email="adrianom@unr.edu",
    description="Running user-defined functions on large datasets via out-of-core computation simplfied.",
    long_description=open("README.MD").read(),
    url="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=parse_requirements('requirements-lock.txt'),  # Use the lock file
    include_package_data=True,
)