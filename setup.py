from setuptools import find_packages, setup

setup(
    name="cough-detection",
    version="0.1.0",
    description="Detect vocal sounds, mainly coughing",
    maintainer="Efthymios Stathakis",
    url="https://github.com/Efthymios-Stathakis/vs-detection.git",
    license="MIT",
    packages=find_packages(exclude=["*tests*"]),
    py_modules=["entry_point"]
)