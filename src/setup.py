from setuptools import setup

setup(
    name="netconstructor",
    version="1.0",
    description="Library for creating neural networks",
    url="https://github.com/IrinaTerlizhenko/ml-neural-net-constructor",
    packages=["netconstructor", ],
    setup_requires=["pytest-runner", ],
    tests_require=["pytest", ],
    zip_safe=False
)
