from setuptools import setup, find_packages

setup(
    name='raytrace',
    version='0.1.0',
    description="A package for ray tracing",
    long_description="A package for ray tracing",
    author='Peter T. Brown',
    author_email='ptbrown1729@gmail.com',
    packages=find_packages(include=['raytrace']),
    python_requires='>=3.8',
    install_requires=["numpy",
                      "matplotlib"])