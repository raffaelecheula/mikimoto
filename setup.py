from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="mikimoto",
    version="0.0.1",
    description="Microkinetic modeling tools.",
    url="https://github.com/raffaelecheula/mikimoto.git",
    author="Raffaele Cheula",
    author_email="cheula.raffaele@gmail.com",
    long_description=long_description,
    license='GPL-3.0',
)