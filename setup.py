'''
The setup.py file is an essential part of packaging and 
distributing python projects. It is used by setuptools 
(or distutils in older python versions) to define the configuration
of your project, such as its metadata, dependencies, and more
 '''

from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    This function reads the requirements.txt file and 
    returns a list of required dependencies.
    """
    requirement_lst = []  # Initialize the list to store requirements

    try:
        with open('requirements.txt', 'r') as file:
            # Read lines from the file
            lines = file.readlines()
            # Process each line
            for line in lines:
                requirement = line.strip()
                # Ignore empty lines and '-e .'
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
    return requirement_lst

# Example usage of get_requirements()
if __name__ == "__main__":
    print(get_requirements())

# Packaging setup
setup(
    name="networksecurity",
    version="0.0.1",
    author="Pinki",
    packages=find_packages(),
    install_requires=get_requirements(),  # Use the get_requirements function here
)
