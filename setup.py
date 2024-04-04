"""Setup script for the miniTransformer package."""

from setuptools import find_packages, setup

# Read requirements.txt, ignoring lines with git+ dependencies
with open("requirements.txt", encoding="utf-8") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="miniTransformer",
    version="1.0",
    description="Project Description",
    packages=find_packages(),
    install_requires=requirements,
    test_suite="tests",
    include_package_data=True,
    scripts=["scripts/miniTransformer-run"],
    zip_safe=False,
)
