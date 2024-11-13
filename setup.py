from setuptools import setup, find_packages

with open("requirements.txt", 'r') as ifile:
    requirements = ifile.read().splitlines()

setup(
    name="cellstitch",
    version="1.0.0",
    description="Cellstitch: 3D cellular anisotropic image segmentation via optimal transport",
    # url="https://cellstitch.readthedocs.io",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=requirements,
    # package_data = ""; include_package_data=True
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
