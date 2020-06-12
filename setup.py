import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymc3_ext", # Replace with your own username
    version="0.2.1",
    author="Vladislav Strashko",
    author_email="wlad962961@gmail.com",
    description="PyMC3 extension for simulating non-numeri random variables",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wlad111/pymc3",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
