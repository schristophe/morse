import sys

import setuptools

sys.path.insert(0, "morse")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="morse-asteroseismo", # Replace with your own username
    version="0.0.2",
    author="Steven Christophe",
    author_email="steven.christophe@obspm.fr",
    description="Asteroseismic analysis of gamma Dor and SPB stars",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/schristophe/morse",
    packages=setuptools.find_packages(),
    package_data={'': ['README.md', 'LICENSE']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha"
    ],
    project_urls={
    'Documentation': 'https://morse-asteroseismo.readthedocs.io/',
    },
    python_requires='>=3.6',
)
