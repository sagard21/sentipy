import platform
import setuptools
import subprocess

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sentipy',
    version='0.1.0',
    author="Sagar Dawda",
    author_email="sagard21@gmail.com",
    description="Text feature identifier",
    long_description=long_description,
    url="https://github.com/sagard21/pyisa",
    license='MIT License',
    packages=['sentipy'],
    include_package_data=True,
    install_requires=['streamlit', 'click', 'spacy'],
    zip_safe=False,  # install source files not egg
    entry_points={'console_scripts': [
        'sentipy = sentipy.text_explainer:main'
    ]},
)