import platform
import setuptools
import subprocess
from setuptools.command.install import install

class MyInstall(install):

    def run(self):
        install.run(self)
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    reqs = f.readlines()

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
    package_data={'sentipy': ['sentipy/data/*', 'sentipy/model/**/*', '*', '*/**', '*/*/**']},
    include_package_data=True,
    install_requires=reqs,
    zip_safe=False,  # install source files not egg
    entry_points={'console_scripts': [
        'sentipy = sentipy.text_explainer:main'
    ]},
    cmdclass={'install': MyInstall}
)