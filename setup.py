from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='vslam2tag',
    version='0.0.1',
    url='https://github.com/tillschulz/vslam_postprocessing.git',
    author='Marius Laska',
    author_email='marius.laska@rwth-aachen.de',
    description='Description of my package',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)
