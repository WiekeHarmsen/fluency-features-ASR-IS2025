from setuptools import setup, find_packages
import os

VERSION = '0.0.1' 
DESCRIPTION = 'Pipeline to compute fluency features of read speech'
LONG_DESCRIPTION = 'Implements ASR and eGeMAPS features'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="fluency-features-ASR-IS2025", 
        version=VERSION,
        author="Wieke Harmsen",
        author_email="<wieke.harmsen@ru.nl>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        keywords=['python', 'asr', 'speech processing', 'fluency features'],
)