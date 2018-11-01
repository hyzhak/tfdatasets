import os
import setuptools

script_dir = os.path.dirname(__file__)

with open(os.path.join(script_dir, 'README.md'), 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='tfdatasets',
    version='0.0.1',
    author='Eugene Krevenets',
    author_email='ievgenii.krevenets@gmail.com',
    description='middleware in pipeline between dataset and tensorflow classifier',
    scripts=['./main.py'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hyzhak/tfdatasets',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
