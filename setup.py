from setuptools import setup
from os import path

DIR = path.dirname(path.abspath(__file__))

with open(path.join(DIR, 'README.rst')) as f:
      README = f.read()

setup(name='goturn_pytorch',
      packages=['GoTurn'],
      description='A PyTorch port of GOTURN tracker',
      long_description=README,
      install_requires=['torch'],
      version='1.0',
      url='https://github.com/aakaashjois/PyTorch-GOTURN',
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords=['machine-learning', 'pytorch', 'deep-learning'],
      author='Aakaash Jois',
      author_email='aakaashjois@gmail.com',
      license='MIT')
