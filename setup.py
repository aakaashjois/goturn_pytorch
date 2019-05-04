from setuptools import setup

setup(name='pytorch-goturn',
      version='1.0',
      description='A port of GOTURN from Caffe to PyTorch',
      url='https://github.com/aakaashjois/PyTorch-GOTURN',
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      author='Aakaash Jois',
      author_email='aakaashjois@gmail.com',
      license='MIT',
      packages=['pytorch_goturn'],
      install_requires=['torch'],
      include_package_data=True,
      zip_safe=False)
