==============
pytorch_goturn
==============
A PyTorch port of GOTURN tracker
________________________________

**Installation:**

  1. Download this repository
  2. navigate to the root of the repository and run :code:`pip install .`
  
**Usage:**

To create the model,

.. code:: python
  from pytorch_goturn import Goturn
  
  goturn = Goturn()

To create a pretrained model, 

.. code:: python
  from pytorch_goturn import Goturn
  
  goturn = Goturn(pretrained="path_to_weights.pkl")

The `weights.pkl` file is located at the root of the repository.

**Future updates:**
  
  1. Add library to PyPi and Conda
  2. Automatic downloading of weights

**License:**

This project is released under `MIT License <./LICENSE>`_.

**Author:**

This project is created and maintained by `Aakaash Jois <https://aakaashjois.com/>`_.
