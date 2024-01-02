from setuptools import setup, find_packages
setup(name='diversampling',
      version='0.1.0',
      description='An unsupervised methodology to implement a diversified under-sampling process to deal with the extremely imbalanced binary classification problem',
      author='ReidZ',
      author_email='reid.zhuang@icloud.com',
      requires= ['numpy','sklearn', 'pandas', 'joblib', 'imblearn'], 
      packages=find_packages(),
      package_data={"diversampling": ["demo.ipynb"], "diversampling": ["README.md"], "diversampling":["LICENSE"], "diversampling":["NOTICE"]},
      license="Apache License 2.0",
      install_requires=['numpy','scikit-learn', 'pandas', 'joblib', 'imbalanced-learn']
      )