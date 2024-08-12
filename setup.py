from setuptools import setup, find_packages

setup(
    name='spgdes',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'xgboost',
        'scipy',
        'pickle-mixin'   # Asegure-se que esta é a biblioteca correta para funcionalidades pickle
    ],
    python_requires='>=3.10',
    author='Alberto Manastarla',
    author_email='manastarla@hotmail.com',
    description='Self-generating Prototype Dynamic Selection Ensemble (SGPDES) Algorithm',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/manastarla/spgdes',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
