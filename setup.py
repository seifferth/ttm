from setuptools import setup, find_packages

with open('readme.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ttm',
    version='0.0.0-dev',
    packages=['ttm'],
    entry_points={'console_scripts': [
        'ttm = ttm.cli:main'
    ]},
    python_requires='>=3.7',
    install_requires=[
        'numpy', 'scipy', 'tqdm', 'sklearn', 'gensim', 'flair',
        'umap-learn', 'hdbscan'
    ],
    author='Frank Seifferth',
    author_email='frankseifferth@posteo.net',
    description='A tsv-based Topic Modelling CLI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/seifferth/ttm',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License '
                                   'v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
)
