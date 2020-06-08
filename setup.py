from setuptools import setup

setup(
    name='frbpa',
    version='0.1',
    packages=['frbpa'],
    url='https://github.com/KshitijAggarwal/frbpa',
    author='Kshitij Aggarwal',
    author_email='ka0064@mix.wvu.edu',
    license='',
    description='Periodicity Analysis of Repeating FRBs',
    install_requires=['astropy', 'numpy', 'matplotlib', 'tqdm', 'scipy', 'P4J', 'riptide-ffa'],
)
