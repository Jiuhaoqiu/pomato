from __future__ import absolute_import


from setuptools import setup, find_packages
from codecs import open


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pomato',
    version='0.1.1',
    author='Robert Mieth (TU Berlin), Richard Weinhold (TU Berlin)',
    author_email='rom@wip.tu-berlin.de, riw@wip.tu-berlin.de',
    description='Power Market Tool',
    long_description='Power Market Tool',
    url='https://github.com/korpuskel91/pomato',
    license='GPLv3',
    packages=['pomato'],
    include_package_data=True,
    install_requires=['numpy','pathlib','scipy','pandas>=0.19.0','bokeh'],
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ])
