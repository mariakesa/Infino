from setuptools import setup, find_packages

setup(
    name='infino',
    version='0.1.0',
    author='Maria Kesa',
    description='Infino: A toy epistemic AI that learns by compression and causality.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mariakesa/Infino',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    include_package_data=True,
)
