from setuptools import setup, find_packages,find_namespace_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Jakub Fara",
    author_email='jakubfara77@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="FEniCS extention for solving interface problems",
    #entry_points={
    #    'console_scripts': [
    #        'admesh=admesh.cli:main',
    #    ],
    #},
    #py_module=["admesh4py"],
    #install_requires=requirements,
    license="MIT license",
    #long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='InterfaceSolver',
    name='InterfaceSolver',
    #package_dir={'':'admesh4py'},
    #packages=find_packages(include=['src/python/*']),
    packages=['InterfaceSolver'],
    package_dir={'InterfaceSolver': 'src/InterfaceSolver'},
    package_data={'InterfaceSolver': ['*.py','cpp/*.cpp','cpp/*.py']},
    #packages=find_namespace_packages(where='admesh4py'),
    #test_suite='tests',
    #tests_require=test_requirements,
    url='https://bitbucket.org/FaraJakub/interfacesolver',
    version='0.1.0',
    zip_safe=False,
)
