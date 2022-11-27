from setuptools import setup, find_packages

requirements_file = open('requirements.txt')
requirements = requirements_file.read().strip().split('\n')

print(f'Setup requirements: {requirements}')

setup(
    name='linc-detector',
    description='Python package that holds model to detect lion parts',
    install_requires=requirements,
    version='0.2dev',
    packages=find_packages(),
    author='LINC',
    author_email=' tech@linclion.org'
)
