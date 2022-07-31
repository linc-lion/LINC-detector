from setuptools import setup, find_packages

requirements_file = open('requirements.txt')
requirements = requirements_file.read().strip().split('\n')

print(f'Setup requirements: {requirements}')

setup(
    name='linc-detector',
    version='0.2dev',
    packages=find_packages(exclude=['datasets', 'notebooks', 'pictures']),
    author='LINC',
    author_email=' tech@linclion.org',
    description='Customized Faster R-CNN for LINC lion detection'
)
