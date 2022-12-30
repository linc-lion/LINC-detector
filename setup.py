from setuptools import setup, find_packages

setup(
    name='linc-detector',
    description='Python package that holds model to detect lion facial parts',
    version='0.1.2',
    packages=find_packages(),
    author='Lion Guardians',
    author_email=' tech@linclion.org',
    long_description='See https://github.com/linc-lion/LINC-detector',
    long_description_content_type='text/x-rst',
    install_requires=[
        'numpy==1.23.3',
        'future==0.16.0',
        'torch==1.10.1',
        'torchvision==0.11.2'
    ],
    classifiers=[
        'Topic :: Software Development :: Libraries',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
)
