from setuptools import find_packages, setup

setup(
    name='pan-motion-retargeting',
    description="The official PyTorch implementation of the paper `Pose-aware Attention Network for Flexible Motion Retargeting by Body Part.`",
    author='Shihong Xia',
    author_email='xsh@ict.ac.cn',
    python_requires='>=3.8.12',
    install_requires=[
        'setuptools==59.5.0',
        'numpy== 1.21.4',
        'scipy==1.7.3',
        'scikit-learn==1.1.3',
        'tensorboard==2.9.1',
        'tqdm==4.62.3',
        'torchsummary==1.5.1',
    ],

    packages=find_packages(),

)