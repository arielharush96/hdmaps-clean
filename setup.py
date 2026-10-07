from setuptools import setup, find_packages
setup(name="hdmaps", version="1.0.0", packages=find_packages(include=["hdmaps", "hdmaps.*", "highwayenv", "highwayenv.*", "src", "src.*"]), python_requires=">=3.10", install_requires=["stable-baselines3==2.8.0a2", "gymnasium==1.2.3", "highway-env==1.10.2", "torch>=2.4", "numpy>=2.0", "scipy", "pandas", "matplotlib", "pygame"])
