from setuptools import setup, find_packages

setup(
    name="grand-pattern-mono",
    version="0.1.0",
    description="Mono-vibe corrected architecture – one vibe, JEPA as weighted history, conservation by construction",
    author="SuperInstance",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    extras_require={"dev": ["pytest"]},
)
