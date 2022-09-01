import setuptools

name = "sy2jax"
version = "0.0.1"
author = "jonkhler"
author_email = "https://twitter.com/jonkhler"
description = "sympy to jax transpiler"
readme = "TODO"  #  TODO
url = "TODO"  #  TODO
license = "MIT"
python_requires = "~=3.10"
install_requires = ["jax>=0.3.4", "sympy>=1.7.1"]

setuptools.setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=url,
    license=license,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    packages=[name],
)
