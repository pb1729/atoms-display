from setuptools import setup

setup(
  name="atoms_display",
  py_modules=["atoms_display"],
  install_requires=["numpy==1.26.3", "pyopengl", "PyOpenGL_accelerate"],
  # Metadata:
  version="0.2",
  description="display a rotatable 3d visualization of molecules with the ability to animate dynamic changes",
  author="pb1729",
  url="https://github.com/pb1729/atoms-display")



