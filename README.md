# Ray tracing code
Python code for ray tracing through realistic lenses given surface and material data. 
Optical systems and materials are extensible, so new types can be easily added. For convenience,
we include a perfect lens model as well. In addition to full physical ray tracing, we also
include various tools for paraxial system analysis, including generating ray transfer matrices
and computing 3rd order (Seidel) aberration coefficients. Higher-order aberration diagrams are also
easy to generate.

The [ray tracing code](src/raytrace/raytrace.py) can be pip installed by running
```
git clone https://github.com/QI2lab/ray_trace_pb.git
cd ray_trace_pb
pip install .
```
If you want to edit the package then install it in editable mode using instead
```
pip install -e .
```
The python package name is `raytrace` and the important components can be imported in your python interpreter using
```commandline
import raytrace.raytrace as rt
import raytrace.materials as rtm
```
# Scripts
Example scripts can be found in the [scripts](scripts) directory

# Documentation
Documentation is generated from function docstrings and built with Sphinx. e.g. navigate to [docs](docs) and run
```
make html
```
Then open `docs/_build/html/index.html` in your browser