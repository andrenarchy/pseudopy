# PseudoPy

PseudoPy computes and visualizes the pseudospectrum of a matrix. It is a Python version of the original [eigtool](http://www.cs.ox.ac.uk/pseudospectra/eigtool/) by Thomas G. Wright. The algorithms used in this package can be found in the book [Spectra and pseudospectra](http://press.princeton.edu/titles/8113.html) by [Nick Trefethen](http://www.maths.ox.ac.uk/people/profiles/nick.trefethen) and [Mark Embree](http://www.caam.rice.edu/~embree/).

## Example
The pseudospectrum of the Grcar matrix looks like this:

![Pseudospectrum of Grcar matrix](grcar.png)

The above figure can be created by running
```python
from pseudopy import demo
demo.grcar_demo()
```
and the corresponding code of `grcar_demo` can be found in [pseudopy/demo.py](pseudopy/demo.py).
