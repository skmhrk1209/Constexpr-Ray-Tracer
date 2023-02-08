# Constexpr Ray Tracer

This repository provides a C++ program that renders a predefined scene at compile time using constexpr functions and just saves the rendered image at run time. Simple path tracing and sphere tracing have been implemented as rendering methods, but I'm aiming to implement more rigorous physically based rendering in the future.

## Separate Rendering

Since rendering a high-resolution image at once would cause the compiler to eat up memory, we support separate rendering that splits all the pixels into small-sized chunks and renders each one. It is implemented by instantiating a templated rendering function that takes an area to be rendered as a template argument and compiling each translation unit.

I provide a python script that takes the size of the image to be rendered and the number of pixels in each chunk as inputs and automatically generates source files each of which instantiates the function template with the corresponding area. The usage of the script is as follows:

```bash
usage: instantiate.py [-h] [-T TYPE] [-W WIDTH] [-H HEIGHT] [-N CHUNKS]

Automatic Instantiation for Separate Rendering

optional arguments:
  -h, --help                  show this help message and exit
  -T TYPE, --type TYPE        Floating-point type used for rendering
  -W WIDTH, --width           WIDTH Width of the image to be rendered
  -H HEIGHT, --height HEIGHT  Height of the image to be rendered
  -N CHUNKS, --chunks CHUNKS  Number of chunks all the pixels are split into
```

The *CMakeLists.txt* is also re-generated based on the exsistent *.CMakeLists.txt* along with the auto-generation of source files.
