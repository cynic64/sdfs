# Raymarching Signed Distance Fields
For now, only includes the code for the traditional Mandelbulb fractal. It runs at about 40 FPS on
my GTX 1060 with sample shading, which is not great but good enough for experimenting about.

Credit goes to Inigo Quilez for the [underlying algorithm](https://iquilezles.org/www/articles/mandelbulb/mandelbulb.htm).

[Video demo](https://www.youtube.com/watch?v=PdOSO2qhIQ8)

# Cloning and Compiling
The project includes `render-c` as a submodule, so clone with submodules:
```
git clone --recurse submodules https://github.com/cynic64/sdfs
cd sdfs/
make
./main
```
However, I give no guarantee that it actually works. Good luck!
