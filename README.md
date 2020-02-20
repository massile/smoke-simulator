# Smoke simulator

A smoke simulator built from scratch.

Link to the associated video (click on the image):

![http://i3.ytimg.com/vi/PDHcTWfZfYs/maxresdefault.jpg](https://www.youtube.com/watch?v=PDHcTWfZfYs)

## Dependencies

In order to use this simulator you need to have [FFMPEG](https://www.ffmpeg.org) and [Nvidia CUDA](https://developer.nvidia.com/cuda-downloads) installed on your computer. Make sure that both FFMPEG and the NVCC compiler are available on your path.

## Build an run

To create an executable simplify run the following command

```
nvcc main.cu -O3 -o bin/smoke
```

Then you can run the executable using this command (on windows)

```
bin\smoke
```
