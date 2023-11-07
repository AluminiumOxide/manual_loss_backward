**для того, чтобы 让这个项目更加 in line with インターネット の Son Sürüm**

the following content will be written by crappy machine translation.

## Whine

This project calculates the gradient manually, further explains the underlying calculation process when using `loss.bakcward()` in `pytorch`

You can use above routines to see how to obtain the grad of weight in `linear` 、 `conv2d` manually.

To run the above routines, you need to make sure that the following libraries installed in your environment.

```
conda create -n manual_back python=3.5
conda activate manual_back
pip install graphviz
pip install torch torchvision torchaudio
```

Similar to `ffmpeg`, `graphviz` is also a separate software, the third-party library only provides an interface to use the software through python, you also need to go to the [official website](https://graphviz.org/download/) to download and install.

Or you can comment `draw_forward()` and the related plot code (you can also see all results on the command line)

For detail explanation, please visit [AluminiumOxide@bilibili]( https://www.bilibili.com/video/BV1ua4y1Q7La/)

enjoy