# CNN Style Transfer

The algorithm is proposed in [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge.

####Prerequisites

Python3
Tensorflow==1.15

####Usage

1. Download a VGG-19 model from  [VGG-19 model weights](http://www.vlfeat.org/matconvnet/pretrained/) into this directory.

2. Add input images into ./image_input and the style images into ./styles

3. Terminal commands

   ~~~
   bash stylize_image.sh
   ~~~

4. Find the result in ./image_output. The corresponding result for 1.jpg as input and 3.jpg as style is named as 1.jpg-3.jpg

5. Or directly run neural_style.py to generate results.

   ~~~
   python neural_style.py --content_img 1.jpg \
                          --style_imgs starry-1.jpg \
                          --max_size 1000 \
                          --max_iterations 100 \
                          --original_colors \
                          --device /cpu:0 \
                          --verbose;
   ~~~

   

## License
The codes are adopted and arranged for clarity from the following resource:

1. https://github.com/cysmith/neural-style-tf
