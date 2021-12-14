#!/bin/bash

content_dir="image_input"
style_dir="styles"

for f in ${content_dir}/*; do
    for style in ${style_dir}/*; do
        echo "$f-$style"
        content_filename=$(basename $f)
        content_dir=$(dirname $f)
        style_filename=$(basename $style)
        style_dir=$(dirname $style)
        img_name=$content_filename"-"$style_filename
        echo "$img_name"
        echo "$content_filename"
        echo "$style_filename"
        # echo "Rendering stylized image. This may take a while..."
        python neural_style.py \
        --img_name "$img_name" \
        --content_img "${content_filename}" \
        --content_img_dir "${content_dir}" \
        --style_imgs "${style_filename}" \
        --style_imgs_dir "${style_dir}" \
        --device '/cpu:0' \
        --verbose;
    done
done
# set -e
# # Get a carriage return into `cr`
# cr=`echo $'\n.'`
# cr=${cr%.}

# # Parse arguments
# content_image="$1"
# content_dir=$(dirname "$content_image")
# content_filename=$(basename "$content_image")

# style_image="$2"
# style_dir=$(dirname "$style_image" )
# style_filename=$(basename "$style_image")
# echo "Rendering stylized image. This may take a while..."
# python neural_style.py \
# --img_name "${content_filename}-${style_filename}" \
# --content_img "${content_filename}" \
# --content_img_dir "${content_dir}" \
# --style_imgs "${style_filename}" \
# --style_imgs_dir "${style_dir}" \
# --device '/gpu:0' \
# --verbose;