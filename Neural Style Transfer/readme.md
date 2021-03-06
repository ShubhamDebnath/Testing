# Art Generation
### with Neural Style Transfer

### Prequisites
##### Download the VGG-19 learned matrix from [here](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) and put it inside pretrained-model folder
##### Download your own style images and please resize them to 400 x 300 from [here](https://www.reduceimages.com/)
##### Rename style images to "style_[name of style]" like [here](https://github.com/ShubhamDebnath/Testing/tree/master/Neural%20Style%20Transfer/images) and put them inside images folder
##### Download your own content images and please resize them to 400 x 300 from [here](https://www.reduceimages.com/)
##### Keep the name of content image short and without spaces and put them inside images folder

### Usage
`usage: art_generation.py [-h] [-a ALL] [-s STYLE] [-c CONTENT]`

`optional arguments:`
`  -h, --help            show this help message and exit`

`  --all ALL             run for some predefined style images`

`  --style STYLE        put name of style image you want to use without extension`

`   --content CONTENT     put name of content image you want to use without extension`
##### Example
`python art_generation.py --content eva`

`python art_generation.py --all`

### Output
##### Output images will be saved in output folder under respestive folders of each style

### Sample Results
![](images/louvre_small.jpg?raw=true)
![](output/monet/generated_image_monet.jpg?raw=true)
![](output/van_gogh/generated_image_van_gogh_louvre_small.jpg?raw=true)
![](output/stone/generated_image_stone_louvre_small.jpg?raw=true)
