import os
import cv2
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from nst_utils import *
import numpy as np
import tensorflow as tf
import argparse


STYLE_LAYERS  = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

style_list = ["composition", "monet", "picasso", "seated_nude", "stone", "van_gogh"]

parser = argparse.ArgumentParser()
parser.add_argument("--all", help="run for some predefined style images")
parser.add_argument("--style", default = "composition", help="put name of style image you want to use without extension " )
parser.add_argument("--content", default = "louvre_small", help="put name of content image you want to use without extension " )
parser.add_argument("--video", default = None, help="put name of video to render")
args = parser.parse_args()


def compute_content_cost(a_C, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C, [-1, n_H * n_W , n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [-1, n_H * n_W , n_C]))
    
    J_content = tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled))/(4 * m * n_H * n_W * n_C)
    
    return J_content

def gram_matrix(A):

    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)))/(4 * (n_C * n_H * n_W)**2)
    
    
    return J_style_layer

def compute_style_cost(model, sess,  STYLE_LAYERS):

    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        out = model[layer_name]
        a_S = sess.run(out)

        a_G = out

        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):

    J = alpha * J_content + beta * J_style
    
    return J

def model_nn(model, sess, input_image, train_step, style_string, content_string = None, num_iterations = 200, video = False):
    
    sess.run(tf.global_variables_initializer())
    sess.run(model["input"].assign(input_image))
    
    for i in range(num_iterations):

        _ = sess.run(train_step)

        generated_image = sess.run(model["input"])

        if i%20 == 0 and not video:
            # Jt, Jc, Js = sess.run([J, J_content, J_style])
            # print("Iteration " + str(i) + " :")
            # print("total cost = " + str(Jt))
            # print("content cost = " + str(Jc))
            # print("style cost = " + str(Js))
            
            save_image("output/{0}/{0}_{1}_{2}.png".format(style_string, content_string, str(i)), generated_image)
    if not video:
        save_image('output/{0}/generated_image_{0}_{1}.jpg'.format(style_string, content_string), generated_image)
    
    return generated_image

def on_video():

    style_image = scipy.misc.imread("images/style_{}.jpg".format(style_string))
    style_image = reshape_and_normalize_image(style_image)


    if not os.path.exists(os.path.join("output/", style_string)):
        os.makedirs(os.path.join("output/", style_string))

    cap = cv2.VideoCapture("videos/{}.mp4".format(video_string))
    print("video opened")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_string = "output/{0}/{0}_{1}.avi".format(style_string, video_string)
    output = cv2.VideoWriter(out_string, fourcc, 20.0, (400, 300))

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            tf.reset_default_graph()
            sess = tf.InteractiveSession()
            model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

            frame = cv2.resize(frame, (400, 300))
            content_image = reshape_and_normalize_image(frame)
            generated_image = generate_noise_image(content_image)

            sess.run(model['input'].assign(content_image))

            out = model['conv4_2']
            a_C = sess.run(out)
            a_G = out

            J_content = compute_content_cost(a_C, a_G)

            sess.run(model['input'].assign(style_image))
            J_style = compute_style_cost(model, sess, STYLE_LAYERS)
            
            J = total_cost(J_content, J_style, alpha = 10, beta = 80)

            optimizer = tf.train.AdamOptimizer(2.0)
            train_step = optimizer.minimize(J)

            generated_image = model_nn(model, sess, generated_image, train_step, style_string, content_string, num_iterations = 200, video = True)

            # Un-normalize the image so that it looks good
            generated_image = generated_image + CONFIG.MEANS
    
            # Clip and Save the image
            generated_image = np.clip(generated_image[0], 0, 255).astype('uint8')

            output.write(generated_image)
            print("*", end = "")
            sess.close()
        else:
            break

    print("video saved")
    cap.release()
    output.release()

def on_image(style_string, content_string):
    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

    style_image = scipy.misc.imread("images/style_{}.jpg".format(style_string))
    style_image = reshape_and_normalize_image(style_image)

    if not os.path.exists(os.path.join("output/", style_string)):
        os.makedirs(os.path.join("output/", style_string))

    content_image = scipy.misc.imread("images/{}.jpg".format(content_string))
    content_image = reshape_and_normalize_image(content_image)

    generated_image = generate_noise_image(content_image)

    sess.run(model['input'].assign(content_image))

    out = model['conv4_2']

    a_C = sess.run(out)

    a_G = out

    J_content = compute_content_cost(a_C, a_G)

    sess.run(model['input'].assign(style_image))

    J_style = compute_style_cost(model, sess, STYLE_LAYERS)

    J = total_cost(J_content, J_style, alpha = 10, beta = 80)

    optimizer = tf.train.AdamOptimizer(2.0)

    train_step = optimizer.minimize(J)

    generated_image = model_nn(model, sess, generated_image, train_step, style_string, content_string, num_iterations = 400)


if __name__ == '__main__':

    if args.all:
        for style_string in style_list:
            on_image(style_string, args.content)
            
    elif args.video:
        style_string = args.style
        video_string = args.video
        on_video(style_string, video_string)

    else:
        style_string = args.style
        content_string = args.content
        on_image(style_string, content_string)
