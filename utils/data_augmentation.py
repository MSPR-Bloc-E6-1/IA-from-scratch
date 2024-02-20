import cv2
import os
import subprocess
import matplotlib.pyplot as plt
from imgaug.augmenters import contrast as iaa_contrast
from imgaug import augmenters as iaa
import numpy as np
import random
from tqdm import tqdm

def rotation(image, angle):
    """
    Performs rotation of the image by the specified angle.

    Parameters:
        image (numpy.ndarray): The image to augment.
        angle (float): The rotation angle in degrees.

    Returns:
        numpy.ndarray: The augmented image.
    """
    rotation_augmentor = iaa.Affine(rotate=angle)
    augmented_image = rotation_augmentor.augment_image(image)
    return augmented_image

def deformation(image, scale=(0.8, 1.2)):
    """
    Performs deformation of the image with a random scale.

    Parameters:
        image (numpy.ndarray): The image to augment.
        scale (tuple): Scale range for deformation.

    Returns:
        numpy.ndarray: The augmented image.
    """
    deformation_augmentor = iaa.Affine(scale=scale)
    augmented_image = deformation_augmentor.augment_image(image)
    return augmented_image

def mosaic(image, size=(10, 10)):
    """
    Creates a mosaic effect on the image.

    Parameters:
        image (numpy.ndarray): The image to augment.
        size (tuple): The size of the mosaic, in pixels.

    Returns:
        numpy.ndarray: The augmented image with the mosaic effect.
    """
    mosaic_augmentor = iaa.CoarseDropout(0.2, size_percent=0.2)
    augmented_image = mosaic_augmentor.augment_image(image)
    return augmented_image

def obfuscation(image, strength=0.2):
    """
    Darkens the image to add an obfuscation effect.

    Parameters:
        image (numpy.ndarray): The image to augment.
        strength (float): The strength of the obfuscation effect.

    Returns:
        numpy.ndarray: The augmented image with the obfuscation effect.
    """
    obfuscation_augmentor = iaa.Multiply((0.7, 1.3), per_channel=strength)
    augmented_image = obfuscation_augmentor.augment_image(image)
    return augmented_image

def superposition(image, overlay_image):
    """
    Overlays a second image on the main image.

    Parameters:
        image (numpy.ndarray): The main image.
        overlay_image (numpy.ndarray): The image to overlay.

    Returns:
        numpy.ndarray: The resulting image after the overlay.
    """
    superposition_augmentor = iaa.BlendAlpha(0.7, iaa.AllChannelsHistogramEqualization())
    overlay_augmented = superposition_augmentor.augment_image(overlay_image)
    augmented_image = cv2.addWeighted(image, 0.7, overlay_augmented, 0.3, 0)
    return augmented_image

def color_light_filter(image, hue=(-20, 20), contrast=(0.5, 1.5), brightness=(0.5, 1.5)):
    """
    Adds color filters and adjusts the brightness of the image.

    Parameters:
        image (numpy.ndarray): The image to augment.
        hue (tuple): Range of hue variation.
        contrast (tuple): Range of contrast variation.
        brightness (tuple): Range of brightness variation.

    Returns:
        numpy.ndarray: The augmented image with color filters and brightness adjustment.
    """
    filter_augmentor = iaa.Sequential([
        iaa.AddToHueAndSaturation(hue),
        iaa_contrast.LinearContrast(contrast),
        iaa.MultiplyBrightness(brightness)
    ])
    augmented_image = filter_augmentor.augment_image(image)
    return augmented_image

def add_blur(image, sigma=(0.5, 1.5)):
    """
    Adds Gaussian blur to the image.

    Parameters:
        image (numpy.ndarray): The image to augment.
        sigma (tuple): Range of standard deviation for the Gaussian blur kernel.

    Returns:
        numpy.ndarray: The augmented image with the Gaussian blur effect.
    """
    blur_augmentor = iaa.GaussianBlur(sigma=sigma)
    augmented_image = blur_augmentor.augment_image(image)
    return augmented_image

def affine_transformation(image, scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-45, 45)):
    """
    Applies an affine transformation to the image.

    Parameters:
        image (numpy.ndarray): The image to augment.
        scale (tuple): Scale factor of the transformation.
        translate_percent (tuple): Percentage of translation relative to the image dimensions.
        rotate (tuple): Rotation angle of the transformation.

    Returns:
        numpy.ndarray: The augmented image with the affine transformation.
    """
    affine_augmentor = iaa.Affine(scale=scale, translate_percent=translate_percent, rotate=rotate)
    augmented_image = affine_augmentor.augment_image(image)
    return augmented_image

def color_inversion(image):
    """
    Inverts the colors of the image.

    Parameters:
        image (numpy.ndarray): The image to augment.

    Returns:
        numpy.ndarray: The augmented image with color inversion.
    """
    inversion_augmentor = iaa.Invert(1.0, per_channel=True)
    augmented_image = inversion_augmentor.augment_image(image)
    return augmented_image

def add_noise(image, scale=(0, 0.1)):
    """
    Adds Gaussian noise to the image.

    Parameters:
        image (numpy.ndarray): The image to augment.
        scale (tuple): Scale of the Gaussian noise.

    Returns:
        numpy.ndarray: The augmented image with added Gaussian noise.
    """
    # Use a single value for scale instead of a range
    noise_augmentor = iaa.AdditiveGaussianNoise(scale=scale[1]*255)
    augmented_image = noise_augmentor.augment_image(image)
    return augmented_image


def random_rotation(image):
    """
    Performs a random rotation of the image.

    Parameters:
        image (numpy.ndarray): The image to augment.

    Returns:
        numpy.ndarray: The augmented image with a random rotation.
    """
    rotation_augmentor = iaa.Affine(rotate=(-180, 180))
    augmented_image = rotation_augmentor.augment_image(image)
    return augmented_image

def adjust_brightness(image, gamma=(0.5, 1.5)):
    """
    Adjusts the brightness of the image using gamma correction.

    Parameters:
        image (numpy.ndarray): The image to augment.
        gamma (tuple): Range of values for brightness adjustment.

    Returns:
        numpy.ndarray: The augmented image with brightness adjustment.
    """
    brightness_augmentor = iaa.GammaContrast(gamma=gamma)
    augmented_image = brightness_augmentor.augment_image(image)
    return augmented_image

def color_shift(image, rgb_shift=(-30, 30)):
    """
    Performs an RGB color shift on the image.

    Parameters:
        image (numpy.ndarray): The image to augment.
        rgb_shift (tuple): Range of RGB value shifts.

    Returns:
        numpy.ndarray: The augmented image with color shift.
    """
    shift_augmentor = iaa.AddToHue(value=rgb_shift[0])
    augmented_image = shift_augmentor.augment_image(image)
    return augmented_image

def elastic_distortion(image, alpha=(0, 40), sigma=5):
    """
    Applies elastic distortion to the image.

    Parameters:
        image (numpy.ndarray): The image to augment.
        alpha (tuple): Amplitude of the elastic distortion.
        sigma (int): Standard deviation of the Gaussian kernel for elastic distortion.

    Returns:
        numpy.ndarray: The augmented image with elastic distortion.
    """
    distortion_augmentor = iaa.ElasticTransformation(alpha=alpha, sigma=sigma)
    augmented_image = distortion_augmentor.augment_image(image)
    return augmented_image


def _random_zoom(image, zoom_factor=(1.0, 1.5)):
    zoom_augmentor = iaa.Affine(scale=zoom_factor)
    augmented_image = zoom_augmentor.augment_image(image)
    return augmented_image

def _random_crop(image, percent=(0.1, 0.2)):
    crop_augmentor = iaa.Crop(percent=percent)
    augmented_image = crop_augmentor.augment_image(image)
    return augmented_image

def _save(image_augmented, output_path, original_filename):
    """
    Saves the augmented image with a modified file name.

    Parameters:
        image_augmented (numpy.ndarray): The augmented image to save.
        output_path (str): The output folder path.
        original_filename (str): The original file name.

    Returns:
        None
    """
    filename_parts = original_filename.split('.')
    
    if len(filename_parts) > 1:
        base_name = '.'.join(filename_parts[:-1])
        extension = filename_parts[-1]
        augmented_filename = f"{base_name}_augmented.{extension}"
    else:
        augmented_filename = f"{original_filename}_augmented"

    augmented_file_path = os.path.join(output_path, augmented_filename)
    cv2.imwrite(augmented_file_path, image_augmented)


def _apply(image):
    augmentation_functions = [
        (rotation, (random.uniform(-180, 180),)),
        (deformation, ()),
        (mosaic, ()),
        (obfuscation, (random.uniform(0, 0.4),)),
        (superposition, (image,)),
        (color_light_filter, (random.uniform(-20, 20), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5))),
        (add_blur, (random.uniform(0.5, 1.5),)),
        (affine_transformation, ()),
        (color_inversion, ()),
        (add_noise, ((0, random.uniform(0, 0.1)),)),
        (random_rotation, ()),
        (adjust_brightness, (random.uniform(0.5, 1.5),)),
        (color_shift, ((random.randint(-30, 30), random.randint(-30, 30), random.randint(-30, 30)),)),
        (elastic_distortion, (random.uniform(0, 40),)),
        (_random_zoom, ()),
        (_random_crop, ()),
    ]

    # Apply a random modification to the image:
    index = np.random.randint(0, len(augmentation_functions))
    function, args = augmentation_functions[index]
    augmented_image = function(image, *args)

    return augmented_image

def debug(image, images_path):
    try :
        augmented_image = _apply(image)
    except :
        new_image = cv2.imread(images_path + "/" + random.choice(os.listdir(images_path)))
        augmented_image = _apply(new_image)
    return augmented_image
    
def _aug(images_path, quantity):
    for i in tqdm(range(quantity), desc="Augmenting images"):
        image_path = random.choice(os.listdir(images_path))
        while "_augmented" in image_path:
            image_path = random.choice(os.listdir(images_path))
        image_path = os.path.join(images_path, image_path)
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        try :
            augmented_image = _apply(image)
        except :
            new_image = cv2.imread(images_path + "/" + random.choice(os.listdir(images_path)))
            while "_augmented" in image_path:
                new_image = cv2.imread(images_path + "/" + random.choice(os.listdir(images_path)))
            augmented_image = debug(new_image, images_path)
        _save(augmented_image, images_path, image_name)


def data_augmentation(path, quantity):
    quantity = int(quantity/2) 
    for folder in ['background', 'beaver', 'cat', 'dog', 'coyote', 'squirrel', 'rabbit', 'wolf', 'lynx', 'bear', 'puma', 'rat', 'raccoon', 'fox']:
        folder_path = os.path.join(path, folder)
        _aug(folder_path, quantity)
