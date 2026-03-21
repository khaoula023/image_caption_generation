import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import requests
from PIL import Image
from io import BytesIO
import arabic_reshaper
from bidi.algorithm import get_display
import textwrap
from math import ceil
import time
import re
from camel_tools.utils.normalize import (
    normalize_unicode,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
    normalize_alef_ar
)


from src.logger import logging
from src.exception import CustomException

def show(images, captions, max_chars=40):
    # Display a set of images with their corresponding Arabic captions.
     try:
        logging.info("Starting visualization of images and captions")

        # Validate inputs
        if len(images) != len(captions):
            raise ValueError("Number of images and captions must be equal")
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

        if n == 1:
            axes = [axes]

        for ax, img, caption in zip(axes, images, captions):
            ax.imshow(img)
            ax.axis("off")
            
            # Reshape and fix Arabic text
            reshaped_text = arabic_reshaper.reshape(caption)
            bidi_text = get_display(reshaped_text)
            
            # Wrap long text into multiple lines
            wrapped_lines = textwrap.wrap(bidi_text, max_chars)
            
            # Reverse the order of lines so they show correctly
            wrapped = "\n".join(reversed(wrapped_lines))
            
            # Place caption under the image
            ax.set_title(wrapped, fontsize=12, fontname="Arial", pad=20)

        plt.tight_layout()
        plt.show()
        logging.info("Visualization completed successfully")
     except Exception as e:
        logging.error("Error occurred during visualization")
        raise CustomException(e, sys)

def clean_intro(text):
    # Clean Arabic captions by removing common introductory phrases and leading punctuation.
    try:
        logging.info("Starting text cleaning")

        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        original_text = text

        # Remove leading punctuation/spaces
        text = re.sub(r"^[\s:\-،]*", "", text)

        # Common intro patterns to remove
        patterns = [
            r"^(?:بالتأكيد(?:،)?(?:\s*إليك(?:\s*وصف(?:\s*للصورة)?)?)?)[:：\s,-]*",
            r"^(?:في(?:\s+هذه)?\s+الصورة)[:：\s،,-]*",
            r"^(?:تُظهر|يظهر|تُظهر|تبدوا|تبدو)[:：\s،,-]*(?:الصورة)?\s*",
            r"^(?:الصورة(?:\s+تظهر|\s+تُظهر))[:：\s،,-]*",
            r"^(?:في\s+الصوره)[:：\s،,-]*",
            r"^(?:يبدو\s+أن(?:\s+ال)?\s*الصورة)[:：\s،,-]*",
            r"^(?:في\s+الصور?ة)[:：\s،,-]*",
            r"(?:في\s+جملة\s+قصيرة[:：\s،,-]*)",
            r"(?:بجملة\s+قصيرة[:：\s،,-]*)",
            r"^(?:الصورة\s+تصور)[:：\s،,-]*",
            r"^(?:الصورة\s+تظهر)[:：\s،,-]*",
            r"^(?:يظهر\s+في\s+الصورة)[:：\s،,-]*",
            r"^(?:الصورة\s+تبرز)[:：\s،,-]*",
            r"^(?:الصورة\s+تعرض)[:：\s،,-]*",
            r"^(?:يمكن\s+رؤية)[:：\s،,-]*",
            r"^(?:الصورة\s+توضح)[:：\s،,-]*",
            r"^(?:تبدو\s+الصورة)[:：\s,-]*",
            r"^(?:الصورة\s+تحتوي\s+علي)[:：\s,-]*",
            r"^(?:تُصور\s+الصورة)[:：\s,-]*",
            r"^(?:الصوره\s+تصور)[:：\s,-]*",
            r"^(?:تُ)\s*"
        ]

        # Apply all patterns
        for p in patterns:
            text = re.sub(p, "", text)

        cleaned_text = text.strip()

        logging.info(f"Text cleaned successfully")

        return cleaned_text

    except Exception as e:
        logging.error("Error occurred during text cleaning")
        raise CustomException(e, sys)
    


def preprocess_caption(text):
   # Preprocess Arabic captions by cleaning, normalizing text,and removing unwanted characters for consistent evaluation.
    try:
        logging.info("Starting caption preprocessing")

        if text is None:
            raise ValueError("Input text is None")

        # Convert to string and remove newlines
        text = str(text).replace("\n", " ").replace("\\n", " ").strip()
        
        # Remove common intro phrases
        text = clean_intro(text)

        # Remove Arabic diacritics 
        text = re.sub(r'[\u064B-\u0652]', '', text)

        # Normalize Arabic letters
        text = normalize_unicode(text)
        text = normalize_alef_ar(text)
        text = normalize_alef_maksura_ar(text)
        text = normalize_teh_marbuta_ar(text)


        # Keep only Arabic characters and spaces
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)

        processed_text = text.strip()

        logging.info("Caption preprocessing completed successfully")

        return processed_text

    except Exception as e:
        logging.error("Error occurred during caption preprocessing")
        raise CustomException(e, sys)
    

def get_existing_images(images_folder, data):
    # Return a list of filenames from `data` that actually exist in the given folder.
    try:
        logging.info(f"Checking existing images in folder: {images_folder}")

        # Input validation
        if not os.path.exists(images_folder):
            raise ValueError(f"Images folder does not exist: {images_folder}")
        if not isinstance(data, list):
            raise ValueError("Data must be a list of filenames")

        # Get all filenames that actually exist in the folder
        folder_files = set(os.listdir(images_folder))

        # Keep only filenames that match
        existing_images = [fname for fname in data if fname in folder_files]
        logging.info(f"{len(existing_images)} files matched from data list")

        return existing_images

    except Exception as e:
        logging.error("Error occurred while checking existing images")
        raise CustomException(e, sys)