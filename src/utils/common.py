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
    
def load_image(img_path):
    """
    Load an image from disk and convert to RGB.
    """
    try:
        if not os.path.exists(img_path):
            logging.warning("Image not found")
            return None
        image = Image.open(img_path).convert("RGB")
        return image
    except Exception as e:
        logging.error("Error loading image ")
        raise CustomException(e, sys)

yolo_to_arabic = {
    # 🔹 People & body
    "person": "شخص",
    "man": "رجل",
    "woman": "امرأة",
    "child": "طفل",
    "baby": "رضيع",
    "face": "وجه",
    "hand": "يد",
    "foot": "قدم",
    "eye": "عين",
    "ear": "أذن",
    "mouth": "فم",
    "hair": "شعر",
    "beard": "لحية",

    # 🔹 Animals
    "cat": "قط",
    "dog": "كلب",
    "horse": "حصان",
    "sheep": "خروف",
    "cow": "بقرة",
    "elephant": "فيل",
    "bear": "دب",
    "zebra": "حمار وحشي",
    "giraffe": "زرافة",
    "lion": "أسد",
    "tiger": "نمر",
    "monkey": "قرد",
    "bird": "طائر",
    "parrot": "ببغاء",
    "chicken": "دجاجة",
    "duck": "بطة",
    "fish": "سمكة",
    "frog": "ضفدع",
    "snake": "ثعبان",

    # 🔹 Vehicles
    "bicycle": "دراجة",
    "motorcycle": "دراجة نارية",
    "scooter": "سكوتر",
    "car": "سيارة",
    "bus": "حافلة",
    "train": "قطار",
    "truck": "شاحنة",
    "boat": "قارب",
    "ship": "سفينة",
    "airplane": "طائرة",
    "helicopter": "مروحية",
    "taxi": "تاكسي",
    "subway": "مترو",
    "van": "فان",

    # 🔹 Household / furniture
    "bed": "سرير",
    "chair": "كرسي",
    "couch": "أريكة",
    "sofa": "كنبة",
    "table": "طاولة",
    "dining table": "طاولة طعام",
    "bookshelf": "رف كتب",
    "wardrobe": "خزانة",
    "desk": "مكتب",
    "drawer": "دولاب",
    "lamp": "مصباح",
    "fan": "مروحة",
    "mirror": "مرآة",
    "window": "نافذة",
    "door": "باب",
    "curtain": "ستارة",
    "carpet": "سجادة",
    "pillow": "وسادة",
    "blanket": "بطانية",
    "vase": "مزهرية",

    # 🔹 Electronics
    "tv": "تلفاز",
    "laptop": "حاسوب محمول",
    "desktop": "حاسوب مكتبي",
    "mouse": "فأرة",
    "keyboard": "لوحة مفاتيح",
    "remote": "جهاز تحكم",
    "cell phone": "هاتف محمول",
    "camera": "كاميرا",
    "headphones": "سماعات",
    "microphone": "ميكروفون",
    "speaker": "مكبر صوت",
    "watch": "ساعة يد",

    # 🔹 Food & drinks
    "apple": "تفاح",
    "banana": "موز",
    "orange": "برتقال",
    "grapes": "عنب",
    "watermelon": "بطيخ",
    "strawberry": "فراولة",
    "lemon": "ليمون",
    "bread": "خبز",
    "cheese": "جبن",
    "egg": "بيض",
    "milk": "حليب",
    "coffee": "قهوة",
    "tea": "شاي",
    "juice": "عصير",
    "pizza": "بيتزا",
    "cake": "كعكة",
    "donut": "دونات",
    "sandwich": "سندويتش",
    "hot dog": "هوت دوغ",
    "bottle": "زجاجة",
    "cup": "كوب",
    "fork": "شوكة",
    "knife": "سكين",
    "spoon": "ملعقة",
    "bowl": "وعاء",
    "wine glass": "كأس نبيذ",

    # 🔹 Clothes & accessories
    "hat": "قبعة",
    "cap": "طاقية",
    "shirt": "قميص",
    "t-shirt": "تي شيرت",
    "pants": "بنطال",
    "jeans": "جينز",
    "dress": "فستان",
    "skirt": "تنورة",
    "shoes": "أحذية",
    "socks": "جوارب",
    "belt": "حزام",
    "tie": "ربطة عنق",
    "gloves": "قفازات",
    "scarf": "وشاح",
    "bag": "حقيبة",
    "handbag": "حقيبة يد",
    "backpack": "حقيبة ظهر",
    "ring": "خاتم",
    "necklace": "عقد",
    "bracelet": "سوار",

    # 🔹 Nature / outdoors
    "tree": "شجرة",
    "flower": "زهرة",
    "grass": "عشب",
    "bush": "شجيرة",
    "rock": "صخرة",
    "mountain": "جبل",
    "river": "نهر",
    "lake": "بحيرة",
    "beach": "شاطئ",
    "sand": "رمل",
    "sky": "سماء",
    "sun": "شمس",
    "moon": "قمر",
    "cloud": "سحابة",
    "star": "نجمة",
    "rain": "مطر",
    "snow": "ثلج",
    "leaf": "ورقة شجر",

    # 🔹 Sports / toys
    "ball": "كرة",
    "basketball": "كرة سلة",
    "soccer ball": "كرة قدم",
    "tennis racket": "مضرب تنس",
    "frisbee": "قرص طائر",
    "skateboard": "لوح تزلج صغير",
    "trampoline": "ترامبولين",
    "swing": "أرجوحة",
    "slide": "زلاجة",
    "doll": "دمية",
    "puzzle": "لغز",
    "kite": "طائرة ورقية",
    "baseball bat": "مضرب بيسبول",
    "baseball glove": "قفاز بيسبول",

    # 🔹 Office / study
    "book": "كتاب",
    "notebook": "دفتر",
    "pen": "قلم",
    "pencil": "قلم رصاص",
    "eraser": "ممحاة",
    "ruler": "مسطرة",
    "scissors": "مقص",
    "stapler": "دباسة",
    "paper": "ورق",
    "folder": "مجلد",
    "calculator": "آلة حاسبة",

    # 🔹 Miscellaneous
    "umbrella": "مظلة",
    "flag": "علم",
    "wallet": "محفظة",
    "key": "مفتاح",
    "toy": "لعبة",
    "clock": "ساعة",
    "bottle cap": "غطاء زجاجة",
    "plastic bag": "كيس بلاستيك",
    "chair mat": "حافظة كرسي",
    "candle": "شمعة",
    "poster": "ملصق",
    "picture": "صورة",
    "phone charger": "شاحن هاتف",
    "remote control": "جهاز تحكم",
    "glasses": "نظارات",
    "sunglasses": "نظارة شمس",
    "towel": "منشفة"
}

arabic_to_yolo = {normalize_arabic(v): k for k, v in yolo_to_arabic.items()}

def normalize_arabic(text):
    """
    Normalize Arabic text for NLP tasks (e.g., object matching).

    Steps:
    - Remove diacritics (tashkeel)
    - Normalize different letter forms
    - Remove non-Arabic characters (punctuation, numbers, symbols)
    - Remove definite article "ال"
    - Clean extra spaces
    - Convert to lowercase
    """
    try:
        # 🔹 Define Arabic diacritics (tashkeel) to remove
        diacritics = re.compile("""
            ّ | َ | ً | ُ | ٌ | ِ | ٍ | ْ | ـ
        """, re.VERBOSE)

        # 🔹 Remove diacritics (e.g., َ ً ُ → improves matching consistency)
        text = re.sub(diacritics, '', text)
        
        # 🔹 Normalize different forms of letters to a standard form
        text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        
        # Normalize variants of yaa and waw
        text = text.replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و")
        
        # Normalize taa marbuta → haa (helps unify word endings)
        text = text.replace("ة", "ه")
        
        # 🔹 Remove everything that is NOT Arabic letters or spaces
        text = re.sub(r"[^ء-ي\s]", " ", text)
        
        # 🔹 Remove the definite article "ال" (prefix)
        text = re.sub(r"\bال", "", text)
        
        # 🔹 Remove extra spaces created during cleaning
        text = re.sub(r"\s+", " ", text).strip()
        
        # 🔹 Convert to lowercase (safe after normalization)
        return text.lower()
    except Exception as e:
        logging.error("Error Normalize Arabic text ")
        raise CustomException(e, sys)

    
def simple_singularize(word):
    try:
        if word.endswith('s') and len(word) > 3:
            return word[:-1]
        return word
    except Exception as e:
        logging.error("Error simple singularize words")
        raise CustomException(e, sys)

def extract_visual_objects(text, arabic_to_yolo):
    try: 
        if not isinstance(text, str) or text.strip() == "":
            return []

        text = normalize_arabic(str(text)).lower()

        # Safer tokenization for Arabic (avoid punkt issues)
        words = re.findall(r'\w+', text)

        objects = []
        for word in words:
            lemma = simple_singularize(word)

            if lemma in arabic_to_yolo:
                objects.append(arabic_to_yolo[lemma])

        return list(set(objects)) 
    except Exception as e:
        logging.error("Error extracting visual objects ")
        raise CustomException(e, sys)
    
    