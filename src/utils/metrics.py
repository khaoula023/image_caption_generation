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
from tqdm import tqdm
import evaluate
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from pycocoevalcap.cider.cider import Cider
from nltk.tokenize import word_tokenize as simple_word_tokenize

from src.logger import logging
from src.exception import CustomException


def bleu_score(gens, refs):
   
    # Compute BLEU score between generated captions and reference captions. keeps the highest (best) score for that specific caption.
   
    try:
        logging.info("Starting BLEU score computation")

        # Validate inputs
        if not isinstance(gens, list) or not isinstance(refs, list):
            raise ValueError("Predictions and references must be lists")

        if len(gens) != len(refs):
            raise ValueError("Predictions and references must have the same length")

        # Load BLEU metric
        bleu = evaluate.load("bleu")
        logging.info("BLEU metric loaded successfully")

        # Compute BLEU score
        result = bleu.compute(predictions=gens, references=refs)

        score = result["bleu"]
        logging.info("BLEU score computed successfully")

        return score

    except Exception as e:
        logging.error("Error occurred during BLEU score computation")
        raise CustomException(e, sys)
    
    

def meteor_score(gens, refs):
    # Compute METEOR score between generated caption and reference captions. keeps the highest (best) score for that specific caption.
    
    try:
        logging.info("Starting METEOR score computation")

        # Validate inputs
        if not isinstance(gens, list) or not isinstance(refs, list):
            raise ValueError("Predictions and references must be lists")
        
        if len(gens) != len(refs):
            raise ValueError("Predictions and references must have the same length")
        
        logging.info(f"Number of samples: {len(gens)}")

        # Load METEOR metric
        meteor = evaluate.load("meteor")
        logging.info("METEOR metric loaded successfully")

        # Compute METEOR score
        result = meteor.compute(predictions=gens, references=refs)
        score = result["meteor"]

        logging.info("METEOR score computed successfully")

        return score

    except Exception as e:
        logging.error("Error occurred during METEOR score computation")
        raise CustomException(e, sys)
    


def compute_clipscore(image, text, model_id="zer0int/LongCLIP-GmP-ViT-L-14"):
    # Compute CLIPScore for a given image and caption using LongCLIP.
   
    try:
        logging.info("Starting CLIPScore computation")

        # Input validation
        if image is None:
            raise ValueError("Image input is None")
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError("Text input must be a non-empty string")

        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
    

        # Load LongCLIP model
        logging.info("Loading LongCLIP model")
        config = CLIPConfig.from_pretrained(model_id)
        config.text_config.max_position_embeddings = 248  

        model = CLIPModel.from_pretrained(model_id, config=config)
        processor = CLIPProcessor.from_pretrained(model_id)
        model = model.to(device)
        model.eval()

        # Tokenizer max length
        max_len = processor.tokenizer.model_max_length
        text = text[:max_len]

        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=max_len
        ).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            image_emb = outputs.image_embeds[0]
            text_emb = outputs.text_embeds[0]

            # Normalize embeddings
            image_emb = image_emb / image_emb.norm()
            text_emb = text_emb / text_emb.norm()

        # Compute cosine similarity and scale
        score = (image_emb @ text_emb).item()
        final_score = max(0, score) * 100

        logging.info("CLIPScore computed successfully")
        return final_score

    except Exception as e:
        logging.error("Error occurred during CLIPScore computation")
        raise CustomException(e, sys)
    


def refclip_score(generated_text, reference_list, model_id="zer0int/LongCLIP-GmP-ViT-L-14"):
    
    # Compute RefCLIP score between a generated caption and a list of reference captions. Returns the average cosine similarity in [0, 100].
    try:
        logging.info("Starting RefCLIP computation")

        # Input validation
        if not isinstance(generated_text, str) or generated_text.strip() == "":
            raise ValueError("Generated text must be a non-empty string")
        if not isinstance(reference_list, list) or len(reference_list) == 0:
            raise ValueError("Reference list must be a non-empty list of strings")
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
    

        # Load LongCLIP model
        logging.info("Loading LongCLIP model")
        config = CLIPConfig.from_pretrained(model_id)
        config.text_config.max_position_embeddings = 248  

        model = CLIPModel.from_pretrained(model_id, config=config)
        processor = CLIPProcessor.from_pretrained(model_id)
        model = model.to(device)
        model.eval()
        max_len = processor.tokenizer.model_max_length

        def encode(text):
            # Tokenize and move to device
            tokens = processor.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_len
            ).to(device)

            with torch.no_grad():
                # Get text embedding
                text_outputs = model.text_model(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask
                )
                embedding = model.text_projection(text_outputs.pooler_output)
                embedding = embedding / embedding.norm(p=2)
            return embedding

        # Encode generated caption
        gen_emb = encode(generated_text)
        logging.info("Generated caption embedding computed")

        # Compute cosine similarity with each reference
        scores = []
        for idx, ref in enumerate(reference_list):
            ref_emb = encode(ref)
            score = (gen_emb @ ref_emb.T).item()
            scores.append(max(0, score) * 100)
            

        avg_score = sum(scores) / len(scores)
        logging.info("RefCLIP score computed successfully ")

        return avg_score

    except Exception as e:
        logging.error("Error occurred during RefCLIP computation")
        raise CustomException(e, sys)



def bert_score(gens, refs):
    """
    Compute BERTScore for Arabic captions between generated captions and references.
    Returns precision, recall, and F1 scores.
    """
    try:
        logging.info("Starting BERTScore computation")

        # Validate inputs
        if not isinstance(gens, list) or not isinstance(refs, list):
            raise ValueError("Predictions and references must be lists")
        if len(gens) != len(refs):
            raise ValueError("Predictions and references must have the same length")

        logging.info(f"Number of samples: {len(gens)}")

        # Load BERTScore metric
        bertscore = evaluate.load("bertscore")
        logging.info("BERTScore metric loaded successfully")

        # Compute scores
        result = bertscore.compute(
            predictions=gens,
            references=refs,
            lang="ar",  # Arabic
            model_type="bert-base-multilingual-cased"
        )

        precision = result['precision']
        recall = result['recall']
        f1 = result['f1']

        logging.info("BERTScore computation completed successfully")
        logging.debug(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

        return precision, recall, f1

    except Exception as e:
        logging.error("Error occurred during BERTScore computation")
        raise CustomException(e, sys)
    


def preprocess_arabic(text):
    """
    Tokenize Arabic text into words and join with spaces.
    """
    return ' '.join(simple_word_tokenize(str(text).strip()))


def compute_cider(caption_data, generated_col):
    """
    Compute CIDEr score for Arabic captions assuming exactly 3 references per caption.

    Args:
        caption_data (pd.DataFrame): DataFrame with columns: image, references , generated captions
        generated_col (str): Name of the column containing generated captions

    Returns:
        float:  Average CIDEr score
    """
    try:
        logging.info("Starting CIDEr computation ")

        preds = {}
        refs = {}

        for _, row in caption_data.iterrows():
            img_id = str(row['image'])

            # Preprocess generated caption from variable column
            generated_caption = row[generated_col]
            preds[img_id] = [preprocess_arabic(generated_caption)]

            # Preprocess exactly n references
            n = 3
            ref_list = row["Reference"]
            if not isinstance(ref_list, list) or len(ref_list) != n:
                raise ValueError(f"Image {img_id} must have exactly n references")
            refs[img_id] = [preprocess_arabic(r) for r in ref_list]

        # Compute CIDEr
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(refs, preds)

        logging.info("CIDEr score calculated successfully")
        return float(cider_score)

    except Exception as e:
        logging.error(f"Error occurred during CIDEr computation for {generated_col}")
        raise CustomException(e, sys)