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
import re
import json
import torch
import traceback

from src.utils import common
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
    


def build_clair_prompt(candidate, references):
    """
    Build CLAIR-style prompt for one candidate caption and its references.
    Captions are in Arabic, prompt in English.
    """
    try:
        candidate_block = f"- {candidate}\n"
        reference_block = "".join([f"- {r}\n" for r in references])

        prompt = f"""\
            You are trying to tell if a candidate set of captions is describing the same image as a reference set of captions.

            The captions are written in Arabic.

            Candidate set:
            {''.join(candidate_block)}

            Reference set:
            {''.join(reference_block)}

            On a precise scale from 0 to 100, how likely is it that the candidate set is describing the same image as the reference set?

            Return ONLY valid JSON with:
            {{"score": number between 0 and 100, "reason": "short explanation"}}
            """
        return prompt

    except Exception as e:
        logging.error(f"Error building CLAIR prompt: {e}")
        raise CustomException(e, sys)


def parse_clair_response(response):
    """
    Parse CLAIR model response to extract 'score' and 'reason'.
    """
    try:
        # Try to parse JSON (if complete)
        match = re.search(r"\{.*\"score\".*?\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = float(data.get("score", 0))
            reason = data.get("reason", "Unknown")
            return score, reason

        # Fallback: extract first occurrence of "score": number
        score_match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response)
        score = float(score_match.group(1)) if score_match else 0.0

        # Extract reason if possible
        reason_match = re.search(r'"reason"\s*:\s*"(.*?)"', response, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else "Parsed from text"

        return score, reason

    except Exception as e:
        logging.error(f"Error parsing CLAIR response: {e}\nResponse was: {response}")
        raise CustomException(e, sys)


def batch_clair_score(gens, refs, model, processor, device,  max_new_tokens=80):
    """
    Compute CLAIR scores for a batch (3 references per caption).

    Args:
        gens (list[str]): generated captions
        refs (list[list[str]]): list of 3 references per caption

    Returns:
        avg_score (float): average CLAIR score (0–1)
        scores (list): individual normalized scores
        reasons (list): explanations
    """
    try:
        logging.info("Starting batch CLAIR computation")

        # Validation
        if not gens or not refs:
            raise ValueError("gens and refs cannot be empty")

        if len(gens) != len(refs):
            raise ValueError("gens and refs must have same length")

        if not all(isinstance(r, list) and len(r) == 3 for r in refs):
            raise ValueError("Each caption must have exactly 3 references")

        # Build prompts
        prompts = [build_clair_prompt(gen, ref_list) for gen, ref_list in zip(gens, refs)]
        logging.info(f"Built {len(prompts)} prompts for CLAIR evaluation")

        # Tokenize batch
        inputs = processor(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        responses = processor.batch_decode(outputs, skip_special_tokens=True)
        logging.debug(f"Raw CLAIR responses: {responses}")

        scores = []
        reasons = []

        for response in responses:
            score, reason = parse_clair_response(response)
            scores.append(score / 100)  # normalize to [0,1]
            reasons.append(reason)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        logging.info(f"Batch CLAIR average score: {avg_score:.4f}")

        return avg_score, scores, reasons

    except Exception as e:
        logging.error(f"Error in batch CLAIR computation: {e}")
        raise CustomException(e, sys)



def build_fleur_prompt(caption):
    """
    Build a FLEUR prompt for a single image caption.
    """
    try:
        if not isinstance(caption, str):
            raise ValueError(f"Caption must be a string, got {type(caption)}")

        prompt = f"""<image>

            You are an image caption evaluator. Your job is to rate how accurately the given caption describes the image content.
            Caption:
            "{caption}"

            Respond ONLY once and EXACTLY in the following format:
            Score: <float number between 0.00 and 10.00 with two decimal digits>
            Explanation (in Arabic): <brief justification in Arabic explaining WHY this score was given>

            Write the explanation in Modern Standard Arabic (الفصحى), without any English words or repetition of the caption.
            """.strip()

        logging.info("FLEUR prompt built successfully for caption..")  # log first 30 chars
        return prompt

    except Exception as e:
        logging.error("Error building FLEUR prompt for caption: ", exc_info=True)
        raise CustomException(e, sys)


def parse_fleur_output(response):
    """
    Safe parsing function with extensive error handling
    """
    try:
        # Check if response is None or empty
        if response is None:
            #print("Response is None")
            return None, None
            
        if not isinstance(response, str):
            #print(f"Response is not string: {type(response)}")
            return None, None
            
        response = response.strip()
        if not response:
            print("Response is empty string")
            return None, None
        
       # print(f"Parsing response: '{response}'")
        
        # Look for score pattern
        score_pattern = r'Score:\s*(\d+\.\d{2}|\d+\.\d|\d+)'
        score_match = re.search(score_pattern, response)
        
        if not score_match:
            #print("No score pattern found")
            return None, None
        
        score_text = score_match.group(1)
        try:
            score = float(score_text)
        except ValueError:
            #print(f"Could not convert score to float: '{score_text}'")
            return None, None
        
        # Validate score range
        if not (0.0 <= score <= 10.0):
            #print(f"Score out of range: {score}")
            return None, None
        
        # Extract explanation - look for Arabic text after score
        explanation = "لا توجد شرح"  # Default
        
        # Method 1: Look for explicit explanation marker
        explanation_patterns = [
            r'Explanation\s*\(in Arabic\):\s*(.*)',
            r'Explanation:\s*(.*)',
            r'شرح:\s*(.*)'
        ]
        
        for pattern in explanation_patterns:
            exp_match = re.search(pattern, response, re.DOTALL)
            if exp_match:
                explanation = exp_match.group(1).strip()
                break
        else:
            # Method 2: Extract text after score (assuming it's Arabic)
            # Split response after the score match
            after_score = response[score_match.end():].strip()
            if after_score:
                # Take the first line or reasonable chunk
                explanation = after_score.split('\n')[0].strip()
                if not explanation or len(explanation) < 5:  # Too short
                    explanation = "لا توجد شرح"
        
        #print(f"Successfully parsed - Score: {score}, Explanation: {explanation}")
        return score, explanation
        
    except Exception as e:
        #print(f"Error in parse_fleur_output: {e}")
        traceback.print_exc()
        return None, None

def generate_fleur_response(image, caption, model, processor, device, max_new_tokens=100 ):
    """
    Generate FLEUR response for a single image and caption.
    """
    try:
        # Build prompt
        prompt = build_fleur_prompt(caption)

        # Conversation format for the model:
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Prepare inputs
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

        # Generate model outputs
        with torch.inference_mode(), torch.autocast("cuda"):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        # Decode response
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Remove prompt from response if included
        if text in response:
            response = response.replace(text, "").strip()

        # Parse FLEUR output
        score, explanation = parse_fleur_output(response)

        # Clean up memory
        del inputs
        torch.cuda.empty_cache()

        return score, explanation

    except Exception as e:
        logging.error(f"Error generating FLEUR response: {e}", exc_info=True)
        return None, None


def compute_fleur_scores(df, images_paths_full, col_name):
    """
    Compute FLEUR scores for a dataframe of captions and corresponding images.

    Args:
        df (pd.DataFrame): dataframe with caption column
        images_paths_full (list): list of full image paths
        col_name (str): name of the column to use for captions

    Returns:
        fleur_scores (list)
        fleur_explanations (list)
    """
    fleur_scores = []
    fleur_explanations = []

    for i in tqdm(range(len(df)), desc=f"FAST FLEUR ({col_name})"):
        img_path = images_paths_full[i]
        caption = df[col_name].iloc[i]

        # Load image
        image = common.load_image(img_path)
        if image is None:
            fleur_scores.append(None)
            fleur_explanations.append(None)
            continue

        # Generate score & explanation
        score, explanation = generate_fleur_response(image, caption)
        fleur_scores.append(score)
        fleur_explanations.append(explanation)

        # Free image memory
        del image
        torch.cuda.empty_cache()

    return fleur_scores, fleur_explanations