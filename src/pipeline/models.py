import os
import sys
from google import genai
from google.genai import types
from groq import Groq
import time
import requests
import base64

from src.logger import logging
from src.exception import CustomException

apikey = os.getenv("GOOGLE_API_KEY")

if apikey is None:
    raise ValueError("API key not found. Please set GOOGLE_API_KEY environment variable.")

# Initialize client
client = genai.Client(api_key=apikey)


def Gemma_generate_caption(
    image_path,
    prompt="صف الصورة في جملة قصيرة",
    model_name="gemma-3-27b-it",  
    mime_type="image/jpeg",
    max_output_tokens=256,
    temperature=0.2,
    top_p=0.9,
    top_k=40,
    max_retries=2
):
    """
    Generate caption using Google GenAI models (Gemma).

    Args:
        image_path (str): Path to image
        prompt (str): Caption prompt
        model_name (str): Model name

    Returns:
        str or None
    """

    try:
        logging.info(f"Generating caption for: {image_path}")

        # Check file
        if not os.path.exists(image_path):
            logging.error(f"File not found: {image_path}")
            return None

        # Read image
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                    {"role": "system", "content": "أجب فقط باللغة العربية الفصحى."},
                    prompt
                ],
                config={
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
            )
            # Extract text (robust)
            caption = None

            # Direct text
            if hasattr(resp, "text") and resp.text:
                caption = resp.text.strip()

            # Candidates fallback
            elif hasattr(resp, "candidates") and resp.candidates:
                parts = resp.candidates[0].content.parts
                caption = "".join(
                    p.text for p in parts if hasattr(p, "text") and p.text
                ).strip()

            if not caption:
                logging.warning(f"Empty response for {image_path}")
                return None

            logging.info("Caption generated successfully")
            return caption
            
        except Exception as e:
            logging.error(f"Error generating Gemini caption: {e}", exc_info=True)
            return None      

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        raise CustomException(e, sys)
    
    
def gemini_generate_caption(
    image_path,
    prompt="صف الصورة في جملة قصيرة",
    model_name="gemini-2.5-flash",
    mime_type="image/jpeg",
    max_output_tokens=256,
    temperature=0.2,
    top_p=0.9,
    top_k=40,
):
    """
    Generate a caption for an image using Gemini multimodal model.

    Args:
        image_path (str): Path to the image
        prompt (str): Caption prompt
        model_name (str): Gemini model name

    Returns:
        str or None: Generated caption
    """
    try:
        logging.info(f"Generating Gemini caption for: {image_path}")

        if not os.path.exists(image_path):
            logging.error(f"File not found: {image_path}")
            return None

        # Read image bytes
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                    {"role": "system", "content": "أجب فقط باللغة العربية الفصحى."},
                    prompt
                ],
                config={
                    "max_output_tokens": max_output_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                }
            )

            # Extract caption safely
            caption = None

            if hasattr(resp, "text") and resp.text:
                caption = resp.text.strip()

            elif hasattr(resp, "candidates") and resp.candidates:
                for cand in resp.candidates:
                    if hasattr(cand, "content") and cand.content.parts:
                        texts = [
                            p.text for p in cand.content.parts
                            if hasattr(p, "text") and p.text
                        ]
                        if texts:
                            caption = " ".join(texts).strip()
                            break

            if not caption:
                logging.warning(f"Empty response for {os.path.basename(image_path)}")
                return None

            logging.info("Gemini caption generated successfully")
            return caption
        
        except Exception as e:
            logging.error(f"Error generating Gemini caption: {e}", exc_info=True)
            return None

    except Exception as e:
        logging.error(f"Unexpected error in gemini_caption: {e}", exc_info=True)
        raise CustomException(e, sys)
    

# Initialize clients
gem_client = genai.Client(api_key=gem_api_key)
llama_client = Groq(api_key=llama_api_key)

def Llama_generate_caption(
    image_url,
    prompt="صف الصورة في جملة قصيرة",
    model_name = "meta-llama/llama-4-scout-17b-16e-instruct",
    max_tokens = 256,
    temperature = 0.9,
):
    """
    Generate a caption for an image URL using LLaMA.
    Returns:
        str | None: Generated caption in Arabic, or None if failed.
    """
    if not image_url:
        logging.error("No image URL provided for LLaMA caption generation.")
        return None
    try:
        logging.info("Start Generating the caption")

        completion = llama_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "أجب فقط باللغة العربية الفصحى."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ],
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )

        caption = completion.choices[0].message.content.strip()
        logging.info("Caption generated successfully.")
        return caption

    except Exception as e:
        logging.error("Error generating caption: %s", e, exc_info=True)
        raise CustomException(e, sys)

API_KEY = os.getenv("Fanar_API_KEY")

if API_KEY is None:
    raise ValueError("API key not found. Please set Fanar_API_KEY environment variable.")    
def fanar_caption(image_path, prompt="صف الصورة في جملة قصيرة."):
    """
    Generate a caption for an image using Fanar API.
    Returns None if there is any error.
    """
    if not os.path.exists(image_path):
        logging.error(f"File not found: {image_path}")
        return None

    try:
        logging.info(f"Generating Fanar caption for: {image_path}")

        # Encode image to Base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": "Fanar-Oryx-IVU-2",
            "messages": [
                {"role": "system", "content": "أجب فقط باللغة العربية الفصحى."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ],
            "max_tokens": 256
        }

        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        response = requests.post("https://api.fanar.qa/v1/chat/completions", json=payload, headers=headers, timeout=30)

        if response.status_code != 200:
            logging.error(f"API Error [{response.status_code}]: {response.text}")
            return None

        caption = response.json().get("choices", [{}])[0].get("message", {}).get("content")
        if not caption:
            logging.warning(f"No caption returned for {os.path.basename(image_path)}")
            return None

        logging.info(f"Caption generated successfully for {os.path.basename(image_path)}")
        return caption.strip()

    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}", exc_info=True)
        raise CustomException(sys,e)