import os
import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import openai
from pinecone import Pinecone  # Updated import (removed grpc)

# Load environment variables
load_dotenv()

# Retrieve keys and endpoints from .env
# (Using the same variable for Hugging Face API key as before)
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

USDA_INDEX_NAME = os.getenv("USDA_INDEX_NAME")
NUTRIENT_INDEX_NAME = os.getenv("NUTRIENT_INDEX_NAME")
CHEM_INDEX_NAME = os.getenv("CHEM_INDEX_NAME")
USDA_INDEX_HOST = os.getenv("USDA_INDEX_HOST")
NUTRIENT_INDEX_HOST = os.getenv("NUTRIENT_INDEX_HOST")
CHEM_INDEX_HOST = os.getenv("CHEM_INDEX_HOST")

# Hugging Face Inference API configuration for chat completions
HF_BASE_URL = "https://router.huggingface.co/together"
HF_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

####################################
# Initialize Pinecone indexes
####################################
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")

# Removed the 'host' parameters for non-gRPC version usage
usda_index = pc.Index(name=USDA_INDEX_NAME)
nutrient_index = pc.Index(name=NUTRIENT_INDEX_NAME)
chem_index = pc.Index(name=CHEM_INDEX_NAME)

####################################
# Similarity Search Function
####################################
def similarity_search(query, index, top_k=5):
    emb_response = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    if emb_response and "values" in emb_response[0]:
        res = index.query(
            vector=emb_response[0]["values"],
            top_k=top_k,
            include_metadata=True
        )
        return res.get("matches", [])
    return []

####################################
# Format Functions for Output
####################################
def format_usda_food_data(matches):
    formatted = ""
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        food_name = meta.get("FOOD_NAME", "Unknown Food")
        formatted += f"**{i}. {food_name}**\n"
        formatted += f"- **FOOD_ID:** {meta.get('FOOD_ID', '')}\n"
        formatted += f"- **Ingredients:** {meta.get('FOOD_INGREDIENTS', '')}\n\n"
    return formatted

def format_nutrient_data(matches):
    formatted = ""
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        formatted += f"**{i}. Nutrient Details for {meta.get('FOOD_NAME', 'Unknown Food')}**\n"
        for key, value in meta.items():
            if key not in ["FOOD_ID", "FOOD_NAME"]:
                formatted += f"- **{key}:** {value}\n"
        formatted += "\n"
    return formatted

def format_chem_data(matches):
    formatted = ""
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        formatted += f"**{i}. Chemical Information**\n"
        for key, value in meta.items():
            formatted += f"- **{key}:** {value}\n"
        formatted += "\n"
    return formatted

####################################
# Chat Completion using Hugging Face Inference API
####################################
def hf_chat(prompt):
    if not HF_API_KEY:
        return "Error: Missing Hugging Face API key."
    try:
        openai.api_base = HF_BASE_URL
        openai.api_key = HF_API_KEY
        response = openai.ChatCompletion.create(
            model=HF_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            stream=True
        )
        full_response = ""
        for chunk in response:
            if "choices" in chunk and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if "content" in delta:
                    full_response += delta["content"]
        return full_response.strip()
    except Exception as e:
        return f"Error: {e}"

####################################
# Improved Prompt Generation
####################################
def generate_prompt(food_name, food_details, nutrient_details, ingredients_str):
    prompt = f"""
You are an expert nutrition and ingredient assistant. Below is the detailed data for the food item **{food_name}**:
**USDA Food Details:**
{food_details}
**Nutrient Information:**
{nutrient_details}
**Ingredients:**
{ingredients_str if ingredients_str else "Not available"}
Using the data above, please provide a clear, concise explanation covering the following:
- A brief description of the food item.
- A detailed nutrient breakdown, including macro- and micronutrients.
- An analysis of the ingredient composition and any potential allergens.
- Relevant chemical insights related to the ingredients.
Please format your response using markdown with headings, bullet points, and, where needed, inline LaTeX equations (wrapped in single dollar signs, e.g. $E=mc^2$). Also, provide some example follow-up questions in *italics* (for example, *Does this food contain any allergens?*, *What is the calorie count per serving?*).
If the provided data is insufficient, reply with "I'm sorry, I don't have enough information to accurately answer that question."
    """
    return prompt

####################################
# Streamlit UI
####################################
st.title("USDA & Chemical Ingredient Assistant")
st.markdown(
    """
Enter a food item (e.g., **Oreo Cookies**) to retrieve USDA details, nutrient information, and chemical insights.
"""
)

query_input = st.text_input("Enter a food item:")

if st.button("Search") and query_input:
    st.info("Searching for food details...")
    # 1) Retrieve USDA Food Details
    usda_matches = similarity_search(query_input, usda_index, top_k=1)
    if not usda_matches:
        st.error("No matches found in USDA food details.")
    else:
        food_details_formatted = format_usda_food_data(usda_matches)
        st.subheader("USDA Food Details")
        st.markdown(food_details_formatted)
        
        best_match_meta = usda_matches[0].get("metadata", {})
        food_name = best_match_meta.get("FOOD_NAME", "Unknown")
        ingredients_str = best_match_meta.get("FOOD_INGREDIENTS", "")
        
        # 2) Retrieve USDA Nutrient Details
        nutrient_matches = similarity_search(food_name, nutrient_index, top_k=1)
        if nutrient_matches:
            nutrient_details_formatted = format_nutrient_data(nutrient_matches)
            st.subheader("USDA Nutrient Details")
            st.markdown(nutrient_details_formatted)
            
            # Prepare nutrient chart data from selected nutrient keys
            nutrient_keys = [
                "CARBOHYDRATE, BY DIFFERENCE (G)",
                "FIBER, TOTAL DIETARY (G)",
                "PROTEIN (G)",
                "TOTAL SUGARS (G)",
                "TOTAL LIPID (FAT) (G)",
                "FATTY ACIDS, TOTAL SATURATED (G)"
            ]
            nutrient_values = {}
            meta_nutrient = nutrient_matches[0].get("metadata", {})
            for key in nutrient_keys:
                try:
                    nutrient_values[key] = float(meta_nutrient.get(key, 0))
                except Exception:
                    nutrient_values[key] = 0
            if nutrient_values:
                df_chart = pd.DataFrame(list(nutrient_values.items()), columns=["Nutrient", "Value"]).set_index("Nutrient")
                st.subheader("Nutrient Chart")
                st.bar_chart(df_chart)
        else:
            nutrient_details_formatted = "No nutrient details found."
            st.info(nutrient_details_formatted)
        
        # 3) Retrieve Chemical Information for Each Ingredient
        st.subheader("Chemical Information for Ingredients")
        if ingredients_str:
            ing_list = [x.strip() for x in ingredients_str.split(",") if x.strip()]
            chem_results = []
            for ing in ing_list:
                chem_matches = similarity_search(ing, chem_index, top_k=1)
                if chem_matches:
                    chem_info_formatted = format_chem_data(chem_matches)
                    chem_results.append(f"**{ing}:**\n{chem_info_formatted}")
                else:
                    chem_results.append(f"**{ing}:** No chemical information found.\n")
            st.markdown("\n".join(chem_results))
        else:
            st.info("No ingredient information available.")
        
        # 4) Combined Explanation via Hugging Face Chat Completion
        prompt_text = generate_prompt(
            food_name, 
            food_details_formatted, 
            nutrient_details_formatted, 
            ingredients_str
        )
        st.subheader("Chemical & Nutrient Explanation")
        st.markdown("Generating explanation...")
        explanation = hf_chat(prompt_text)
        st.markdown(explanation)
