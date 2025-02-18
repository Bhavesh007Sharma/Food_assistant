import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec

# 1) Load environment variables
load_dotenv()

SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

USDA_INDEX_NAME = os.getenv("USDA_INDEX_NAME")      # "usda-food-details"
NUTRIENT_INDEX_NAME = os.getenv("NUTRIENT_INDEX_NAME")  # "usda-nutrients"
CHEM_INDEX_NAME = os.getenv("CHEM_INDEX_NAME")          # "food-chemicals"

SAMBA_BASE_URL = "https://api.sambanova.ai/v1/"
MODEL_NAME = "Meta-Llama-3.1-405B-Instruct"

# 2) Initialize Pinecone using the new client approach
pc = Pinecone(api_key=PINECONE_API_KEY)

# Access the existing indexes by name.
# Make sure these names match your .env EXACTLY.
usda_index = pc.Index(name=USDA_INDEX_NAME)
nutrient_index = pc.Index(name=NUTRIENT_INDEX_NAME)
chem_index = pc.Index(name=CHEM_INDEX_NAME)

# 3) Similarity Search using OpenAI Embeddings
def similarity_search(query, index, top_k=5):
    # Use OpenAI to create an embedding for the query
    try:
        embed_response = openai.Embedding.create(
            input=[query],
            model="text-embedding-ada-002"
        )
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []
    embedding = embed_response["data"][0]["embedding"]
    
    # Query the specified Pinecone index
    res = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return res.get("matches", [])

# 4) Formatting Functions
def format_usda_food_data(matches):
    formatted = ""
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        food_name = meta.get("FOOD_NAME", "Unknown Food")
        formatted += f"{i}. {food_name}\n"
        formatted += f"   FOOD_ID: {meta.get('FOOD_ID', '')}\n"
        formatted += f"   Ingredients: {meta.get('FOOD_INGREDIENTS', '')}\n\n"
    return formatted

def format_nutrient_data(matches):
    formatted = ""
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        formatted += f"{i}. Nutrient Details for {meta.get('FOOD_NAME', 'Unknown Food')}\n"
        for key, value in meta.items():
            if key not in ["FOOD_ID", "FOOD_NAME"]:
                formatted += f"   {key}: {value}\n"
        formatted += "\n"
    return formatted

def format_chem_data(matches):
    formatted = ""
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        formatted += f"{i}. Chemical Info: {meta}\n"
    return formatted

# 5) SambaNova Chat Completion
def sambanova_chat(prompt):
    if not SAMBANOVA_API_KEY:
        return "Error: Missing SambaNova API key."
    try:
        client = openai.OpenAI(
            base_url=SAMBA_BASE_URL,
            api_key=SAMBANOVA_API_KEY
        )
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        return response.strip()
    except Exception as e:
        return f"Error: {e}"

# 6) Streamlit UI
st.title("USDA & Chemical Ingredient Assistant")
st.markdown("Type in a food (e.g., **Oreo Cookies**) to retrieve details from your three Pinecone indexes.")

query_input = st.text_input("Enter a food item:")

if st.button("Search") and query_input:
    # 1) USDA Food Details
    usda_matches = similarity_search(query_input, usda_index, top_k=1)
    if not usda_matches:
        st.error("No matches found in USDA food details.")
    else:
        food_details = format_usda_food_data(usda_matches)
        st.subheader("USDA Food Details")
        st.text(food_details)

        best_match_meta = usda_matches[0].get("metadata", {})
        food_name = best_match_meta.get("FOOD_NAME", "Unknown")
        ingredients_str = best_match_meta.get("FOOD_INGREDIENTS", "")

        # 2) USDA Nutrient Details
        nutrient_matches = similarity_search(food_name, nutrient_index, top_k=1)
        nutrient_details = format_nutrient_data(nutrient_matches)
        st.subheader("USDA Nutrient Details")
        if nutrient_details:
            st.text(nutrient_details)
        else:
            st.info("No nutrient details found.")

        # 3) Chemical Info
        st.subheader("Chemical Info per Ingredient")
        if ingredients_str:
            ing_list = [x.strip() for x in ingredients_str.split(",") if x.strip()]
            chem_results = []
            for ing in ing_list:
                chem_matches = similarity_search(ing, chem_index, top_k=1)
                if chem_matches:
                    chem_info = format_chem_data(chem_matches)
                    chem_results.append(f"**{ing}:**\n{chem_info}")
                else:
                    chem_results.append(f"**{ing}:** No chemical info found.")
            st.markdown("\n".join(chem_results))
        else:
            st.info("No ingredients information found.")

        # 4) SambaNova Explanation
        prompt_text = f"""
You are an expert nutrition and ingredient assistant. The user asked about "{food_name}".
USDA Food Details:
{food_details}
USDA Nutrient Details:
{nutrient_details}
Chemical Information for Ingredients:
{ingredients_str if ingredients_str else 'None'}
Please provide a clear, concise explanation of the food, its nutritional profile, ingredient composition, potential allergens, and any relevant chemical insights.
        """
        st.subheader("SambaNova Explanation")
        chat_response = sambanova_chat(prompt_text)
        st.write(chat_response)
