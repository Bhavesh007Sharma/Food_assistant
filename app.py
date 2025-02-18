import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import openai
import pinecone

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

USDA_INDEX_NAME = os.getenv("USDA_INDEX_NAME")
NUTRIENT_INDEX_NAME = os.getenv("NUTRIENT_INDEX_NAME")
CHEM_INDEX_NAME = os.getenv("CHEM_INDEX_NAME")

SAMBA_BASE_URL = "https://api.sambanova.ai/v1/"
MODEL_NAME = "Meta-Llama-3.1-405B-Instruct"

# -----------------------------
# Initialize Pinecone (v2 client)
# -----------------------------
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
usda_index = pinecone.Index(USDA_INDEX_NAME)
nutrient_index = pinecone.Index(NUTRIENT_INDEX_NAME)
chem_index = pinecone.Index(CHEM_INDEX_NAME)

# -----------------------------
# Similarity Search Function
# -----------------------------
def similarity_search(query, index, top_k=5):
    # Use OpenAI's embedding API for the query embedding
    try:
        embed_response = openai.Embedding.create(
            input=[query],
            model="text-embedding-ada-002"
        )
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []
    embedding = embed_response["data"][0]["embedding"]
    # Query the Pinecone index with the embedding
    res = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return res["matches"] if "matches" in res else []

# -----------------------------
# Formatting Functions for Display
# -----------------------------
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

# -----------------------------
# SambaNova Chat Completion
# -----------------------------
def sambanova_chat(prompt):
    if not SAMBANOVA_API_KEY:
        return "Error: Missing SambaNova API key."
    try:
        client = openai.OpenAI(base_url=SAMBA_BASE_URL, api_key=SAMBANOVA_API_KEY)
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

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("USDA & Chemical Ingredient Assistant")
st.markdown("Enter a food query (e.g., **Oreo Cookies**) to retrieve details, view nutrient charts, and receive expert insights.")

query_input = st.text_input("Enter a food item:")

if st.button("Search") and query_input:
    # 1) Retrieve USDA Food Details
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
        
        # 2) Retrieve USDA Nutrient Details (using food name as query)
        nutrient_matches = similarity_search(food_name, nutrient_index, top_k=1)
        nutrient_details = format_nutrient_data(nutrient_matches)
        st.subheader("USDA Nutrient Details")
        if nutrient_details:
            st.text(nutrient_details)
            # Prepare nutrient chart data
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
                except:
                    nutrient_values[key] = 0
            if nutrient_values:
                df_chart = pd.DataFrame(list(nutrient_values.items()), columns=["Nutrient", "Value"]).set_index("Nutrient")
                st.subheader("Nutrient Chart")
                st.bar_chart(df_chart)
        else:
            st.info("No nutrient details found.")
        
        # 3) Retrieve Chemical Information for Each Ingredient
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
        
        # 4) Combined Explanation via SambaNova Chat Completion
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
