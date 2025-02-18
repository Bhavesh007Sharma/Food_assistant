import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import openai
from pinecone.grpc import PineconeGRPC as Pinecone

# Load environment variables
load_dotenv()

# Retrieve keys and endpoint info from .env
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

USDA_INDEX_NAME = os.getenv("USDA_INDEX_NAME")
NUTRIENT_INDEX_NAME = os.getenv("NUTRIENT_INDEX_NAME")
CHEM_INDEX_NAME = os.getenv("CHEM_INDEX_NAME")
USDA_INDEX_HOST = os.getenv("USDA_INDEX_HOST")
NUTRIENT_INDEX_HOST = os.getenv("NUTRIENT_INDEX_HOST")
CHEM_INDEX_HOST = os.getenv("CHEM_INDEX_HOST")

# SambaNova API configuration
SAMBA_BASE_URL = "https://api.sambanova.ai/v1/"
MODEL_NAME = "Meta-Llama-3.1-405B-Instruct"

####################################
# Initialize Pinecone indexes
####################################
pc = Pinecone(api_key=PINECONE_API_KEY or "")

# Create index objects using explicit hosts (names must match your Pinecone project exactly)
usda_index = pc.Index(name=USDA_INDEX_NAME, host=USDA_INDEX_HOST)
nutrient_index = pc.Index(name=NUTRIENT_INDEX_NAME, host=NUTRIENT_INDEX_HOST)
chem_index = pc.Index(name=CHEM_INDEX_NAME, host=CHEM_INDEX_HOST)

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
        return res["matches"] if res.get("matches") else []
    return []

####################################
# Format Functions for Output
####################################
def format_usda_food_data(matches):
    # Format basic food details from USDA index
    formatted = ""
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        food_name = meta.get("FOOD_NAME", "Unknown Food")
        formatted += f"{i}. {food_name}\n"
        formatted += f"   FOOD_ID: {meta.get('FOOD_ID', '')}\n"
        formatted += f"   Ingredients: {meta.get('FOOD_INGREDIENTS', '')}\n\n"
    return formatted

def format_nutrient_data(matches):
    # Format nutrient details from the Nutrient index
    formatted = ""
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        formatted += f"{i}. Nutrient Details for {meta.get('FOOD_NAME', 'Unknown Food')}\n"
        for key, value in meta.items():
            # Skip linking fields if desired
            if key not in ["FOOD_ID", "FOOD_NAME"]:
                formatted += f"   {key}: {value}\n"
        formatted += "\n"
    return formatted

def format_chem_data(matches):
    # Format chemical info from the Chemicals index
    formatted = ""
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        formatted += f"{i}. Chemical Info: {meta}\n"
    return formatted

####################################
# SambaNova Chat Completion using OpenAI Module
####################################
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

####################################
# Streamlit UI
####################################
st.title("USDA & Chemical Ingredient Assistant")
st.markdown("Enter a food item (e.g., **Oreo Cookies**) to retrieve USDA details, nutrient information, and chemical insights.")

query_input = st.text_input("Enter a food item:")

if st.button("Search") and query_input:
    # --- 1) Retrieve USDA Food Details ---
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
        
        # --- 2) Retrieve USDA Nutrient Details ---
        nutrient_matches = similarity_search(food_name, nutrient_index, top_k=1)
        nutrient_details = format_nutrient_data(nutrient_matches)
        st.subheader("USDA Nutrient Details")
        if nutrient_details:
            st.text(nutrient_details)
            # Prepare nutrient chart data from selected keys
            nutrient_keys = [
                "CARBOHYDRATE, BY DIFFERENCE (G)",
                "FIBER, TOTAL DIETARY (G)",
                "PROTEIN (G)",
                "TOTAL SUGARS (G)",
                "TOTAL LIPID (FAT) (G)",
                "FATTY ACIDS, TOTAL SATURATED (G)"
            ]
            nutrient_values = {}
            if nutrient_matches:
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
        
        # --- 3) Retrieve Chemical Information for Each Ingredient ---
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
                    chem_results.append(f"No chemical info found for {ing}.")
            chem_output = "\n".join(chem_results)
            st.markdown(chem_output)
        else:
            st.info("No ingredients information found.")
        
        # --- 4) Combined Explanation via SambaNova Chat Completion ---
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
