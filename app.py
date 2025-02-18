import os
import streamlit as st
from dotenv import load_dotenv
import openai
from pinecone.grpc import PineconeGRPC as Pinecone

# Load environment variables
load_dotenv()

# Retrieve keys and endpoint info from .env
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

USDA_INDEX_NAME = os.getenv("USDA_INDEX_NAME")
CHEM_INDEX_NAME = os.getenv("CHEM_INDEX_NAME")
USDA_INDEX_HOST = os.getenv("USDA_INDEX_HOST")
CHEM_INDEX_HOST = os.getenv("CHEM_INDEX_HOST")

# SambaNova API configuration
SAMBA_BASE_URL = "https://api.sambanova.ai/v1/"
MODEL_NAME = "Meta-Llama-3.1-405B-Instruct"

####################################
# Initialize Pinecone
####################################
pc = Pinecone(api_key=PINECONE_API_KEY or "")

# Create index objects with explicit hosts
usda_index = pc.Index(name=USDA_INDEX_NAME, host=USDA_INDEX_HOST)
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
# Format USDA Data Function
####################################
def format_food_data(matches):
    formatted = ""
    for i, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        food_name = metadata.get("FOOD_NAME", "Unknown Food")
        formatted += f"{i}. {food_name}\n"
        for k, v in metadata.items():
            if k != "FOOD_NAME":
                formatted += f"   {k}: {v}\n"
        formatted += "\n"
    return formatted

####################################
# SambaNova Chat Completion using OpenAI Module
####################################
def sambanova_chat(prompt):
    if not SAMBANOVA_API_KEY:
        return "Error: Missing SambaNova API key."
    try:
        # Create an OpenAI client configured with the SambaNova API details
        client = openai.OpenAI(
            base_url=SAMBA_BASE_URL,
            api_key=SAMBANOVA_API_KEY
        )
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True  # Enable streaming
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
st.title("USDA & Chemical Ingredient Assistant (SambaNova)")

query_input = st.text_input("Enter a food query:")
if st.button("Search"):
    if not query_input.strip():
        st.write("Please provide a valid query.")
    else:
        st.markdown("### 1) Searching USDA Index")
        usda_matches = similarity_search(query_input, usda_index, top_k=1)
        if usda_matches:
            st.code(format_food_data(usda_matches))
            best_match_meta = usda_matches[0].get("metadata", {})
            food_name = best_match_meta.get("FOOD_NAME", "Unknown")
            ingredients_str = best_match_meta.get("FOOD_INGREDIENTS", "")
            if ingredients_str:
                st.markdown("### 2) Checking Chemicals Index per Ingredient")
                ing_list = [x.strip() for x in ingredients_str.split(",") if x.strip()]
                chem_summary = []
                for ing in ing_list:
                    chem_matches = similarity_search(ing, chem_index, top_k=1)
                    if chem_matches:
                        top_chem_meta = chem_matches[0].get("metadata", {})
                        chem_summary.append(f"Ingredient '{ing}' -> {top_chem_meta}")
                    else:
                        chem_summary.append(f"No match found for '{ing}'.")
                for item in chem_summary:
                    st.write(item)
                
                st.markdown("### 3) SambaNova Chat Completion")
                prompt_text = f"""
You are an expert nutrition and ingredient assistant. The user asked about "{food_name}".
We retrieved USDA data that includes these ingredients:
{ingredients_str}

From the chemicals index, we found these matches:
{chem_summary}

Please provide a clear, concise explanation of the food, its ingredients, potential allergens, and any relevant chemical notes.
"""
                response_text = sambanova_chat(prompt_text)
                st.write("**SambaNova response:**")
                st.write(response_text)
            else:
                st.write("No ingredient list found in USDA data.")
        else:
            st.write("No matches found in USDA index.")
