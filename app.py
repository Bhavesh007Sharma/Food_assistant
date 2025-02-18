import os
import chainlit as cl
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

# Create index objects using explicit hosts
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
# Chainlit UI Setup & Handlers
####################################
@cl.on_chat_start
async def start():
    welcome_message = """
## Welcome to the USDA & Chemical Ingredient Assistant!
Type the name of a food item (e.g., "Oreo Cookies") to get started.
    """
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def handle_message(message: cl.Message):
    user_query = message.content

    # --- 1) Retrieve USDA Food Details ---
    usda_matches = similarity_search(user_query, usda_index, top_k=1)
    if not usda_matches:
        await cl.Message(content="No matches found in USDA food details.").send()
        return
    food_details = format_usda_food_data(usda_matches)
    await cl.Message(content="**USDA Food Details:**\n" + food_details).send()

    best_match_meta = usda_matches[0].get("metadata", {})
    food_name = best_match_meta.get("FOOD_NAME", "Unknown")
    
    # --- 2) Retrieve USDA Nutrient Details ---
    # Here, we use the food name (or ID) to look up nutrient details.
    nutrient_matches = similarity_search(food_name, nutrient_index, top_k=1)
    nutrient_details = format_nutrient_data(nutrient_matches)
    if nutrient_details:
        await cl.Message(content="**USDA Nutrient Details:**\n" + nutrient_details).send()
    else:
        await cl.Message(content="No nutrient details found.").send()

    # --- 3) Retrieve Chemical Information for Each Ingredient ---
    ingredients_str = best_match_meta.get("FOOD_INGREDIENTS", "")
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
        await cl.Message(content="**Chemical Info per Ingredient:**\n" + chem_output).send()
    else:
        await cl.Message(content="No ingredients information found.").send()

    # --- 4) Combined Explanation via SambaNova Chat Completion ---
    prompt_text = f"""
You are an expert nutrition and ingredient assistant. The user asked about "{food_name}".
USDA Food Details:
{food_details}

USDA Nutrient Details:
{nutrient_details}

Chemical Information for Ingredients:
{chem_output if ingredients_str else 'None'}

Please provide a clear, concise explanation of the food, its nutritional profile, ingredient composition, potential allergens, and any relevant chemical insights.
    """
    chat_response = sambanova_chat(prompt_text)
    await cl.Message(content="**SambaNova Response:**\n" + chat_response).send()
