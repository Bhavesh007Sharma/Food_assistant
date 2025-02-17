import os
import logging
import chainlit as cl
from dotenv import load_dotenv

# For Pinecone
import pinecone
from pinecone.grpc import PineconeGRPC as PC  # only if you're using the GRPC approach

# For SambaNova (OpenAI-compatible)
from openai import AsyncOpenAI

# For plotting a bar chart
import matplotlib.pyplot as plt
import io
import base64

#############################################################################
# 1) ENVIRONMENT SETUP
#############################################################################
load_dotenv()  # If local .env; on HF Spaces, use os.environ directly

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY", "")

# Pinecone init
if PINECONE_API_KEY and PINECONE_ENV:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
else:
    logging.warning("Pinecone credentials not found. Make sure they're set in environment variables.")

# The two indexes we need
USDA_INDEX_NAME = "usda-food-data"
CHEM_INDEX_NAME = "food-chemicals"

# Check if they exist in your Pinecone project
existing_indexes = pinecone.list_indexes()
if USDA_INDEX_NAME not in existing_indexes:
    logging.warning(f"Index '{USDA_INDEX_NAME}' not found.")
if CHEM_INDEX_NAME not in existing_indexes:
    logging.warning(f"Index '{CHEM_INDEX_NAME}' not found.")

usda_index = pinecone.Index(USDA_INDEX_NAME) if USDA_INDEX_NAME in existing_indexes else None
chem_index = pinecone.Index(CHEM_INDEX_NAME) if CHEM_INDEX_NAME in existing_indexes else None

#############################################################################
# 2) HELPER FUNCTIONS
#############################################################################
def embed_e5(queries):
    """
    Use Pinecone's gRPC-based 'inference.embed' with the 'multilingual-e5-large' model.
    Adjust if you're using the standard Python client instead of gRPC.
    """
    try:
        pc = PC(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        resp = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=queries,
            parameters={"input_type": "query"}
        )
        return [r["values"] for r in resp]
    except Exception as e:
        logging.error(f"Error embedding with e5: {e}")
        return []

def pinecone_sim_search(query, index, top_k=3):
    """
    1) Generate e5 embeddings
    2) Query the specified Pinecone index
    """
    if not index:
        logging.warning("No Pinecone index provided, can't search.")
        return []

    emb = embed_e5([query])
    if not emb:
        return []

    try:
        results = index.query(
            vector=emb[0],
            top_k=top_k,
            include_metadata=True
        )
        return results.matches if results and hasattr(results, "matches") else []
    except Exception as e:
        logging.error(f"Pinecone query error: {e}")
        return []

def retrieve_usda_data(food_query):
    """
    Retrieve top USDA match for 'food_query' from 'usda-food-data' index.
    Return a dict with:
      {
        "Food Name": ...,
        "Nutrients": { full metadata ... }
      }
    or None if no data.
    """
    if not usda_index:
        return None

    matches = pinecone_sim_search(food_query, usda_index, top_k=1)
    if not matches:
        return None

    # Just the top match
    meta = matches[0].metadata
    return {
        "Food Name": meta.get("FOOD_NAME", "Unknown Food"),
        "Nutrients": meta
    }

def parse_ingredients_from_usda(item):
    """
    If USDA data has 'FOOD_INGREDIENTS', parse them (split by commas).
    Return list of ingredient strings.
    """
    if not item or "Nutrients" not in item:
        return []
    ingred_str = item["Nutrients"].get("FOOD_INGREDIENTS", "")
    if not ingred_str.strip():
        return []
    return [i.strip() for i in ingred_str.split(",") if i.strip()]

def retrieve_chemical_data(ingredient):
    """
    Search the 'food-chemicals' index for the ingredient. Return top match or None.
    """
    if not chem_index:
        return None
    matches = pinecone_sim_search(ingredient, chem_index, top_k=1)
    if matches:
        return matches[0].metadata
    return None

def get_chemicals_for_ingredients(ingredients):
    """
    For each ingredient in the list, fetch chemical data from 'food-chemicals'.
    Return a list of dicts.
    """
    results = []
    for ing in ingredients:
        c = retrieve_chemical_data(ing)
        if c:
            # Merge the retrieved metadata with the ingredient name
            c["Ingredient"] = ing
            results.append(c)
        else:
            results.append({
                "Ingredient": ing,
                "Info": "No chemical match found"
            })
    return results

def build_prompt(user_query, usda_item, chemicals):
    """
    Creates a big prompt that includes disclaimers about 100g data, brand disclaimers, 
    mention 0.0 means below detection limit, etc., referencing both USDA and chemical info.
    """
    if not usda_item:
        return f"User asked: {user_query}\nNo USDA data available."

    # Format USDA data
    food_name = usda_item["Food Name"]
    nutr = usda_item["Nutrients"]
    nutr_text = "\n    ".join([f"{k}: {v}" for k, v in nutr.items()])

    # Format chemical data
    chem_text = ""
    for cinfo in chemicals:
        chem_text += f"- Ingredient: {cinfo.get('Ingredient', 'Unknown')}\n"
        for key, val in cinfo.items():
            if key != "Ingredient":
                chem_text += f"   {key}: {val}\n"
        chem_text += "\n"

    big_prompt = f"""
Answer the user's query: {user_query}

We have USDA data for a food item:
- Food Name: {food_name}
  Nutrients:
    {nutr_text}

We also checked each ingredient in the 'food-chemicals' index:
{chem_text}

IMPORTANT:
1. Nutrient data is typically per 100g. 0.0 => below detection limit.
2. If brand-specific info is missing but user specifically wants that brand, say "not enough info."
3. If user wants serving size details but we only have 100g data, clarify that.
4. If user wants chemical or allergen info, use what we've retrieved or say we lack data.
5. Start your answer referencing the item found. Use *Markdown formatting*. 
6. If not enough info, respond with "I'm sorry, I don't have enough information..."

At the end, provide example *follow-up questions* in italics, e.g.:
*What vitamins are present?*, *Does it contain any allergens?*, etc.
Use $...$ for in-line LaTeX if needed. Avoid square brackets in LaTeX.
"""
    return big_prompt

def plot_nutrients(nutr):
    """
    Create a bar chart of some selected numeric nutrients: 
    Energy (kcal), Protein (g), Total lipid (fat) (g), Carbohydrate (g), Sugars (g).
    Return as a PNG in bytes.
    """
    # Keys we might care about:
    keys = {
        "Energy (kcal)": "Calories",
        "Protein (g)": "Protein",
        "Total lipid (fat) (g)": "Fat",
        "Carbohydrate (g)": "Carbs",
        "Sugars (g)": "Sugar"
    }

    data = []
    for raw_key, label in keys.items():
        val = nutr.get(raw_key, 0)
        try:
            val = float(val)
        except:
            val = 0
        data.append((label, val))

    # If all zero, skip
    if all(v == 0 for _, v in data):
        return None

    # Plot using matplotlib
    labels = [x[0] for x in data]
    values = [x[1] for x in data]

    plt.figure(figsize=(5, 3))
    plt.bar(labels, values, color="teal")
    plt.title("Nutrient Chart (per 100g)")
    plt.ylabel("Amount")
    plt.tight_layout()

    # Save to PNG in-memory
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf.read()

#############################################################################
# 3) CHAINLIT SETUP
#############################################################################
SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1/"
MODEL_NAME = "Meta-Llama-3.1-405B-Instruct"

# We'll create a single AsyncOpenAI client (SambaNova)
client = AsyncOpenAI(
    base_url=SAMBANOVA_BASE_URL,
    api_key=SAMBANOVA_API_KEY
)

async def stream_llm_response(messages):
    """
    Streams the LLM response in Chainlit.
    """
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Stream from SambaNova
        stream = await client.chat.completions.create(
            messages=messages,
            stream=True,
            model=MODEL_NAME,
            temperature=0
        )

        final_content = ""
        async for part in stream:
            piece = part.choices[0].delta.get("content", "")
            await msg.stream_token(piece)
            final_content += piece

        # Finalize
        messages.append({"role": "assistant", "content": final_content})
        await msg.send()

    except Exception as e:
        logging.error(f"Streaming error: {e}")
        await msg.update(content=f"Error from SambaNova: {e}")


#############################################################################
# 4) CHAINLIT EVENT HANDLERS
#############################################################################
@cl.on_chat_start
async def start():
    welcome = """
**Welcome to the USDA Food & Chemicals Assistant (Single-File Chainlit)!**

**How it Works**:
1. We'll look up the food in the 'usda-food-data' index.
2. We'll parse its 'FOOD_INGREDIENTS' and check each ingredient in 'food-chemicals'.
3. We'll plot a small bar chart of key nutrients (calories, protein, etc.).
4. Finally, we pass everything to SambaNova for a final, detailed answer.

Just type the name of a food item to begin!
"""
    await cl.Message(content=welcome).send()

@cl.on_message
async def main_flow(message: cl.Message):
    user_query = message.content.strip()
    if not user_query:
        await cl.Message(
            content="Please enter a valid query (e.g., 'Oreo Cookies')."
        ).send()
        return

    # 1) Retrieve USDA data
    usda_item = retrieve_usda_data(user_query)
    if not usda_item:
        await cl.Message(
            content="I'm sorry, I don't have enough information to accurately answer that question."
        ).send()
        return

    # 2) Parse ingredients, get chemical info
    ingredients = parse_ingredients_from_usda(usda_item)
    chemical_info = get_chemicals_for_ingredients(ingredients)

    # 3) Plot nutrients
    chart_buf = plot_nutrients(usda_item["Nutrients"])
    if chart_buf:
        # Convert to base64 for chainlit image
        b64chart = base64.b64encode(chart_buf).decode("utf-8")
        await cl.Image(content=b64chart, name="nutrient_chart.png").send()
    else:
        await cl.Message(content="No numeric nutrient data to plot.").send()

    # 4) Build the big prompt
    system_prompt = build_prompt(user_query, usda_item, chemical_info)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    # 5) Stream the LLM response
    await stream_llm_response(messages)
