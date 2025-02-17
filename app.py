import os
import logging
import streamlit as st
import pinecone
import openai
import matplotlib.pyplot as plt
import base64
import io

###############################################################################
# 1) ENVIRONMENT & PINECONE SETUP
###############################################################################
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY", "")

# Initialize Pinecone the way you do in your code.
# (NOT the standard python client usage—assuming your local version supports pc.inference.)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Indices
USDA_INDEX_NAME = "usda-food-data"
CHEM_INDEX_NAME = "food-chemicals"

# Attempt connecting to indexes
try:
    existing_indexes = pinecone.list_indexes()
    if USDA_INDEX_NAME not in existing_indexes:
        logging.warning(f"Index '{USDA_INDEX_NAME}' not found in Pinecone.")
    if CHEM_INDEX_NAME not in existing_indexes:
        logging.warning(f"Index '{CHEM_INDEX_NAME}' not found in Pinecone.")

    usda_index = pinecone.Index(USDA_INDEX_NAME) if USDA_INDEX_NAME in existing_indexes else None
    chem_index = pinecone.Index(CHEM_INDEX_NAME) if CHEM_INDEX_NAME in existing_indexes else None
except Exception as e:
    logging.error(f"Error initializing Pinecone: {e}")
    usda_index = None
    chem_index = None

###############################################################################
# 2) EMBEDDING & PINECONE SEARCH (Your style: pc.inference.embed(...))
###############################################################################
def embed_query(query_text: str):
    """
    Use your Pinecone inference approach. 
    Example from your code snippet:
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query_text],
        parameters={"input_type": "query"}
    )
    Then return query_embedding[0]['values'] or handle errors gracefully.
    """
    try:
        # 'pc' is presumably pinecone or something in your environment that has .inference
        # We'll assume 'pinecone' (or an alias) is your same object. Adjust if needed.
        query_embedding = pinecone.inference.embed(
            model="multilingual-e5-large",
            inputs=[query_text],
            parameters={"input_type": "query"}
        )
        if query_embedding and "values" in query_embedding[0]:
            return query_embedding[0]["values"]
    except Exception as e:
        logging.error(f"Embedding error: {e}")
    return []

def similarity_search(query_text, index, top_k=3):
    """
    1) Embed with e5
    2) Query the Pinecone index
    3) Return top_k matches
    """
    if not index:
        return []
    vec = embed_query(query_text)
    if not vec:
        return []
    try:
        result = index.query(vector=vec, top_k=top_k, include_metadata=True)
        return result.matches if result and hasattr(result, "matches") else []
    except Exception as e:
        logging.error(f"Pinecone query error: {e}")
        return []

###############################################################################
# 3) RETRIEVAL FUNCTIONS
###############################################################################
def retrieve_usda_item(food_query):
    """
    Get top match from 'usda-food-data'.
    Return { 'Food Name': ..., 'Nutrients': {...} }
    or None if not found.
    """
    matches = similarity_search(food_query, usda_index, top_k=1)
    if not matches:
        return None
    meta = matches[0].metadata
    return {
        "Food Name": meta.get("FOOD_NAME", "Unknown Food"),
        "Nutrients": meta
    }

def retrieve_chemical_info(ingredient):
    """
    Query 'food-chemicals' index for a single ingredient.
    Return top match's metadata or a fallback if none found.
    """
    matches = similarity_search(ingredient, chem_index, top_k=1)
    if not matches:
        return {"Ingredient": ingredient, "Info": "No chemical match found."}
    data = matches[0].metadata
    data["Ingredient"] = ingredient
    return data

def parse_ingredients(nutr_metadata):
    """
    Splits the 'FOOD_INGREDIENTS' field by commas.
    """
    ingred_str = nutr_metadata.get("FOOD_INGREDIENTS", "")
    if not ingred_str.strip():
        return []
    return [x.strip() for x in ingred_str.split(",") if x.strip()]

###############################################################################
# 4) BUILD PROMPT FOR SAMBANOVA
###############################################################################
def build_prompt(user_query, usda_item, chem_data):
    """
    Combine USDA + chemical data into a single big prompt:
    - disclaimers about 100g-based nutrients
    - mention 0.0 means below detection
    - brand disclaimers
    - highlight chemical info if found
    - final instructions about follow-up
    """
    if not usda_item:
        return f"User asked: {user_query}\nNo data found."

    # Format USDA item
    nutr_str = "\n".join([f"   {k}: {v}" for k, v in usda_item["Nutrients"].items()])

    # Format chemical data
    chem_str_list = []
    for c in chem_data:
        lines = [f"{k}: {v}" for k, v in c.items() if k != "Ingredient"]
        chem_str = f"- Ingredient: {c.get('Ingredient', 'Unknown')}\n   " + "\n   ".join(lines)
        chem_str_list.append(chem_str)
    chem_block = "\n".join(chem_str_list)

    big_prompt = f"""
Answer the user's query: {user_query}

We have USDA data (per 100g):
- Food Name: {usda_item['Food Name']}
  Nutrients:
{nutr_str}

We also checked each ingredient against 'food-chemicals':
{chem_block}

IMPORTANT POINTS:
1. 0.0 => below detection limit.
2. If brand-specific info is missing but user specifically wants that brand, say not enough info.
3. If user wants serving size detail but we only have 100g, clarify that.
4. Start your answer referencing the item. Use *Markdown* formatting.
5. If not enough info, say "I'm sorry, I don't have enough information..."
6. End with some *example follow-up questions* in italics.
Use $ for inline LaTeX if needed. Avoid square brackets in LaTeX.
"""
    return big_prompt

###############################################################################
# 5) SAMBANOVA CODE (SIMILAR TO YOUR EXAMPLE)
###############################################################################
def generate_response_sambanova(api_key, prompt):
    """
    Streams the SambaNova response, collecting each chunk to build final output.
    """
    if not api_key:
        return "Error: Please enter a valid SambaNova API key."

    try:
        # We'll do the same pattern as your code
        client = openai.OpenAI(
            base_url="https://api.sambanova.ai/v1/",
            api_key=api_key
        )

        completion = client.chat.completions.create(
            model="Meta-Llama-3.1-405B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        response = []
        for chunk in completion:
            piece = chunk.choices[0].delta.get("content", "")
            response.append(piece)

        return "".join(response).strip()

    except Exception as e:
        return f"Error: {str(e)}"

###############################################################################
# 6) OPTIONAL: PLOT KEY NUTRIENTS (CALORIES, PROTEIN, ETC.) VIA MATPLOTLIB
###############################################################################
def plot_nutrients_bar(nutr_metadata):
    """
    Build a bar chart of e.g.:
    - Energy (kcal)
    - Protein (g)
    - Total lipid (fat) (g)
    - Carbohydrate (g)
    - Sugars (g)
    Return base64-encoded PNG or None if no numeric data to plot.
    """
    fields = {
        "Energy (kcal)": "Calories",
        "Protein (g)": "Protein",
        "Total lipid (fat) (g)": "Fat",
        "Carbohydrate (g)": "Carbs",
        "Sugars (g)": "Sugar"
    }

    import matplotlib.pyplot as plt
    labels = []
    vals = []
    for raw, label in fields.items():
        val = nutr_metadata.get(raw, 0)
        try:
            val = float(val)
        except:
            val = 0
        labels.append(label)
        vals.append(val)

    if all(v == 0 for v in vals):
        return None

    plt.figure(figsize=(5,3))
    plt.bar(labels, vals, color="teal")
    plt.title("Key Nutrients (per 100g)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    b64chart = base64.b64encode(buf.read()).decode("utf-8")
    return b64chart

###############################################################################
# 7) STREAMLIT UI
###############################################################################
st.title("USDA + Food Chemicals (2 Index) + SambaNova [Single File]")

st.markdown("""
**Instructions**:
1. Input your SambaNova API key.
2. Enter a food item (e.g. "Oreo Cookies").
3. We'll query:
   - `usda-food-data` for nutrient info
   - `food-chemicals` for each ingredient
4. We'll also generate a bar chart for key nutrients.
5. Finally, SambaNova will produce a streamed LLM response using a big prompt.
""")

samba_key = st.text_input("SambaNova API Key:", type="password")
food_query = st.text_input("Food item (e.g., 'Oreo Cookies'):")

if st.button("Analyze"):
    if not food_query.strip():
        st.error("Please enter a valid food item.")
    else:
        # 1) Retrieve USDA item
        usda_item = retrieve_usda_item(food_query.strip())
        if not usda_item:
            st.error("No USDA data found for that query in Pinecone.")
        else:
            st.success(f"USDA data found for: {usda_item['Food Name']}")
            # Show entire USDA metadata
            st.subheader("USDA Metadata")
            st.json(usda_item["Nutrients"])

            # 2) Plot nutrients
            st.subheader("Key Nutrient Bar Chart")
            chart_b64 = plot_nutrients_bar(usda_item["Nutrients"])
            if chart_b64:
                st.image(f"data:image/png;base64,{chart_b64}")
            else:
                st.info("No numeric nutrient data to plot.")

            # 3) Retrieve chemical info for each ingredient
            st.subheader("Chemical Info for Ingredients")
            ingredients = parse_ingredients(usda_item["Nutrients"])
            chem_data_list = []
            if ingredients:
                for ing in ingredients:
                    chem_data = retrieve_chemical_info(ing)
                    chem_data_list.append(chem_data)
                    st.write(f"**Ingredient:** {chem_data.get('Ingredient','Unknown')}")
                    st.json(chem_data)
            else:
                st.info("No ingredients found in USDA data.")

            # 4) Build final prompt & call SambaNova
            st.subheader("SambaNova LLM Explanation")
            final_prompt = build_prompt(food_query, usda_item, chem_data_list)
            if not samba_key:
                st.warning("Please enter your SambaNova API key above.")
            else:
                with st.spinner("Generating response from SambaNova..."):
                    response_text = generate_response_sambanova(samba_key, final_prompt)
                st.markdown(response_text)
