import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone

###############################################
# 1) Load environment variables
###############################################
load_dotenv()  # Loads .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
USDA_INDEX_NAME = os.getenv("USDA_INDEX_NAME", "usda-food-data")
CHEM_INDEX_NAME = os.getenv("CHEM_INDEX_NAME", "food-chemicals")

# SambaNova-specific
BASE_URL = "https://api.sambanova.ai/v1/"
MODEL_NAME = "Meta-Llama-3.1-405B-Instruct"

###############################################
# 2) Initialize Pinecone
###############################################
pc = Pinecone(api_key=PINECONE_API_KEY)
usda_index = pc.Index(
    name=USDA_INDEX_NAME,
    host=f"https://{USDA_INDEX_NAME}-7z0q4pp.svc.aped-4627-b74a.pinecone.io"  # Example host
)
chem_index = pc.Index(
    name=CHEM_INDEX_NAME,
    host=f"https://{CHEM_INDEX_NAME}-7z0q4pp.svc.aped-4627-b74a.pinecone.io"  # Example host
)

###############################################
# 3) Pinecone similarity search functions
###############################################
def similarity_search(query, index, top_k=10):
    """Searches Pinecone index using an e5 embedding for 'query'."""
    query_emb = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    if query_emb and "values" in query_emb[0]:
        results = index.query(
            vector=query_emb[0]["values"],
            top_k=top_k,
            include_metadata=True
        )
        return results.get("matches", [])
    return []

def format_food_match(match):
    meta = match["metadata"]
    food_name = meta.get("FOOD_NAME", "Unknown Food")
    food_ingredients = meta.get("FOOD_INGREDIENTS", "N/A")
    return f"- **Name**: {food_name}\n- **Ingredients**: {food_ingredients}"

def format_chem_match(match):
    meta = match["metadata"]
    chem_name = meta.get("Chemical_Name", "Unknown Chemical")
    use_groups = meta.get("SIN_Use_Groups", "N/A")
    echa = meta.get("ECHA_Classification", "N/A")
    return f"- **Chemical**: {chem_name}\n- **Use Groups**: {use_groups}\n- **ECHA Classification**: {echa}"

###############################################
# 4) LLM (SambaNova) request
###############################################
def generate_completion_with_sambanova(api_key, prompt):
    if not api_key:
        return "Error: Please enter a valid SambaNova API key in .env."
    try:
        # Use openai.py's "OpenAI" but point base_url to SambaNova
        client = OpenAI(
            base_url=BASE_URL,
            api_key=api_key
        )

        # We stream the response, collecting partial content
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        return response.strip()
    except Exception as e:
        return f"Error: {str(e)}"

###############################################
# 5) Build a better prompt
###############################################
def build_prompt(user_food_query, food_matches, chemical_infos):
    """
    - user_food_query: The raw query from the user
    - food_matches: List of Pinecone matches from the USDA index
    - chemical_infos: List of (ingredient_name, pinecone_match or None)
    """

    # Format matched USDA foods
    matched_food_str = ""
    for i, fm in enumerate(food_matches, start=1):
        matched_food_str += f"Match {i}:\n{format_food_match(fm)}\n\n"

    # For each ingredient, show chemical data if found
    chemicals_section = "## Potential Chemical Details\n"
    for ing_name, chem_match in chemical_infos:
        if chem_match:
            chemicals_section += f"**Ingredient:** {ing_name}\n"
            chemicals_section += f"{format_chem_match(chem_match)}\n\n"
        else:
            chemicals_section += f"**Ingredient:** {ing_name}\nNo chemical match found.\n\n"

    # More elaborate instructions to the LLM
    # Focus on the fact that we have chemical info & their potential effects
    # so the LLM can mention them if relevant.
    prompt = f"""
You are a helpful assistant with access to USDA-branded food data and a chemicals database.

User is asking about: **{user_food_query}**

Here is the relevant USDA food data we found:
{matched_food_str}

Here is relevant chemical data from the ingredients above:
{chemicals_section}

Now respond to the user. Provide any relevant info on nutrients, ingredients, and if needed,
highlight possible effects of those chemicals on the human body. Do not speculate beyond data.
If there's insufficient data, say so.

Answer thoroughly, but keep it to known facts only. Provide a short summary at the end of your answer,
and when appropriate, you can format data in bullet lists or brief tables. Avoid disclaimers 
like "always check the packaging"; you have direct USDA data. Keep your answer concise, 
but well-structured. 
"""
    return prompt

###############################################
# 6) Streamlit UI
###############################################
st.title("USDA + Chemical Info Food Assistant (SambaNova)")

st.write("Enter a food query to retrieve data from Pinecone, then let the SambaNova LLM answer.")
user_food_query = st.text_input("Food Query", "")

if st.button("Search & Get Info"):
    if not user_food_query.strip():
        st.warning("Please enter a food item or query.")
        st.stop()

    # Step 1: Search USDA index
    food_results = similarity_search(user_food_query, usda_index, top_k=1)

    if not food_results:
        st.error("No USDA matches found for that query. Try a different query.")
        st.stop()

    # Show top match
    st.markdown("### USDA Food Match:")
    st.write(format_food_match(food_results[0]))

    # Step 2: If it has "FOOD_INGREDIENTS", parse them & check chemical DB
    best_match_meta = food_results[0]["metadata"]
    ingredients_str = best_match_meta.get("FOOD_INGREDIENTS", "")
    ingredients_list = [i.strip() for i in ingredients_str.split(",") if i.strip()]

    chemical_infos = []
    st.markdown("### Checking each ingredient in chemical index...")
    for ing in ingredients_list:
        matches = similarity_search(ing, chem_index, top_k=1)
        if matches:
            st.write(f"**Ingredient**: {ing}")
            st.write(format_chem_match(matches[0]))
            chemical_infos.append((ing, matches[0]))
        else:
            st.write(f"**Ingredient**: {ing} -> No match found")
            chemical_infos.append((ing, None))
        st.write("---")

    # Step 3: Build prompt for the LLM
    prompt = build_prompt(user_food_query, food_results, chemical_infos)

    # Step 4: Send to SambaNova
    st.markdown("### LLM Response")
    with st.spinner("Generating response via SambaNova..."):
        final_answer = generate_completion_with_sambanova(SAMBANOVA_API_KEY, prompt)

    st.write(final_answer)
