import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from together import Together
from pinecone import Pinecone  # Updated import

# Load environment variables
load_dotenv()

# Retrieve keys and endpoints from .env
TOGETHER_API_KEY = os.getenv("SAMBANOVA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

USDA_INDEX_NAME = os.getenv("USDA_INDEX_NAME")
NUTRIENT_INDEX_NAME = os.getenv("NUTRIENT_INDEX_NAME")
CHEM_INDEX_NAME = os.getenv("CHEM_INDEX_NAME")
USDA_INDEX_HOST = os.getenv("USDA_INDEX_HOST")
NUTRIENT_INDEX_HOST = os.getenv("NUTRIENT_INDEX_HOST")
CHEM_INDEX_HOST = os.getenv("CHEM_INDEX_HOST")

####################################
# Initialize Pinecone indexes
####################################
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")

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
# Chat Completion using Together API
####################################
def together_chat(prompt):
    if not TOGETHER_API_KEY:
        return "Error: Missing Together API key."
    try:
        client = Together(api_key=TOGETHER_API_KEY)
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=messages,
            max_tokens=500
        )
        response = completion.choices[0].message.content
        return response.strip()
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
- A comparison of the product's nutrient values with fixed World Health Organization (WHO) recommended daily allowances (see our fixed values in the app).
- A thorough analysis of the ingredient composition, highlighting any potential allergens.
- An evaluation of any hazardous effects associated with preservatives or additives present in the dish, including known adverse health effects.
- Include complete chemical insights where applicable (e.g., show chemical structures using inline LaTeX equations such as $E=mc^2$).

Please format your response using markdown with headings and bullet points, and provide some example follow-up questions in *italics* (for example, *Does this food contain any allergens?*, *What is the calorie count per serving?*).

If the provided data is insufficient, reply with "I'm sorry, I don't have enough information to accurately answer that question."
    """
    return prompt

####################################
# Streamlit UI
####################################
st.title("USDA & Chemical Ingredient Assistant")
st.markdown(
    """
Enter a food item (e.g., **Oreo Cookies**) to retrieve USDA details, nutrient information, chemical insights, and hazard evaluations.
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
                # Basic nutrient chart
                df_chart = pd.DataFrame(list(nutrient_values.items()), columns=["Nutrient", "Value"]).set_index("Nutrient")
                st.subheader("Nutrient Chart")
                st.bar_chart(df_chart)
                
                # Fixed WHO recommendation values
                who_recommendations = {
                    "CARBOHYDRATE, BY DIFFERENCE (G)": 275,
                    "FIBER, TOTAL DIETARY (G)": 25,
                    "PROTEIN (G)": 50,
                    "TOTAL SUGARS (G)": 25,
                    "TOTAL LIPID (FAT) (G)": 70,
                    "FATTY ACIDS, TOTAL SATURATED (G)": 20
                }
                comp_data = []
                for nutrient, value in nutrient_values.items():
                    who_value = who_recommendations.get(nutrient, None)
                    if who_value is not None:
                        comp_data.append({"Nutrient": nutrient, "Value": value, "Type": "Product"})
                        comp_data.append({"Nutrient": nutrient, "Value": who_value, "Type": "WHO Recommended"})
                if comp_data:
                    df_comp = pd.DataFrame(comp_data)
                    fig = px.bar(df_comp, x="Nutrient", y="Value", color="Type", barmode="group",
                                 title="Nutrient Comparison: Product vs WHO Recommendations")
                    st.plotly_chart(fig)
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
        
        # 4) Combined Explanation via Together Chat Completion
        prompt_text = generate_prompt(
            food_name, 
            food_details_formatted, 
            nutrient_details_formatted, 
            ingredients_str
        )
        st.subheader("Chemical & Nutrient Explanation")
        st.markdown("Generating explanation...")
        explanation = together_chat(prompt_text)
        st.markdown(explanation)
