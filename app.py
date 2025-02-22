import os
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from dotenv import load_dotenv
from together import Together
from pinecone import Pinecone  # Updated import
from pyzbar.pyzbar import decode  # For barcode decoding

# LangChain imports
from langchain.llms.base import LLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Any, Dict

# --- Load environment variables --
load_dotenv()

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
def similarity_search(query, index, top_k=1):
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
        formatted += f"- **FOOD_ID (Barcode):** {meta.get('FOOD_ID', '')}\n"
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
# Original Prompt Generation Function
####################################
def generate_prompt(food_name, food_details, nutrient_details, ingredients_str):
    prompt = f"""
You are an expert nutritionist and food safety advisor with in-depth knowledge of USDA food data and dietary guidelines. Below is the data for the food item **{food_name}**:
**USDA Food Details:**
{food_details}
**Nutrient Information:**
{nutrient_details}
**Ingredients:**
{ingredients_str if ingredients_str else "Not available"}
Based on the above information, please provide a concise explanation that includes:
- A brief overview of the food item.
- A detailed breakdown of macro- and micronutrients.
- An analysis of any hazardous effects associated with chemicals, additives, or preservatives present in the food.
- Identification of potential allergens within the ingredient list.
- Suggestions for healthier alternative options, explaining why they might be nutritionally superior.
- Use inline LaTeX (wrapped in single dollar signs, e.g. `$E=mc^2$`) for any chemical structures if needed.
- If the data provided is insufficient, respond with "I'm sorry, I don't have enough information to accurately answer that question."
Format your response using markdown with headings and bullet points.
    """
    return prompt

####################################
# Together Chat Completion Function
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
            max_tokens=1000
        )
        response = completion.choices[0].message.content
        return response.strip()
    except Exception as e:
        return f"Error: {e}"

####################################
# Custom Together LLM for LangChain
####################################
class TogetherLLM(LLM):
    def _call(self, prompt, stop=None):
        return together_chat(prompt)

    @property
    def _llm_type(self):
        return "together"

####################################
# Custom Retriever for LangChain (subclassing BaseRetriever)
####################################
class PineconeRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        docs = []
        # USDA Food Details
        usda_matches = similarity_search(query, usda_index, top_k=1)
        if usda_matches:
            food_details_formatted = format_usda_food_data(usda_matches)
            docs.append(Document(page_content=food_details_formatted, metadata={"source": "USDA"}))
            best_meta = usda_matches[0].get("metadata", {})
            food_name = best_meta.get("FOOD_NAME", "Unknown Food")
            ingredients_str = best_meta.get("FOOD_INGREDIENTS", "")
            # USDA Nutrient Details
            nutrient_matches = similarity_search(food_name, nutrient_index, top_k=1)
            if nutrient_matches:
                nutrient_details_formatted = format_nutrient_data(nutrient_matches)
                docs.append(Document(page_content=nutrient_details_formatted, metadata={"source": "Nutrient"}))
                # Save nutrient values for graph generation in session state
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
                st.session_state["nutrient_values"] = nutrient_values
            else:
                nutrient_details_formatted = "No nutrient details found."
            # Chemical Information for each ingredient
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
                chem_combined = "\n".join(chem_results)
                docs.append(Document(page_content=chem_combined, metadata={"source": "Chem"}))
            # Generate detailed prompt document
            prompt_text = generate_prompt(food_name, food_details_formatted, nutrient_details_formatted, ingredients_str)
            docs.append(Document(page_content=prompt_text, metadata={"source": "Prompt"}))
        return docs

    @property
    def search_kwargs(self) -> Dict[str, Any]:
        return {}

    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self.get_relevant_documents(query, **kwargs)

####################################
# Persistent Conversational Chain Setup
####################################
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = ConversationalRetrievalChain.from_llm(
        TogetherLLM(),
        retriever=PineconeRetriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        chain_type="stuff"
    )
if "messages" not in st.session_state:
    # Preload a welcome message
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm here to help you with nutritional insights and food safety. How can I assist you today?"}
    ]

####################################
# Dark-Themed & Animated Gradient Background (Custom CSS)
####################################
st.markdown(
    """
    <style>
    /* Animate the gradient background */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    body {
        background: linear-gradient(-45deg, #2E2E2E, #141414, #2E2E2E, #1F1F1F);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #F8F8F8;
    }
    .main {
        background: rgba(30, 30, 30, 0.9);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.4);
        margin: 2rem auto;
        max-width: 1200px;
    }
    .stButton>button {
        background-color: #ff6b6b;
        color: white;
        padding: 0.6em 1.2em;
        border: none;
        border-radius: 5px;
        font-size: 1rem;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff5a5a;
    }
    .stRadio > label {
        font-weight: bold;
        font-size: 1.1rem;
        color: #EAEAEA !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 1rem;
    }
    /* Override Streamlit's text color in some components */
    .css-1u8p6ry p, .css-1u8p6ry span, .css-1u8p6ry div, .css-1u8p6ry label {
        color: #F0F0F0 !important;
    }
    .css-1cpxkw7 {
        color: #F8F8F8 !important;
    }
    /* Chat message styling */
    .stChatMessage .stChatMessageContent {
        background-color: #333333;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 1em;
    }
    .stChatMessage--user .stChatMessageContent {
        background-color: #444444;
    }
    .stChatMessage--assistant .stChatMessageContent {
        background-color: #3A3A3A;
    }
    </style>
    """,
    unsafe_allow_html=True
)

####################################
# App Title & Description
####################################
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="header-text">
      <h1 style="color:#FEE440;">Nutritional Insights & Food Safety Assistant</h1>
      <p>Discover detailed USDA food data, nutrient insights, chemical analysis, and healthy recipe suggestions in a sleek dark theme.</p>
    </div>
    """,
    unsafe_allow_html=True
)

####################################
# Streamlit UI - Chat Interface
####################################
# Select input mode
input_mode = st.radio("Select input mode:", ["Text", "Barcode Image"])

# Get user input (either via chat or barcode)
query_input = ""
if input_mode == "Text":
    query_input = st.chat_input("Enter your query:")
else:
    uploaded_file = st.file_uploader("Upload a barcode image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Barcode Image", use_column_width=True)
        barcodes = decode(image)
        if barcodes:
            barcode_data = barcodes[0].data.decode("utf-8")
            st.success(f"Decoded Barcode: {barcode_data}")
            query_input = barcode_data
        else:
            st.error("No barcode detected. Please try again.")

# Select query type
query_type = st.selectbox("Select query type:", ["USDA Database Query", "Home Made Food Analysis"])

# Process user query if provided
if query_input:
    st.session_state.messages.append({"role": "user", "content": query_input})
    if query_type == "USDA Database Query":
        result = st.session_state.chat_chain({"question": query_input})
        answer = result.get("answer", "I'm sorry, I could not generate an answer.")
        st.session_state.messages.append({"role": "assistant", "content": answer})
    elif query_type == "Home Made Food Analysis":
        st.info("Fetching data from CalorieNinjas API...")
        api_url = 'https://api.calorieninjas.com/v1/nutrition?query='
        headers = {'X-Api-Key': 'xZy/uYgnYZyoJAiAAl1obw==FG9eY7kclEvmkgMY'}  # Replace with your actual API key
        response = requests.get(api_url + query_input, headers=headers)

        if response.status_code == requests.codes.ok:
            data = response.json()
            st.write("### API Response Data")
            st.json(data)

            items = data.get("items", [])
            if items:
                # Bar chart for nutrient breakdown
                nutrient_keys = ["calories", "serving_size_g", "fat_total_g", "fat_saturated_g",
                                   "protein_g", "sodium_mg", "potassium_mg", "cholesterol_mg",
                                   "carbohydrates_total_g", "fiber_g", "sugar_g"]
                plot_data = []
                for item in items:
                    for key in nutrient_keys:
                        plot_data.append({
                            "Food": item.get("name", "Unknown"),
                            "Nutrient": key,
                            "Value": item.get(key, 0)
                        })
                df_plot = pd.DataFrame(plot_data)
                fig2 = px.bar(df_plot, x="Nutrient", y="Value", color="Food", barmode="group",
                              title="Nutrient Breakdown for Home Made Food Items")
                st.plotly_chart(fig2)

                # Radar (spider) chart for a more advanced view
                radar_fig = go.Figure()
                selected_nutrients = ["calories", "protein_g", "fat_total_g", "carbohydrates_total_g", "fiber_g"]
                for item in items:
                    food_name = item.get("name", "Unknown")
                    values = [item.get(nutr, 0) for nutr in selected_nutrients]
                    values += [values[0]]
                    radar_fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=selected_nutrients + [selected_nutrients[0]],
                        fill='toself',
                        name=food_name
                    ))
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True)
                    ),
                    showlegend=True,
                    title="Radar Chart: Nutrient Comparison for Home Made Food"
                )
                st.plotly_chart(radar_fig)
            else:
                st.error("No items found in API response.")

            # Create a prompt for the Together AI LLM integrating exercise and recipe instructions.
            prompt = f"""
You are a nutrition and fitness expert. Based on the following food items and their nutrient details:
{json.dumps(items, indent=2)}
Please provide:
- An exercise integration plan that calculates approximate calorie requirements based on user activity levels and suggests how many calories need to be burned.
- A recipe suggestion with step-by-step cooking instructions for a healthy meal using the available ingredients, ensuring the meal stays under a specified calorie limit.
Format your response in markdown with headings and bullet points.
            """
            st.write("### LLM Generated Response")
            llm_response = together_chat(prompt)
            st.markdown(llm_response)
        else:
            st.error(f"Error: {response.status_code} {response.text}")

# Display conversation
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message("assistant", avatar="🤖").write(msg["content"])
    else:
        st.chat_message("user", avatar="🙂").write(msg["content"])

####################################
# Nutrient Graph Button
####################################
if "nutrient_values" in st.session_state:
    if st.button("Generate Nutrient Graph"):
        nutrient_values = st.session_state["nutrient_values"]
        if nutrient_values:
            df_chart = pd.DataFrame(list(nutrient_values.items()), columns=["Nutrient", "Value"]).set_index("Nutrient")
            st.subheader("Nutrient Bar Chart")
            st.bar_chart(df_chart)

            # Nutrient comparison graph using WHO recommendations
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
                fig = px.bar(
                    df_comp, x="Nutrient", y="Value", color="Type",
                    barmode="group", title="Nutrient Comparison: Product vs WHO Recommendations"
                )
                st.plotly_chart(fig)
        else:
            st.info("No nutrient data available to generate graphs.")

st.markdown("</div>", unsafe_allow_html=True)
