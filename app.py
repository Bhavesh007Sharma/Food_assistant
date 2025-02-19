import os
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
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
# Initialize LangChain LLM, Memory & Retrieval Chain using Together API
####################################
llm = TogetherLLM()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = PineconeRetriever()
chat_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, chain_type="stuff"
)

####################################
# Streamlit UI - Search, Graphs, and Continuous Chat
####################################
st.title("USDA & Chemical Ingredient Assistant with Continuous Chat")
st.markdown(
    """
Enter a food item (e.g., **Oreo Cookies**) or upload a barcode image to retrieve USDA details, nutrient information, chemical insights, allergen analysis, hazardous effects, and healthier alternatives.
    """
)

# Choose input mode: text or barcode image
input_mode = st.radio("Select input mode:", ["Text", "Barcode Image"])

if input_mode == "Text":
    query_input = st.text_input("Enter a food item:")
else:
    uploaded_file = st.file_uploader("Upload a barcode image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Decode the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Barcode Image", use_column_width=True)
        barcodes = decode(image)
        if barcodes:
            barcode_data = barcodes[0].data.decode("utf-8")
            st.success(f"Decoded Barcode: {barcode_data}")
            query_input = barcode_data
        else:
            st.error("No barcode detected in the image. Please try again.")
            query_input = ""
    else:
        query_input = ""

# Search functionality that displays details and graphs
if st.button("Search") and query_input:
    st.info("Searching for food details...")
    usda_matches = similarity_search(query_input, usda_index, top_k=1)
    if not usda_matches:
        st.error("No matches found in USDA food details.")
    else:
        # USDA Food Details
        food_details_formatted = format_usda_food_data(usda_matches)
        st.subheader("USDA Food Details")
        st.markdown(food_details_formatted)
        
        best_meta = usda_matches[0].get("metadata", {})
        food_name = best_meta.get("FOOD_NAME", "Unknown")
        ingredients_str = best_meta.get("FOOD_INGREDIENTS", "")
        food_id = best_meta.get("FOOD_ID", "")
        if food_id:
            st.markdown(f"**Decoded Product Barcode:** {food_id}")
        
        # USDA Nutrient Details and Graphs
        nutrient_matches = similarity_search(food_name, nutrient_index, top_k=1)
        if nutrient_matches:
            nutrient_details_formatted = format_nutrient_data(nutrient_matches)
            st.subheader("USDA Nutrient Details")
            st.markdown(nutrient_details_formatted)
            
            # Prepare and display nutrient chart
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
                df_chart = pd.DataFrame(list(nutrient_values.items()), columns=["Nutrient", "Value"]).set_index("Nutrient")
                st.subheader("Nutrient Chart")
                st.bar_chart(df_chart)
                
                # Create nutrient comparison graph using WHO recommendations
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
        
        # Chemical Information for each ingredient
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
        
        # Combined explanation via Together API using your prompt
        prompt_text = generate_prompt(
            food_name, 
            food_details_formatted, 
            nutrient_details_formatted, 
            ingredients_str
        )
        st.subheader("Chemical, Allergen, and Health Analysis")
        st.markdown("Generating explanation...")
        explanation = together_chat(prompt_text)
        st.markdown(explanation)

####################################
# Continuous Chat Interface using LangChain
####################################
st.markdown("---")
st.subheader("Continuous Chat")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_chat_input = st.text_input("Enter your follow-up query:")
if st.button("Send Chat") and user_chat_input:
    result = chat_chain({"question": user_chat_input})
    answer = result.get("answer", "No answer generated.")
    st.session_state.chat_history.append((user_chat_input, answer))
    
for i, (q, a) in enumerate(st.session_state.chat_history, start=1):
    st.markdown(f"**User ({i}):** {q}")
    st.markdown(f"**Assistant ({i}):** {a}")
