# NutriVision Pro: Your AI-Driven Nutritional Intelligence Hub 🚀
Bridging Industrial Food Science & Everyday Kitchen Wisdom 🌉

## Two Pillars of Innovation
### 1. USDA Food Database Engine 📊
- **Barcode/Text → Chemical Safety + Allergen Detection**
- Leverages 3 Pinecone indexes (Foods ↔ Nutrients ↔ Chemicals) for deep semantic search 🔍
- Identifies preservative risk (“TBHQ exceeds EU limits”) and alerts on common allergens (e.g., soy, nuts) 🚫

### 2. Home Food Intelligence 🏠
- **"Chicken & Fries" → Instant Calorie Math + Meal Plans → SMS Delivery**
- Fetches data from CalorieNinjas API and merges it with the LLM for recipe suggestions 🍗
- Offers leftover-based ideas, budget ingredient swaps, and allergen-free variations 🔄

## Core Technologies 🔧
- **AI Models:** Meta-Llama-3.1-8B (analysis), multilingual-e5-large (embeddings)
- **Data Infrastructure:** Pinecone (3 indexes: USDA foods, nutrients, chemicals)
- **APIs:** USDA FoodData Central, CalorieNinjas, Twilio (for SMS)
- **Visualization:** Plotly (radar charts, nutrient comparisons)
- **Workflow:** Streamlit + LangChain RAG pipeline for end-to-end orchestration 🛠️

## How It Works ⚙️
### For Packaged Foods (USDA Mode) 📦
- Scan a Barcode → Decodes FOOD_ID
- Pinecone Index Query → Retrieves industrial metadata (ingredients, nutrients)
- Chemical Index Cross-Reference → Flags preservatives (e.g., sodium benzoate)
- LLM Summaries, Chemical Notations, and Allergen Alerts

### For Homemade Meals (Kitchen Mode) 🍳
- Text a Dish (e.g., "Chicken rice bowl")
- CalorieNinjas API → Gets macros (protein, fat, carbs)
- LLM Crafts Meal Plans and SMS Shopping Lists 📲

## Unique Value 💡
### For Food Scientists & Manufacturers
- Chemical Risk Scores and Preservative Alerts 🛡️
- Exportable Reports: LLM insights in shareable PDFs 📄

### For Home Cooks & Dieticians
- Leftover Wizard and Allergy Guard 🧙‍♂️
- Budget Mode: Cheaper ingredient swaps without losing flavor 💸

## Try It Yourself 📱
- Scan a Snack Barcode 📸
- Type “600-calorie vegan lunch” 🌱
- Ask “Is this sauce safe for kids?” 👶

## Live Demo 🌍
Explore NutriVision Pro live on Hugging Face Spaces: [NutriVision Pro Live Demo](https://huggingface.co/spaces/Bhavesh7895/Food_assistant)

## Let’s Connect 🤝
- Interested in harnessing AI for next-gen food tech? Let’s discuss custom deployments or collaboration opportunities!

# Tags
- #AI #NutritionTech #FoodSafety #LLM #HealthTech #Innovation #FoodScience #MealPlanning #OpenSource


https://github.com/user-attachments/assets/1023185c-c7c0-4fc5-9631-f8fc41c4f532
