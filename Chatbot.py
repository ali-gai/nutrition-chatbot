from dotenv import load_dotenv
import os
import json
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set Streamlit page
st.set_page_config(page_title="Nutrition Assistant Chatbot", page_icon="🥗")
st.title("🥗 Nutrition Assistant - Healthy Meals & Diet Planning")

# Sidebar Controls
with st.sidebar:
    st.subheader("Chat Settings")
    # Health Goal Selection
    goal = st.selectbox(
    "🎯 Your Health Goal",
    ["None", "Weight Loss", "Muscle Gain", "Vegan", "Balanced Diet"]
)

    # Ingredient Input
    ingredients = st.text_area(
    "🧄 Ingredients You Have (comma-separated)",
    placeholder="e.g. eggs, spinach, tomatoes"
)

    
    model_name = st.selectbox(
        "Groq Model",
        ["deepseek-r1-distill-llama-70b", "gemma2-9b-it", "llama3-70b-8192"],
        index=2
    )
    
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 300, 150)
    
    user_prompt = st.text_area(
        "Custom Instructions (Optional)",
        value=""
    )

    st.caption("💡 Example: 'Focus on low-carb diets for weight loss.'")

    if st.button("🧹 Clear Chat"):
        st.session_state.pop("history", None)
        st.session_state.pop("memory", None)
        st.rerun()

# Check API key
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing. Add it to your .env file.")
    st.stop()

# Initialize memory and history
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Chat Input
user_input = st.chat_input("Ask me anything about diet, recipes, or nutrition...")

# LLM Setup
llm = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    groq_api_key=GROQ_API_KEY
)

# Default system prompt for Nutrition Assistant
default_prompt = (
    "You are a friendly and knowledgeable nutrition assistant. "
    "You help users plan healthy meals, suggest recipes based on ingredients, "
    "and build diet plans based on their goals. "
    "Always ask follow-up questions to understand their needs better."
)

# Setup conversation chain
conv = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

    # Handle Chat Turn
if user_input:
    st.session_state.history.append(("user", user_input))
    
    # Use custom or default system prompt
    system_prompt = user_prompt.strip() if user_prompt else default_prompt

    # Append goal and ingredient info
    goal_info = f"My health goal is: {goal}." if goal != "None" else ""
    ingredient_info = f"I currently have: {ingredients}." if ingredients.strip() else ""

    # Combine user context + message
    user_context = f"{goal_info} {ingredient_info}".strip()
    user_message = f"{user_context}\n\n{user_input}" if user_context else user_input

    # Final prompt
    full_prompt = f"[System]\n{system_prompt}\n\n[User]\n{user_message}"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = conv.predict(input=full_prompt)
        typed = ""
        for ch in response:
            typed += ch
            placeholder.markdown(typed)
            time.sleep(0.005)
        ai_response = typed

    st.session_state.history.append(("assistant", ai_response))

# Render Chat History
for role, text in st.session_state.history:
    st.chat_message(role).write(text)

# Download Chat
if st.session_state.history:
    save_data = [{"role": r, "text": t} for r, t in st.session_state.history]
    st.download_button(
        label="⬇️ Download Nutrition Chat",
        data=json.dumps(save_data, ensure_ascii=False, indent=2),
        file_name="nutrition_chat_history.json",
        mime="application/json"
    )
