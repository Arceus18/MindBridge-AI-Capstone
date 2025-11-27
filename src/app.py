import streamlit as st
import joblib
from rag_engine import load_retriever

# 1. Load Resources
st.set_page_config(page_title="MindBridge AI", page_icon="🧠")

@st.cache_resource
def get_resources():
    # Load Safety Model
    safety_model = joblib.load("models/crisis_classifier.pkl")
    # Load Empathy Database
    vector_db = load_retriever()
    return safety_model, vector_db

classifier, vector_db = get_resources()

# 2. UI Layout
st.title("🧠 MindBridge AI")
st.markdown("### Your Safe Space for Mental Health Support")
st.info("This is an AI Assistant using RAG (Retrieval Augmented Generation). It retrieves professional advice from past counselor transcripts.")

user_input = st.text_area("How are you feeling today?", height=100)

if st.button("Get Support"):
    if user_input:
        # Step A: Safety Check
        # Assuming 1 = Crisis/Toxic, 0 = Safe
        prediction = classifier.predict([user_input])[0]
        
        if prediction == 1:
            st.error("⚠️ CRISIS DETECTED")
            st.write("It seems you are going through a very difficult time. I am an AI and cannot provide emergency help.")
            st.markdown("**Please call a suicide prevention hotline immediately: 988 (USA) or your local emergency number.**")
        
        else:
            # Step B: Retrieve Advice (RAG)
            st.success("Finding the best advice for you...")
            
            # Search the database for the 3 most similar past conversations
            results = vector_db.similarity_search(user_input, k=3)
            
            # Display the best match
            best_response = results[0].page_content
            
            st.markdown("### 🤖 MindBridge Suggests:")
            st.write(best_response)
            
            with st.expander("See other perspectives"):
                st.write(f"**Alternative 1:** {results[1].page_content}")
                st.write(f"**Alternative 2:** {results[2].page_content}")

    else:
        st.warning("Please type something to start.")

# Footer
st.markdown("---")
st.caption("Disclaimer: This is a Capstone Project AI. Not a replacement for a doctor.")