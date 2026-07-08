import os
import re
import streamlit as st
from dotenv import load_dotenv
from google import genai

# ==============================
# NEW RAG IMPORT
# ==============================
from rag_engine import CouncilRAG

# ======================================
# ENV
# ======================================
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ==============================
# LOAD RAG ENGINE
# ==============================
rag = CouncilRAG()
rag.build_index(data_path="data")

# ======================================
# PAGE
# ======================================
st.set_page_config(
    page_title="UK Council AI Assistant",
    page_icon="🏛️"
)

st.title("🏛️ UK Council AI Assistant")

st.markdown("""
Describe your council problem naturally.


""")

# ======================================
# SESSION MEMORY
# ======================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ======================================
# ACTION LINKS
# ======================================
ACTION_LINKS = {

    "pay_council_tax": {
        "title": "Pay Council Tax",
        "link": "https://www.gov.uk/pay-council-tax"
    },

    "student_discount": {
        "title": "Student Council Tax Discount",
        "link": "https://www.gov.uk/council-tax/discounts-for-full-time-students"
    },

    "report_missed_bin": {
        "title": "Report Missed Bin Collection",
        "link":"https://www.gov.uk/missed-bin-collection"

    },

    "bins_info": {
        "title": "Bins and Recycling",
        "link": "https://www.gov.uk/bins-waste-recycling"
    },

    "housing_repair": {
        "title": "Report Council Housing Repair",
        "link": "https://www.gov.uk/council-housing/complaints"
    },

    "find_council": {
        "title": "Find Your Local Council",
        "link": "https://www.gov.uk/find-local-council"
    }
}

# ======================================
# INFORMATION RESPONSES
# ======================================
INFO_RESPONSES = {

    "council_tax_penalty": """
If you do not pay council tax:

• The council may send reminder notices
• Extra charges can be added
• Legal action may happen
• Bailiffs could recover unpaid debt
• Your wages or benefits may be affected

If you are struggling financially, contact your local council immediately for support.
""",

    "council_tax_info": """
Council tax is a local tax collected by councils in the UK.

It helps pay for:
• Waste collection
• Police and fire services
• Schools and roads

Most adults over 18 living in a property must pay council tax.
""",

    "recycling_info": """
Recycling rules depend on your local council.

Usually recyclable:
• Paper
• Cardboard
• Plastic bottles
• Metal cans
• Glass bottles

Food waste and general rubbish should be separated.
""",

    "housing_info": """
The council usually repairs:
• Roof damage
• Plumbing issues
• Heating systems
• Structural problems

Tenants are normally responsible for minor internal damage.
"""
}

# ======================================
# SMART INTENT DETECTION
# ======================================
def detect_intent(user_input):

    text = user_input.lower()

    # =================================
    # ACTION INTENTS
    # =================================

    if any(x in text for x in [
        "pay council tax",
        "pay my tax",
        "pay bill",
        "council tax payment"
    ]):

        return {
            "type": "action",
            "intent": "pay_council_tax"
        }

    if any(x in text for x in [
        "student discount",
        "student exemption",
        "student council tax"
    ]):

        return {
            "type": "action",
            "intent": "student_discount"
        }

    if any(x in text for x in [
        "missed bin",
        "bin not collected",
        "rubbish not collected",
        "waste collection missed"
    ]):

        return {
            "type": "action",
            "intent": "report_missed_bin"
        }

    if any(x in text for x in [
        "housing repair",
        "report repair",
        "water leak",
        "repair issue",
        "mould",
        "mold"
    ]):

        return {
            "type": "action",
            "intent": "housing_repair"
        }

    if any(x in text for x in [
        "find council",
        "contact council",
        "local council"
    ]):

        return {
            "type": "action",
            "intent": "find_council"
        }

    # =================================
    # INFORMATION INTENTS
    # =================================

    if any(x in text for x in [
        "not pay council tax",
        "dont pay council tax",
        "don't pay council tax",
        "late council tax",
        "miss council tax"
    ]):

        return {
            "type": "information",
            "intent": "council_tax_penalty"
        }

    if any(x in text for x in [
        "what is council tax",
        "about council tax",
        "who pays council tax"
    ]):

        return {
            "type": "information",
            "intent": "council_tax_info"
        }

    if any(x in text for x in [
        "recycling rules",
        "what goes in recycling",
        "how recycling works"
    ]):

        return {
            "type": "information",
            "intent": "recycling_info"
        }

    if any(x in text for x in [
        "who fixes repairs",
        "repair responsibility"
    ]):

        return {
            "type": "information",
            "intent": "housing_info"
        }

    # =================================
    # GENERAL
    # =================================

    return {
        "type": "general",
        "intent": "general_help"
    }

# ======================================
# AI RESPONSE GENERATOR
# ======================================
def generate_ai_response(user_input):

    # ==============================
    # NEW RAG CONTEXT
    # ==============================
    docs = rag.query(user_input)

    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    prompt = f"""
You are a smart UK council assistant.

Use the council information below to answer.

Council Information:
{context}

User question:
{user_input}

Write:
- clear helpful answer
- maximum 5 lines
- understand broken English
- explain naturally
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text.strip()

    except:
        return "Sorry, I could not process your request."

# ======================================
# FINAL RESPONSE BUILDER
# ======================================
def build_response(user_input):

    result = detect_intent(user_input)

    response_type = result["type"]
    intent = result["intent"]

    # =================================
    # ACTION RESPONSE
    # =================================

    if response_type == "action":

        item = ACTION_LINKS[intent]

        title = item["title"]
        link = item["link"]

        ai_text = generate_ai_response(user_input)

        return f"""
{ai_text}

🔗 Official Link:
{link}
"""

    # =================================
    # INFORMATION RESPONSE
    # =================================

    if response_type == "information":

        info = INFO_RESPONSES[intent]

        return info

    # =================================
    # GENERAL RESPONSE
    # =================================

    return generate_ai_response(user_input)

# ======================================
# DISPLAY CHAT
# ======================================
for role, message in st.session_state.messages:

    with st.chat_message(role):
        st.markdown(message)

# ======================================
# USER INPUT
# ======================================
user_input = st.chat_input(
    "Describe your council problem..."
)

if user_input:

    st.chat_message("user").markdown(user_input)

    st.session_state.messages.append(
        ("user", user_input)
    )

    # generate response
    answer = build_response(user_input)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        ("assistant", answer)
    )