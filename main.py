import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu

from gemini_utility import load_gemini_pro_model, gemini_pro_vision_response, embedding_model_response, \
    gemini_pro_response, gemini_pro_translate, gemini_pro_analyse, gemini_pro_ner, gemini_pro_anomaly, \
    gemini_pro_beautify, gemini_pro_correct, gemini_pro_summarise

# getting the working directory
working_directory = os.path.dirname(os.path.abspath(__file__))

# setting up the page configuration
st.set_page_config(page_title="Gemini Pro AI", page_icon="🧠", layout="wide")

with st.sidebar:
    selected = option_menu(
        "Gemini Pro AI",
        [
            "Introduction",
            "ChatBot",
            "Translate",
            "Image Captioning",
            "Text Embedding",
            "Sentiment Analysis",
            "Document Summarization",
            "Named Entity Recognition",
            "Anomaly Detection",
            "Code Beautifier",
            "Code Correction",
            "Query Resolver",
            "Connect"
        ],
        menu_icon="robot",
        icons=[
            "info-circle-fill",
            "chat-dots-fill",
            "translate",
            "image-fill",
            "textarea-t",
            "emoji-smile",
            "file-text",
            "person-badge-fill",
            "exclamation-triangle-fill",
            "file-code",
            "wrench",
            "patch-question-fill",
            "person-lines-fill"
        ],
        default_index=0,
    )


if selected == "Introduction":
    st.title("Multi-Functional AI Toolkit")
    st.write("Welcome to versatile AI-powered toolkit, built using Google's Gemini Pro, designed to streamline your tasks with cutting-edge technologies. ")
    st.write("Explore a range of functionalities through the sidebar navigation to discover how each tool can enhance your workflow.")

    st.write("")
    st.subheader("Tools Available:")
    st.write("- 🤖 **ChatBot:** Engage in interactive conversations.")
    st.write("- 🌍 **Translate:** Translate text between languages seamlessly.")
    st.write("- 🖼️ **Image Captioning:** Generate captions for images automatically.")
    st.write("- 📝 **Text Embedding:** Encode text into numerical representations.")
    st.write("- 😊 **Sentiment Analysis:** Analyze sentiment polarity in text.")
    st.write("- 📑 **Document Summarization:** Summarize lengthy documents quickly.")
    st.write("- 🔍 **Named Entity Recognition:** Identify entities within text.")
    st.write("- ⚠️**Anomaly Detection:** Detect outliers or anomalies in data.")
    st.write("- 💻 **Code Beautifier:** Format your code for readability.")
    st.write("- 🛠️ **Code Correction:** Automatically correct syntax errors in code.")
    st.write("- 🔮 **Query Resolver:** Resolve queries based on predefined knowledge.")
    st.markdown("""
    <div style='color: gray; font-size: 15px;'>
        Last updated on 29/06/2024.
    </div>
    """, unsafe_allow_html=True)

# function to translate role between gemini-pro and streamlit terminologies
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role


# chatbot page
if selected == "ChatBot":

    model = load_gemini_pro_model()

    # initialize chat session in streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # streamlit page title
    st.title("🤖 Virtual Assistant")

    # display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # input field for user's message
    user_prompt = st.chat_input("Ask Gemini-Pro...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)

        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        # display gemini pro response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

# image captioning page
if selected == "Image Captioning":

    # streamlit page title
    st.title("🖼️ Automated Image Description")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption"):

        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            resized_image = image.resize((800, 500))
            st.image(resized_image)

        default_prompt = "Write a short caption for this image"

        # getting the response from gemini-pro-vision model
        caption = gemini_pro_vision_response(default_prompt, image)

        with col2:
            st.info(caption)


# text embedding page
if selected == "Text Embedding":

    st.title("📝 Text Representation Tool")

    # input text box
    input_text = st.text_area(
        label="Enter text to get embeddings", placeholder="Type or paste your text here"
    )

    if st.button("Generate Embeddings"):
        response = embedding_model_response(input_text)
        st.success(response)

# question answering page
if selected == "Query Resolver":
    st.title("🔮 Knowledge Base")

    # text box to enter prompt
    user_prompt = st.text_area(
        label="Enter your question", placeholder="Ask Gemini Pro..."
    )

    if st.button("Get an answer"):
        response = gemini_pro_response(user_prompt)
        st.success(response)

# translate page
if selected == "Translate":
    st.title("🌍 Language Translation Tool")

    col1, col2 = st.columns(2)

    with col1:
        input_language_list = [
            "Abkhaz",
            "Acehnese",
            "Acholi",
            "Afar",
            "Afrikaans",
            "Albanian",
            "Alur",
            "Amharic",
            "Arabic",
            "Armenian",
            "Assamese",
            "Avar",
            "Awadhi",
            "Aymara",
            "Azerbaijani",
            "Balinese",
            "Baluchi",
            "Bambara",
            "Baoulé",
            "Bashkir",
            "Basque",
            "Batak Karo",
            "Batak Simalungun",
            "Batak Toba",
            "Belarusian",
            "Bemba",
            "Bengali",
            "Betawi",
            "Bhojpuri",
            "Bikol",
            "Bosnian",
            "Breton",
            "Bulgarian",
            "Buryat",
            "Cantonese",
            "Catalan",
            "Cebuano",
            "Chamorro",
            "Chechen",
            "Chichewa",
            "Chinese (Simplified)",
            "Chinese (Traditional)",
            "Chuukese",
            "Chuvash",
            "Corsican",
            "Crimean Tatar",
            "Croatian",
            "Czech",
            "Danish",
            "Dari",
            "Dhivehi",
            "Dinka",
            "Dogri",
            "Dombe",
            "Dutch",
            "Dyula",
            "Dzongkha",
            "English",
            "Esperanto",
            "Estonian",
            "Ewe",
            "Faroese",
            "Fijian",
            "Filipino",
            "Finnish",
            "Fon",
            "French",
            "Frisian",
            "Friulian",
            "Fulani",
            "Ga",
            "Galician",
            "Georgian",
            "German",
            "Greek",
            "Guarani",
            "Gujarati",
            "Haitian Creole",
            "Hakha Chin",
            "Hausa",
            "Hawaiian",
            "Hebrew",
            "Hiligaynon",
            "Hindi",
            "Hmong",
            "Hungarian",
            "Hunsrik",
            "Iban",
            "Icelandic",
            "Igbo",
            "Ilocano",
            "Indonesian",
            "Irish",
            "Italian",
            "Jamaican Patois",
            "Japanese",
            "Javanese",
            "Jingpo",
            "Kalaallisut",
            "Kannada",
            "Kanuri",
            "Kapampangan",
            "Kazakh",
            "Khasi",
            "Khmer",
            "Kiga",
            "Kikongo",
            "Kinyarwanda",
            "Kituba",
            "Kokborok",
            "Komi",
            "Konkani",
            "Korean",
            "Krio",
            "Kurdish (Kurmanji)",
            "Kurdish (Sorani)",
            "Kyrgyz",
            "Lao",
            "Latgalian",
            "Latin",
            "Latvian",
            "Ligurian",
            "Limburgish",
            "Lingala",
            "Lithuanian",
            "Lombard",
            "Luganda",
            "Luo",
            "Luxembourgish",
            "Macedonian",
            "Madurese",
            "Maithili",
            "Makassar",
            "Malagasy",
            "Malay",
            "Malay (Jawi)",
            "Malayalam",
            "Maltese",
            "Mam",
            "Manx",
            "Maori",
            "Marathi",
            "Marshallese",
            "Marwadi",
            "Mauritian Creole",
            "Meadow Mari",
            "Meiteilon (Manipuri)",
            "Minang",
            "Mizo",
            "Mongolian",
            "Myanmar (Burmese)",
            "Nahuatl (Eastern Huasteca)",
            "Ndau",
            "Ndebele (South)",
            "Nepalbhasa (Newari)",
            "Nepali",
            "NKo",
            "Norwegian",
            "Nuer",
            "Occitan",
            "Odia (Oriya)",
            "Oromo",
            "Ossetian",
            "Pangasinan",
            "Papiamento",
            "Pashto",
            "Persian",
            "Polish",
            "Portuguese (Brazil)",
            "Portuguese (Portugal)",
            "Punjabi (Gurmukhi)",
            "Punjabi (Shahmukhi)",
            "Quechua",
            "Qʼeqchiʼ",
            "Rajasthani",
            "Rarotongan",
            "Riffian",
            "Romani",
            "Romanian",
            "Romansh",
            "Rundi",
            "Russian",
            "Sami (North)",
            "Samoan",
            "Sango",
            "Sanskrit",
            "Santali",
            "Scots Gaelic",
            "Sepedi",
            "Serbian",
            "Sesotho",
            "Seychellois Creole",
            "Shan",
            "Shona",
            "Sicilian",
            "Silesian",
            "Sindhi",
            "Sinhala",
            "Slovak",
            "Slovenian",
            "Somali",
            "Spanish",
            "Sundanese",
            "Susu",
            "Swahili",
            "Swati",
            "Swedish",
            "Tahitian",
            "Tajik",
            "Tamazight",
            "Tamazight (Tifinagh)",
            "Tamil",
            "Tatar",
            "Telugu",
            "Tetum",
            "Thai",
            "Tibetan",
            "Tigrinya",
            "Tiv",
            "Tok Pisin",
            "Tongan",
            "Tsonga",
            "Tswana",
            "Tulu",
            "Tumbuka",
            "Turkish",
            "Turkmen",
            "Tuvan",
            "Twi",
            "Udmurt",
            "Ukrainian",
            "Urdu",
            "Uyghur",
            "Uzbek",
            "Venda",
            "Venetian",
            "Vietnamese",
            "Waray",
            "Welsh",
            "Wolof",
            "Xhosa",
            "Yakut",
            "Yiddish",
            "Yoruba",
            "Yucatec Maya",
            "Zapotec",
            "Zulu",
        ]

        input_language = st.selectbox(
            label="Select Input Language", options=input_language_list
        )

    with col2:
        output_language_list = [x for x in input_language_list if x != input_language]
        output_language = st.selectbox(
            label="Select Output Language", options=output_language_list
        )

    input_text = st.text_area(
        "Enter text to be translated", placeholder="Enter text here"
    )

    if st.button("Translate Text"):
        translated_text = gemini_pro_translate(
            input_text, input_language, output_language
        )
        st.success(translated_text)


# sentiment analysis page
if selected == "Sentiment Analysis":
    st.title("😊 Sentiment Analysis Tool")

    text = st.text_area(
        label="Enter text for sentiment analysis", placeholder="Enter text here"
    )

    if st.button("Analyze Sentiment"):
        response = gemini_pro_analyse(text)
        st.success(response)

# document summarization page
if selected == "Document Summarization":
    st.title("📑 Document Summarizer")

    text = st.text_area(label="Enter text to summarize", placeholder="Enter text here")

    if st.button("Generate Summary"):
        response = gemini_pro_summarise(text)
        st.success(response)

# named entity recognition page
if selected == "Named Entity Recognition":
    st.title("🔍 Named Entity Recognition Tool")

    text = st.text_area(
        label="Enter text for entity recognition", placeholder="Enter text here"
    )

    if st.button("Detect Entities"):
        response = gemini_pro_ner(text)
        st.success(response)

# anomaly detection page
if selected == "Anomaly Detection":
    st.title("⚠️ Anomaly Detection Tool")

    text = st.text_area(
        label="Enter data for anomaly detection", placeholder="Enter data here"
    )

    if st.button("Detect Anomalies"):
        response = gemini_pro_anomaly(text)
        st.success(response)

# code beautifier page
if selected == "Code Beautifier":
    st.title("💻 Code Beautifier Tool")

    text = st.text_area(label="Enter code to beautify", placeholder="Enter code here")

    if st.button("Beautify Code"):
        response = gemini_pro_beautify(text)
        st.success(response)

# code correction page
if selected == "Code Correction":
    st.title("🛠️ Code Correction Tool")

    text = st.text_area(label="Enter code to correct", placeholder="Enter code here")

    if st.button("Correct Code"):
        response = gemini_pro_correct(text)
        st.success(response)

# connect page
if selected == "Connect":
    import streamlit as st

    st.title("Connect with Me")
    st.markdown("""
        Feel free to reach out to me via email or connect with me on LinkedIn and GitHub! 💬
    """)
    st.write("")

    st.subheader("Contact Information 📩")
    st.write("**Name:** Aryan Shah")
    st.write("**Email:** aryanshah1957@gmail.com")
    st.write("")

    st.subheader("Social Media 🌍")
    st.write("[LinkedIn](https://www.linkedin.com/in/aryanashah/) 🔗")
    st.write("[GitHub](https://github.com/AryanShah30) 🔗")
