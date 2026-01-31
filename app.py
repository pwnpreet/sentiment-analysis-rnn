import streamlit as st
import numpy as np
import pickle 
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences 

# ================ LOAD MODEL ===================
model = load_model("review_rnn.keras")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer= pickle.load(f)
with open('tokenizer2.pkl', 'rb') as f:
    tokenizer2= pickle.load(f)

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="RNN Streamlit App",
    page_icon="🧠",
    layout="wide"
)

# ================= LOAD CSS =================
st.markdown("""
    <style>
    .stApp {background-color: #E6E6E6;}
    
    div.stButton > button {
            background-color: #574964;
            color: white;
            border-radius:8px;
            padding: 0.5em 1em;
            }
    div.stButton > button:hover {
            background-color: #6b5b60;
            color: white;
            }
    </style>
    """,unsafe_allow_html=True
)

# ================= SESSION STATE =================
if 'users' not in st.session_state:
    st.session_state.users= {}
if 'current_user' not in st.session_state:
    st.session_state.current_user= None
if 'page' not in st.session_state:
    st.session_state.page= 'auth'

# ================= FIRST PAGE =================
def auth_page():
    st.title("Sentiment Analysis Portal💭")
    st.write('Welcome to our Review analysis application🎉')
    st.info('To access this portal you have to login.🔐')
    st.info('New to our Portal?🤔 Please SignUp first...')

    col1, col2, col3= st.columns([1,2,1])
    with col2:
        login, signup= st.columns([1,1])
        if login.button("Login", use_container_width=True, key='lg_btn'):
            st.session_state.auth_mode= 'login'
        if signup.button('SignUp', use_container_width=True, key='sup_btn'):
            st.session_state.auth_mode= 'signup'
        # LOGIN SECTION 
        if st.session_state.get('auth_mode')== 'login':
            st.subheader("Welcome Back!☺️")
            user= st.text_input('Username', key= 'lg_user')
            pwd= st.text_input('Password', type='password', key= 'lg_pwd')
            if st.button('Login'):
                if user in st.session_state.users and st.session_state.users[user]==pwd:
                    st.session_state.current_user= user
                    st.session_state.page= 'home'
                    st.success(f'Logged in as {user}')
                    st.rerun()
                else:
                    st.error('Invalid username or pasword🫤')
        # SIGNUP SECTION 
        elif st.session_state.get('auth_mode')== 'signup':
            st.subheader('🌟Welcome to our Portal🌟')
            new_user= st.text_input('Create Username', key='sup_user')
            new_pwd= st.text_input('Create Password', type='password', key='sup_pwd')
            if st.button('SignUp'):
                if new_user in st.session_state.users:
                    st.error('Username already exist!')
                else:
                    st.session_state.users[new_user]= new_pwd
                    st.success('Conguralations!🎉, Account is created.')
                    st.success('Now you can logged in🤗')
                    st.session_state.auth_mode= 'login'

# ================= NAVIGATION =================
def home_page():
    page = option_menu(
        None,
        ["Home", "Prediction", "Flowchart", "About"],
        icons=["house", "activity", "diagram-3", "info-circle"],
        orientation="horizontal",
        styles={
            "container": {
                "padding": "5px",
                "background-color": "#574964"
            },
            "nav-link": {
                "font-size": "15px",
                "color": "#9F8383",
                "margin": "0px 3px",
                "text-transform": "capitalize"
            },
            "nav-link-selected": {
                "background-color": "#9F8383",
                "color": "#574964"
            }
        }
    )

    # ================= HOME =================
    if page == "Home":
        st.title("Recurrent Neural Network (RNN)")

        st.write("""
        A **Recurrent Neural Network (RNN)** is designed for **sequential data**
        where previous information helps predict future output.
        """)

        st.subheader("Why RNN?")
        st.write("""
        • Works with sequence data  
        • Maintains memory using hidden states  
        • Used in NLP and time-series analysis  
        """)

        st.subheader('Basic Working Idea of RNN model')
        cola, colb, colc =st.columns(3)
        cola.info('A RNN processes input sequences one step at a time and maintain hidden state that' \
        'carries information from previous step. This recurrent connection allows the network to ' \
        'remember past context while processing the current input')
        colb.info('The same set of weights is used at every time step in RNN. this weight sharing helps the model learn sequential' \
        'patterns and reduces the total number of parameters compared input.')
        colc.info('At each time step, the RNN updates its hidden state and can produce an output. The final' \
        'hidden state is used for tasks like sequence classification or language modeling.')


    # ================= PREDICTION PAGE =================
    elif page == "Prediction":
        st.title("📈 Review Prediction (Demo)")
        st.write("Compare predictions based on different review sizes.")

        col1, col2 = st.columns(2)
        # -------- COLUMN 1 : 50 REVIEWS --------
        with col1:
            st.subheader("📝 50 Reviews")
            review_50 = st.text_area(
                "Enter reviews (50)",
                height=180,
                placeholder="Enter or paste 50 reviews here..."
            )
            if st.button("Predict (50 Reviews)"):
                if review_50.strip() == "":
                    st.warning("Please enter reviews")
                else:
                    seq2= tokenizer2.texts_to_sequences([review_50])
                    padded2= pad_sequences(seq2, maxlen=50)
                    pred2= model.predict(padded2)

                    if pred2 >= 0.5:
                        st.success(f"✅ Positive Review ({pred2})")
                    else:
                        st.error(f"❌ Negative Review ({pred2})")
        # -------- COLUMN 2 : 150 REVIEWS --------
        with col2:
            st.subheader("📝 150 Reviews")
            review_150 = st.text_area(
                "Enter reviews (150)",
                height=180,
                placeholder="Enter or paste 150 reviews here..."
            )
            if st.button("Predict (150 Reviews)"):
                if review_150.strip() == "":
                    st.warning("Please enter reviews")
                else:
                    seq= tokenizer.texts_to_sequences([review_150])
                    padded= pad_sequences(seq, maxlen=150)
                    pred= model.predict(padded)

                    if pred >= 0.5:
                        st.success(f"✅ Positive Review ({pred})")
                    else:
                        st.error(f"❌ Negative Review ({pred})")
        st.info("⚠️ This is a demo prediction for UI and project presentation.")

    # ================= FLOWCHART =================
    elif page == "Flowchart":
        st.title("🔁 RNN Flowchart")
        st.code("""
    Start
    ↓
    Input Sequence (x₁, x₂, x₃)
    ↓
    Initialize Hidden State (h₀)
    ↓
    For each time step:
        Combine Input + Previous Memory
        Apply Activation Function
        Update Hidden State
    ↓
    Generate Output
    ↓
    Calculate Loss
    ↓
    Backpropagation Through Time
    ↓
    Update Weights
    ↓
    End
    """)

    # ================= ABOUT =================
    elif page == "About":
        st.subheader("🎯 Objective")
        st.write("""
        • Understand RNN working  
        • Demonstrate sequence prediction  
        • Build an interactive Streamlit UI  
        """)
        st.subheader("📌 Applications of RNN")
        st.write("""
        • Text prediction  
        • Speech recognition  
        • Time-series forecasting  
        • Chatbots  
        """)
        st.subheader("🛠️ Technologies Used")
        st.write("""
        • Python  
        • Streamlit  
        • HTML & CSS  
        """)
        st.markdown("---")
        st.markdown("### 👩‍💻 Developed By")
        st.write("**Pawanpreet Kaur**")
# ================ SESSION STATE ==============
if st.session_state.page== 'home':
    home_page()
else:
    auth_page()
