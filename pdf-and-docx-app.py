# Import libraries
import io
import streamlit as st

from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA

import PyPDF2
import docx

# ------------------------------------------------------------------------

# Function to generate response based on the uploaded file and user query
def generate_response(uploaded_file, openai_api_key, query_text):
    try:
        # Load document if file is uploaded
        if uploaded_file is not None:
            # Check file type and extract text accordingly
            if uploaded_file.type == 'application/pdf':
                # Handle PDF file
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                documents = [pdf_reader.pages[i].extract_text() for i in range(len(pdf_reader.pages))]
            elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                # Handle Word document
                doc_reader = docx.Document(io.BytesIO(uploaded_file.read()))
                documents = [para.text for para in doc_reader.paragraphs]
            else:
                # Default to plain text extraction
                documents = [uploaded_file.read().decode()]
            
            # Split documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.create_documents(documents)
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            # Create a vector store from documents
            db = Chroma.from_documents(texts, embeddings)
            
            # Create retriever interface
            retriever = db.as_retriever()
            
            # Create and run QA chain
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
            
            # Return the QA response
            return qa.run(query_text)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# ------------------------------------------------------------------------

# Streamlit configuration
st.set_page_config(page_title='TextSum: Simplify Your Documents 📑🔍 🩺🔗')
st.title("TextSum: Simplify Your Documents 📑🔍")
st.subheader("`Application by:` Aarish Khan and Asif Ali")
st.subheader("`Date:` 8th April 2024")

# ------------------------------------------------------------------------

# File upload widget to accept txt, pdf, and docx files
uploaded_file = st.file_uploader('Proceed further and upload the Required documents or files:', type=['txt', 'pdf', 'docx'])

# Query text input
query_text = st.text_input('Please provide your inquiry or question in the designated input field:', placeholder='...', disabled=not uploaded_file)

#  ---------------------------------------------------------------------

# List to store results
result = []

# ------------------------------------------------------------------------

# Defining a form using Streamlit to collect user input
with st.form('myform', clear_on_submit=True):
    # Creating a text input field in the sidebar to collect the OpenAI API key
    openai_api_key = st.sidebar.text_input('Provide your OpenAI Key!', type='password', disabled=not (uploaded_file and query_text))
    
    # Creating a submit button to trigger the form submission
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    
    # Upon submission and presence of API key, fetching the response and appending it to the results list
    if submitted and openai_api_key:
        # Displaying a spinner while the result is being processed
        with st.spinner('Checking the results...'):
            # Generating response using the provided file, OpenAI API key, and query text
            response = generate_response(uploaded_file, openai_api_key, query_text)
            
            # Appending the response to the result list
            if response:
                result.append(response)
                
# ------------------------------------------------------------------------