import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

load_dotenv()

page_sidebar = f"""
<style>
[data-testid="stSidebar"] {{
background: #3AE8AD;
}}
</style>
"""
page_head = f"""
<style>
[data-testid="stHeader"] {{
background: #1751B2;
}}
</style>
"""

page_main = f"""
<style>
</style>
"""

st.markdown(page_main, unsafe_allow_html=True)
st.markdown(page_head, unsafe_allow_html=True)
logo = "RootsFoodGroup_White_Tree_Logo.png"
with st.container():
    st.image(logo,use_column_width=True)
st.markdown("<h1 style='text-align: center;'>HIGH-QUALITY, ALL-NATURAL FOOD PRODUCTS</h1>", unsafe_allow_html=True)

with st.sidebar:
    icon = "TECHMENT_LOGO1.png"  # Replace this with the path to your image file
    st.image(icon, use_column_width=True)
    st.title('💬 Ask your question')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    st.markdown(page_sidebar,unsafe_allow_html=True)
    add_vertical_space(5)
    st.write('@Made by Techment technology')


if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
# if 'model_name' not in st.session_state:
#     st.session_state['model_name'] = []
# if 'cost' not in st.session_state:
#     st.session_state['cost'] = []
# if 'total_tokens' not in st.session_state:
#     st.session_state['total_tokens'] = []
# if 'total_cost' not in st.session_state:
# #     st.session_state['total_cost'] = 0.0
# # Sidebar contents
# st.sidebar.title("Sidebar")
# model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
# counter_placeholder = st.sidebar.empty()
# counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
# clear_button = st.sidebar.button("Clear Conversation", key="clear")

# # Map model names to OpenAI model IDs
# if model_name == "GPT-3.5":
#     model = "gpt-3.5-turbo"
# else:
#     model = "gpt-4"

# # reset everything
# if clear_button:
#     st.session_state['generated'] = []
#     st.session_state['past'] = []
#     st.session_state['messages'] = [
#         {"role": "system", "content": "You are a helpful assistant."}
#     ]
#     st.session_state['number_tokens'] = []
#     st.session_state['model_name'] = []
#     st.session_state['cost'] = []
#     st.session_state['total_cost'] = 0.0
#     st.session_state['total_tokens'] = []
#     counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    
 
def main():
    st.header("Chat with PDF 💬")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            # st.write(response)

            st.session_state['messages'].append({"role": "user", "content": query})
            st.session_state['messages'].append({"role": "assistant", "content": response})

        response_container = st.container()
        # container for text box
        container = st.container()

        with container:
            if query:
                output = response
                st.session_state['past'].append(query)
                st.session_state['generated'].append(output)
                # st.session_state['model_name'].append(model_name)
                # st.session_state['total_tokens'].append(total_tokens)

                # from https://openai.com/pricing#language-models
                # if model_name == "GPT-3.5":
                #     cost = total_tokens * 0.002 / 1000
                # else:
                #     cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

                # st.session_state['cost'].append(cost)
                # st.session_state['total_cost'] += cost

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))
                    # st.write(
                    #     f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
                    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

 
if __name__ == '__main__':
    main()