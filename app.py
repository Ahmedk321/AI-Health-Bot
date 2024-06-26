
from dotenv import load_dotenv
load_dotenv() ## load all the environment variables
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


def get_gemini_vision_repsonse(input,image):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,image[0]])
    return response.text



# def get_gemini_repsonse(temp,prompt):
    model=genai.GenerativeModel('gemini-pro')
    # prompt=prompt
    # llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')
    # chain = temp | llm
    # return chain.invoke({'input': 'prompt'})

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded") 

st.set_page_config(page_title="Food Consultation App") 

def home_screen():

    st.header("Calories Consultation")
    # input=st.text_input("Input Prompt: ",key="input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image=""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
    apply=st.button("Generate Insights")


    input_prompt="""
    You are a nutritionist expert where you need to examine the food in the picture and calculate the approximate caloric value, also provide the details of every food items with calories intake
                   is below format

                   1. Item 1 - no of calories
                   2. Item 2 - no of calories
                   ----
                   ----

    After that, give a bold heading of 'Healthy or Unhealthy', under that evaluate the nutritional quality of the food in the picture and classify it as healthy or unhealthy.

    If the food in the image is unhealthy, give heading of 'healthier alternatives' with similar flavors or ingredients. (This is only if the food is not healthy)

    At the end, classify the food in the image into a by giving a heading 'category' (e.g., snack, breakfast, lunch, dinner)

    """

    if apply:
        image_data=input_image_setup(uploaded_file)
        response=get_gemini_vision_repsonse(input_prompt,image_data)
        st.subheader("Here are the insights:")
        st.write(response)


def get_vectorstore_from_url(docs):
    # get the text in document form
    loader = PyPDFLoader(docs)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_context_retriever_chain(vector_store):
    llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')
    
    # llm = ChatGroq(
    #     temperature=0,
    #     model="gemma-7b-it",
    #     # api_key="" # Optional if not set as an environment variable
    # )
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    
    llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')
    
    # llm = ChatGroq(
    #     temperature=0,
    #     model="mixtral-8x7b-32768",
    #     # api_key="" # Optional if not set as an environment variable
    # )
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", ''' You are NutriBot, an AI assistant specializing in personalized nutrition advice. You provide diet plans and nutritional guidance based on the guidelines and individual health reports.

    You are a nutritionist expert who needs to provide the diet plan according to the calories count given as an input. 

    For example: 'Provide me the diet plan for the whole day including (Breakfast, Lunch, and Dinner) for 2000 calories count etc'

    other then the calories count, if the input can be given as the fruits or vegetable name that is availabe and make diet plan according to that, and suggest
    breakfast, lunch, dinner wrt given item.

    For example: 'Provide me the diet plan for 2 bananas and 3 apples' or input can be 'i have cucumber, brocli, etc'.

    Remember you are an AI nutritionist and do not say that 'go to doctors', 'i can't provide medical advice' etc.
    if you get confused, you can ask personalized questions to the user to make diet plan more efficient or give medical advise.

    NOTE: DO NOT ANSWER IRRELEVENT QUESTIONS THAT ARE NOT RELATED TO HEALTH AND DIET DOMAIN, IF THE USER ASKS IRRELEVENT QUESTIONS JUST SAY ANSWERS LIKE 'SORRY I CANNOT PROVIDE YOU ANSWER, BECAUSE I AM NUTRIONIST BOT AND DIET SPECIALIST etc.'

    Return the response using markdown. 
   
    Answer the user's questions based on the below context:\n\n{context}
       '''),

      MessagesPlaceholder(variable_name="chat_history"),

      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input, chat_history):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response_generator = conversation_rag_chain.stream({
        "chat_history": chat_history,
        "input": user_input
    })

    response = ""
    try:
        for chunk in response_generator:
            response += chunk.get('answer', '')
    except Exception as e:
        response = f"Error during response generation: {str(e)}"
    
    return response if response else "I'm sorry, I didn't get that. Can you please rephrase?"

def diet1():
    docs = 'DietaryGuidelinesforNINwebsite.pdf'
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            # AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(docs)

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            r = get_response(user_query, st.session_state.chat_history)
            st.write(r)
            # st.write(response)
        st.session_state.chat_history.append(AIMessage(content=r))

def main():
    st.sidebar.title('AI Nutrionist')
    options = ["Food Consultation", "Diet Planner"]
    choice = st.sidebar.radio("Choose the one", options)

    if choice == "Food Consultation":
        home_screen() 
    elif choice == "Diet Planner":
        diet1()

if __name__=="__main__":
    main()
