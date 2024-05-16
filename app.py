import os
import streamlit as st
from pathlib import Path
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain import hub
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, LLMChain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
import sys

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def retrieve_it(text):
    # # _______________Splitting the Document________________

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 10)
    docs = text_splitter.split_documents(text)

    # _____________Defining the embedding and creating the vector db__________

    embedding = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embedding)

    #_____________Defining the retriever_______________

    retriever = db.as_retriever()

    #____________________Building the full QA retrieval chain__________________

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

def answer(question, rag_chain):
    msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), msg["answer"]])
    source = msg["context"][1].metadata["source"]
    page = msg["context"][1].metadata["page"]
    ans = msg["answer"]
    output = f"Answer: {ans} \n" + "\n" + f"Source: {source} \n" + "\n" + f"Page: {page}"
    return output


llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
chat_history = []


st.title("RAG based QA")

st.sidebar.title("Document Uploader")
uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
length = st.sidebar.radio("Select answer length", ["Short", "Medium", "Long"])
dynamic = st.sidebar.checkbox("Use dynamic prompt")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("Ask a question"):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    if dynamic:
        prompting = """You need to create dynamic prompts for specific queries or tasks across various domains, assigning ChatGPT a specific role to play in generating responses. Follow the steps below to craft the prompt:

        Identify the Role: Determine the role ChatGPT should play in the prompt. This could be a profession, expertise, or specific skill set. It should be very specific to the query. 

        Generate the Prompt: Write a prompt that clearly communicates the task to ChatGPT, including the assigned role and any relevant details.

        Iterate as Needed: Review the prompt to ensure clarity and relevance. Make adjustments as necessary.

        Your goal is to create prompts that effectively guide ChatGPT in generating insightful and relevant responses based on the assigned role. Generate a prompt following the steps above.

        Query: {user_input}

        Give me the output strictly in the following format:

        Role: "" 

        Role should be within two words only.

        Short description of role: ""

        Final prompt: ""

        Don't mention your role in the final prompt.
        """
        zeroth_step = ChatPromptTemplate.from_template(prompting)
        chain0 = LLMChain(prompt=zeroth_step, llm=llm)
        res0 = chain0.invoke({"user_input": user_input})
        role = res0["text"].split('\n\nFinal prompt')[0].strip()
        final_prompt = res0["text"].split('Final prompt:')[1].strip()

        first_prompt = """You will be given a query. Identify the task told to be done in the query (without subjects).
        Here are some examples:
        Query: Create a pitch deck on Jasper.AI (a marketing copilot for startups)
        Task: Create a pitch deck

        Query: Write a Google ad for my automatic toothbrush
        Task: Write a Google ad

        Query: Write a LinkedIn post for my new certificate in Deep Learning
        Task: Write a LinkedIn post

        Query: {query}
        Task:
        """
        first_step = ChatPromptTemplate.from_template(first_prompt)
        chain1 = LLMChain(prompt=first_step, llm=llm,output_key="task")
        res1= chain1.invoke({"query": user_input})

        second_prompt = """You will be given a task. 
        Create a very detailed list of points to be covered by ChatGPT to execute this task. 
        Don't give much description. Just short bullet points.
        Prescribe gaurdrails ChatGPT needs to follow for doing this task.

        Task: {task}

        Strictly follow the output format given below:

        Points to be covered: 
        
        Guardrails:
        """

        second_step = ChatPromptTemplate.from_template(second_prompt)
        chain2 = LLMChain(prompt = second_step, llm=llm, output_key = "temp")
        res2 = chain2.invoke({"task": res1["task"]})

        third_prompt = """You will be given a task. 
        Create an ideal format of the output of the task.
        Keep in mind that this format will be given to ChatGPT to execute a the task.
        NEVER use json.

        Task: {task} 
        """

        third_step = ChatPromptTemplate.from_template(third_prompt)
        chain3 = LLMChain(prompt = third_step, llm=llm, output_key = "form")
        res3 = chain3.invoke({"task": res1["task"]})

        system_template = f"{role}\n\n{res1['task']} using the following rules:\n\n{res2['temp']}"
        system_message_prompt = SystemMessagePromptTemplate.from_template(f"{system_template}")
        # st.header("System Prompt:")
        # st.write(system_template)

        m_inst = "Response should also be formatted to markdown format."
        human_template="{context}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        chain4 = LLMChain(llm=llm, prompt=chat_prompt)

        if uploaded_files:
            text = []
            for file in uploaded_files:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name

                loader = None
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)

                if loader:
                    text.extend(loader.load())
                    os.remove(temp_file_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=10)
            text_chunks = text_splitter.split_documents(text)
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
            template_search= """Task: {ques}
            \n
            Write a set of 6 questions to ask about more information required to execute the task. Frame the questions as queries from a vector database.
            The questions should be very subjective to the task.
            Output should be a python list of strings. Only give the python list as output, nothing else.\n
            Output format:\n

            ["","","","","",""]
            """
            prompt_search = ChatPromptTemplate.from_template(template_search)
            chain_search = LLMChain(llm=llm, prompt=prompt_search)
            res_search = chain_search.invoke({"ques": user_input})
            ques_list = eval(res_search['text'])
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})
            prompt_template = """Given the following context and a question, generate an answer based on this context only.
            In the answer try to provide as much text as possible from  the source document context without making much changes.
            Only give answer to the question asked, nothing else.

            CONTEXT: {context}

            QUESTION: {question}"""

            PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
            )
            qa = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=retriever,    
                                            input_key="query",
                                            return_source_documents=True,
                                            chain_type_kwargs={"prompt": PROMPT})
            doc_text = ""
            for i in ques_list:
                docs = retriever.get_relevant_documents(i)
                tempo = qa.invoke({"query":i})
                doc_text += i+"\n\n"+tempo["result"]+"\n\n"+f"Source: {tempo['source_documents'][0].page_content}\n\nDocument name:{tempo['source_documents'][0].metadata['source']}\n\nPage number:{tempo['source_documents'][0].metadata['page']}"+"\n\n"
            final_prompt = final_prompt +"\n\nUse the information given below to craft your response:\n\n"+doc_text
            chain4 = LLMChain(llm=llm, prompt=chat_prompt)
        else:
            chain4 = LLMChain(llm=llm, prompt=chat_prompt)

    else:
        if uploaded_files:
            text = []
            for file in uploaded_files:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name

                loader = None
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)

                if loader:
                    text.extend(loader.load())
                    os.remove(temp_file_path)
            
            chain4 = retrieve_it(text)    
        else:
            s_prompt = """You are a chatbot like chatgpt. Whatever question is asked, assess and answer it.
            {context}"""
            simple_prompt = ChatPromptTemplate.from_template(s_prompt)
            chain4 = LLMChain(llm=llm, prompt=simple_prompt)

        # question_answer_chain = create_stuff_documents_chain(llm, chat_prompt)

        # rag_chain = create_retrieval_chain(question_answer_chain)

        # st.header("Human Prompt:")
        # st.write(f"{final_prompt}\n\nGive output in the following format using the context given above:\n\n{res3['form']}\n\nUse proper headings in ### wherever necessary.\n\nThe length of generated output should be {length.lower()}.")
        # st.write(res3["form"])
        
    if dynamic:
        result = chain4.invoke({"context": f"{final_prompt}\n\nGive output in the following format using the context given above:\n\n{res3['form']}\n\nUse proper headings in markdown format wherever necessary.\n\nThe length of generated output should be {length.lower()}."})
        # st.subheader("Final output:")
        # st.write(result["answer"])
        full_res = result['text']  
    else:
        result = chain4.invoke({"context": f"{user_input} \n\nThe length of generated output should be {length.lower()}."})
        full_res = result["text"] 

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(full_res)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_res})

    

    




