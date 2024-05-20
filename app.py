import os
import streamlit as st
from pathlib import Path
import tempfile
import uuid
import time
import pandas as pd
from langsmith import Client
from langchain import callbacks
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

os.environ['LANGCHAIN_TRACING_V2'] = "false"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
chat_history = []
chain_id = {}

def retrieve_it(text):
    # loader = PyPDFLoader(save_path)
    # pages = loader.load()

    # # _______________Splitting the Document________________

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 10)
    docs = text_splitter.split_documents(text)

    # _____________Defining the embedding and creating the vector db__________

    embedding = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embedding)

    #_____________Defining the retriever_______________

    retriever = db.as_retriever(search_kwargs={"k":1})

    #______________Contextualizing the question using history________________

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

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

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def answer(question, rag_chain):
    msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), msg["answer"]])
    source = msg["context"][1].metadata["source"]
    page = msg["context"][1].metadata["page"]
    ans = msg["answer"]
    output = f"Answer: {ans} \n" + "\n" + f"Source: {source} \n" + "\n" + f"Page: {page}"
    return output

def create_dynamic_prompt(user_input):
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
    with callbacks.collect_runs() as cb:
        res0 = chain0.invoke({"user_input": user_input})
        run_id = cb.traced_runs[0].id
        chain_id["chain0"] = run_id
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
    with callbacks.collect_runs() as cb:
        res1= chain1.invoke({"query": user_input})
        run_id = cb.traced_runs[0].id
        chain_id["chain1"] = run_id

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
    with callbacks.collect_runs() as cb:
        res2 = chain2.invoke({"task": res1["task"]})
        run_id = cb.traced_runs[0].id
        chain_id["chain2"] = run_id

    third_prompt = """You will be given a task. 
    Create an ideal format of the output of the task.
    Keep in mind that this format will be given to ChatGPT to execute a the task.
    NEVER use json.

    Task: {task} 
    """

    third_step = ChatPromptTemplate.from_template(third_prompt)
    chain3 = LLMChain(prompt = third_step, llm=llm, output_key = "form")
    with callbacks.collect_runs() as cb:
        res3 = chain3.invoke({"task": res1["task"]})
        run_id = cb.traced_runs[0].id
        chain_id["chain3"] = run_id

    system_template = f"{role}\n\n{res1['task']} using the following rules:\n\n{res2['temp']}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(f"{system_template}")
    # st.header("System Prompt:")
    # st.write(system_template)

    m_inst = "Response should also be formatted to markdown format."
    human_template="{context}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt, final_prompt, res3

def extraxt_doc_text(text, uploaded_files):
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
    return text

def retrieve_questions(text, user_input):
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
    with callbacks.collect_runs() as cb:
        res_search = chain_search.invoke({"ques": user_input})
        run_id = cb.traced_runs[0].id
        chain_id["chain5"] = run_id
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
        with callbacks.collect_runs() as cb:
            tempo = qa.invoke({"query":i})
            run_id = cb.traced_runs[0].id
            chain_id["chain6"] = run_id
        doc_text += i+"\n\n"+tempo["result"]+"\n\n"+f"Source: {tempo['source_documents'][0].page_content}\n\nDocument name:{tempo['source_documents'][0].metadata['source']}\n\nPage number:{tempo['source_documents'][0].metadata['page']}"+"\n\n"
    final_prompt = final_prompt +"\n\nUse the information given below to craft your response:\n\n"+doc_text
    text, final_prompt

def get_chain_data(chain_id):
    id_list = []
    for i in chain_id.keys():
        id_list.append(str(chain_id[i]))

    # print(id_list)

    time.sleep(2)
    client = Client()
    runs = client.list_runs(run_ids=id_list)
    # print(runs)
    lis =[]
    for r in runs:
        run_stats = {}
        run_stats['name'] = r.name
        run_stats['time'] =(r.end_time - r.start_time).total_seconds()
        run_stats['prompt_tokens'] =r.prompt_tokens
        run_stats['completion_tokens'] =r.completion_tokens
        run_stats['total_tokens'] =r.total_tokens

        lis.append(run_stats)
    # print(lis)

    time_taken = 0
    tokens = 0
    df = pd.DataFrame()
    for i in lis:
        time_taken += i['time']
        tokens += i["total_tokens"]

    df = pd.DataFrame(lis)
    lst = []
    for i in range(df.shape[0]):
        df["name"][i] = str(df["name"][i]) + f" {df.shape[0] - i - 1}"
        lst.append(df["time"][i]*1000/df["total_tokens"][i])

    df["time per token (in ms)"] = lst
    df = df[::-1]
    df.set_index(df.columns[0], inplace=True)
    return df, time_taken

def main():
    history = ""
    with open(r"history.txt", "r") as file:
        stored_history = file.read()
    # print(stored_string)
    st.title("Query Chatbot")

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
            chat_prompt, final_prompt, res3 = create_dynamic_prompt(user_input)
            chain4 = LLMChain(llm=llm, prompt=chat_prompt)

            if uploaded_files:
                text = []
                text = extraxt_doc_text(text, uploaded_files)
                text, final_prompt = retrieve_questions(text, user_input)
                chain4 = LLMChain(llm=llm, prompt=chat_prompt)
            else:
                chain4 = LLMChain(llm=llm, prompt=chat_prompt)

        else:
            if uploaded_files:
                text = []
                text = extraxt_doc_text(text)
                chain4 = retrieve_it(text)
            else:
                s_prompt = """You are a chatbot like chatgpt. Whatever question is asked, assess and answer it.
                {context}"""
                simple_prompt = ChatPromptTemplate.from_template(s_prompt)
                chain4 = LLMChain(llm=llm, prompt=simple_prompt)

        # question_answer_chain = create_stuff_documents_chain(llm, chat_prompt)

        # rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # st.header("Human Prompt:")
        # st.write(f"{final_prompt}\n\nGive output in the following format using the context given above:\n\n{res3['form']}\n\nUse proper headings in ### wherever necessary.\n\nThe length of generated output should be {length.lower()}.")
        # st.write(res3["form"])
        note1 = "Your goal is to be a friendly and conversational chatbot. Make users feel welcome and engaged in a warm manner. Whenever a user asks a question, respond with helpful information, incorporating conversational elements to create an enjoyable interaction. Use a warm and inviting tone, and include small talk or ask follow-up questions to keep the conversation flowing naturally. Follow this format: \n\n 1. Start with a Friendly Greeting: Begin your response with a warm, engaging, and conversational greeting. Change this everytime according to the context, don't be monotonous. \n\n 2. Answer the Query: Provide a helpful and informative answer to the user’s question. \n\n 3. End on a Positive Note: Conclude with a friendly remark and ask a relevant follow-up question to keep the conversation going. Don't keep it same everytime, try changing a bit with the context. For example, \n\n Query: \n\n Write a sales email. \n\n Your Response: \n\n 'Hi there! I'm happy to help you craft a great sales email today. Here’s a sample email for you: \n\n [Your generated answer] \n\n Feel free to personalize this email to better match your style and the recipient's needs. Is there anything else you’d like to add or ask about? I’m here to help!' \n\n Also keep these principles in mind: \n\n 1. Warm Greetings: Starting with a warm greeting sets a positive tone. \n\n 2. Empathy: Trying to understand and relate to the user's needs and emotions. \n\n 3. Conversational Tone: Using a conversational and approachable language rather than a formal or technical one. \n\n 4. Personalization: Tailoring responses to make them feel more personal and relevant to the user's query. \n\n 5. Encouragement: Encouraging users to ask more questions and engage further by showing genuine interest in helping them."
        history_prompt = f" \n\n Here is the chat history between you and the user. Everytime a user asks a query try to relate it to this chat history and generate a better answer using both. If the question is clear enough then don't use the chat history but try to relate. \n\n {stored_history}"
        note = "You are a friendly and conversational chatbot. Your goal is to make users feel welcome and engaged in a friendly manner. Whenever a user asks a question, respond with helpful information, and include some conversational elements to make the interaction enjoyable. Use a warm and inviting tone, and feel free to include small talk or ask follow-up questions to keep the conversation flowing naturally. Follow this format: \n\n 1. Start your response with a friendly greeting.Use a warm, engaging, and conversational tone. \n\n 2. Answer the query. \n\n 3. End on a good note. Ask relevant follow-up questions to keep the conversation going. \n\n  For example, \n\n query: write a sales email. \n\n your output: Hi user, I am happy to assist you today. Here is the sales email. \n\n your generated answer \n\n Feel free to personalize this email further to better fit your style and the recipient's specific needs. I am here to help you if you have any further query"
        if dynamic:

            # final_input = f"{final_prompt}\n\nGive output in the following format using the context given above:\n\n{res3['form']}\n\nUse proper headings in markdown format wherever necessary.\n\n {note} \n\n The length of generated output should be {length.lower()}."

            # For getting the chain id for tracing the time taken using Langsmith
            with callbacks.collect_runs() as cb:
                result = chain4.invoke({"context": f"{final_prompt}\n\nGive output in the following format using the context given above:\n\n{res3['form']}\n\n Don't give random answers or makeup answers about topic not asked by the user. Stick to the query asked and if required use the chat history given. \n\n Use proper headings in markdown format wherever necessary.\n\n {note1} \n\n The length of generated output should be {length.lower()}. \n\n {history_prompt}"})
                # result = conversation.predict(input=final_input)
                run_id = cb.traced_runs[0].id
                chain_id["chain4"] = run_id
            
            # st.subheader("Final output:")
            # st.write(result["answer"])
            full_res = result["text"]
        else:
            with callbacks.collect_runs() as cb:
                if uploaded_files:
                    result = chain4.invoke({"input": f"{user_input} \n\n {note1} \n\n {history_prompt} \n\nThe length of generated output should be {length.lower()}.", "chat_history": chat_history})
                    full_res = result["answer"]
                else:
                    result = chain4.invoke({"context": f"{user_input} Give output based on the context above. Don't give random answers or makeup answers about topic not asked by the user. Stick to the query asked and if required use the chat history given (then combine both answer correctly). \n\n {note1} \n\n {history_prompt} \n\n The length of generated output should be {length.lower()}."})
                    full_res = result["text"]
                run_id = cb.traced_runs[0].id
                chain_id["chain4"] = run_id

        history += f"user: {user_input} \n\n"
        history += f"Chatbot: {full_res} \n\n" 
        # print(chat_history)
        # Open the file in write mode and write the string to it
        with open(r"history.txt", "w") as file:
            file.write(history)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(full_res)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_res})

        df, time_taken = get_chain_data(chain_id)

        st.sidebar.text(f"Total time taken: {round(time_taken, 3)} s")
        # st.sidebar.text(f"Total tokens used: {tokens}")
        # st.sidebar.text(f"Time per token: {round(time_taken*1000/tokens, 3)} ms")
        # detail = st.sidebar.button("view detail")
        st.sidebar.table(df)
        

if __name__ == "__main__":
    main()
