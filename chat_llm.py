# pip install python-dotenv langchain langchain-openai langchain-community langchain-text-splitters docx2txt langchain-chroma
# pip install langchain_upstage
# pip install langchain-pinecone

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain import hub
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from chat_config import answer_examples

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
  ### Chroma
  # embedding = UpstageEmbeddings(model='solar-embedding-1-large')

  # 이미 저장된 데이터를 사용할 때 
  # database = Chroma(collection_name='chroma-job', persist_directory="./chroma", embedding_function=embedding)
  # database.delete_collection()

  # 데이터를 처음 저장할 때 
  # database = Chroma.from_documents(documents=document_list, embedding=embedding, collection_name='chroma-job', persist_directory="./chroma")

  ### Pinecone
  embedding = OpenAIEmbeddings(model='text-embedding-3-large')
  
  index_name = "bgwell-guide"
  database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

  # database = PineconeVectorStore.from_documents(documents=document_list, embedding=embedding, index_name=index_name)

  retriever = database.as_retriever()
  return retriever


# def get_llm(model='gpt-4o-mini'):
#     llm = ChatOpenAI(model=model)
#     return llm


def get_llm(model='solar-mini'):
  llm = ChatUpstage(model=model)
  return llm


def get_dictionary_chain():
  dictionary = ["근로자를 나타내는 표현 -> 사원"]
  llm = get_llm()

  prompt = ChatPromptTemplate.from_template(f"""
          사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
          만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
          그런 경우에는 질문만 리턴해주세요
          사전: {dictionary}
          
          질문: {{question}}
      """)
  
  dictionary_chain = prompt | llm | StrOutputParser()
    
  return dictionary_chain


def get_history_retriever():
  llm = get_llm()
  retriever=get_retriever()

  # qa_chain = RetrievalQA.from_chain_type(
  #     llm, 
  #     retriever=retriever,
  #     chain_type_kwargs={"prompt": prompt}
  # )

  contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
  )

  contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", contextualize_q_system_prompt),
      MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
    ]
  )

  history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
  )

  return history_aware_retriever


def get_rag_chain():
  llm = get_llm()

  example_prompt = ChatPromptTemplate.from_messages(
      [
          ("human", "{input}"),
          ("ai", "{answer}"),
      ]
  )
  few_shot_prompt = FewShotChatMessagePromptTemplate(
      example_prompt=example_prompt,
      examples=answer_examples,
  )

  system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
  )

  qa_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system_prompt),
      few_shot_prompt,
      MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
    ]
  )

  history_aware_retriever = get_history_retriever()
  question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

  rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

  conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
  ).pick('answer')

  return conversational_rag_chain


def get_ai_response(user_message):
  # query = '입사 6년차인 직원의 연차는 며칠인가요?'
  dictionary_chain = get_dictionary_chain()
  # input_query = dictionary_chain.invoke({"input": user_message})
  rag_chain = get_rag_chain()
  chat_chain = {"input": dictionary_chain} | rag_chain
  ai_response = chat_chain.stream(
    {
      "question": user_message
    },
    config={
      "configurable": {"session_id": "abc123"}
    },
  )

  return ai_response