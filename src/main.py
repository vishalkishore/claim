# import pymupdf # imports the pymupdf library
# import os

# doc = pymupdf.open("goog-10-k-2023.pdf") # open a document
# text = "" # initialize an empty string
# for page in doc: # iterate the document pages
#   text += page.get_text() # get plain text encoded as UTF-8

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def split_text(text, chunk_size=500, chunk_overlap=100):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return text_splitter.split_text(text)

# os.environ['GOOGLE_API_KEY'] = 'AIzaSyBnUrkdKsrOHOQJwn5NOd9nnk8b69JKcbQ'

# from langchain import PromptTemplate
# from langchain import hub
# from langchain.docstore.document import Document
# from langchain.schema import StrOutputParser
# from langchain.schema.prompt_template import format_document
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.vectorstores import Chroma

# docs =  [Document(page_content=text, metadata={"source": "local"})]

# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# vectorstore = Chroma.from_documents(
#                      documents=docs,                 # Data
#                      embedding=gemini_embeddings,    # Embedding model
#                      persist_directory="./chroma_db" # Directory to save data
#                      )
# vectorstore_disk = Chroma(
#                         persist_directory="./chroma_db",       # Directory of db
#                         embedding_function=gemini_embeddings   # Embedding model
#                    )
# from langchain_google_genai import ChatGoogleGenerativeAI

# llm = ChatGoogleGenerativeAI(model="gemini-pro",
#                  temperature=0.7, top_p=0.85)

# llm_prompt_template = """You are an assistant for question-answering tasks.
# Use the following context to answer the question.
# If you don't know the answer, just say that you don't know.
# Use five sentences maximum and keep the answer concise.\n
# Question: {question} \nContext: {context} \nAnswer:"""

# llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# print(llm_prompt)

# # Combine data from documents to readable string format.
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Create stuff documents chain using LCEL.
# #
# # This is called a chain because you are chaining together different elements
# # with the LLM. In the following example, to create the stuff chain, you will
# # combine the relevant context from the website data matching the question, the
# # LLM model, and the output parser together like a chain using LCEL.
# #
# # The chain implements the following pipeline:
# # 1. Extract the website data relevant to the question from the Chroma
# #    vector store and save it to the variable `context`.
# # 2. `RunnablePassthrough` option to provide `question` when invoking
# #    the chain.
# # 3. The `context` and `question` are then passed to the prompt where they
# #    are populated in the respective variables.
# # 4. This prompt is then passed to the LLM (`gemini-pro`).
# # 5. Output from the LLM is passed through an output parser
# #    to structure the model's response.
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | llm_prompt
#     | llm
#     | StrOutputParser()
# )


# rag_chain.invoke("What is Gemini?")

from claim_risk_predictor import ClaimRiskPredictor
import json

def main():
    predictor = ClaimRiskPredictor()
    file_paths = ["./src/car.pdf"]
    query = "Analyze the claim risk considering policy terms and claim history"
    
    result = predictor.process_claim(file_paths, query)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()