#!/usr/bin/env python3
""" Copyright 2025 Emanuel Bierschneider

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. """
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*pydantic_v1.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*resume_download.*")

# Suppress noisy LangChain / Pydantic / HuggingFace warnings in CLI runs
warnings.filterwarnings("ignore", message=r".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"huggingface_hub.file_download")
warnings.filterwarnings("ignore", module=r"pydantic\._internal\._generate_schema")
# Suppress specific warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore")

import os
from datetime import datetime
import subprocess
from typing import Any, List
from io import StringIO
from glob import glob
import json
import jsonstreams
import argparse
import sys
import asyncio
import re
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from pydantic import Field
from langchain_core.tools import BaseTool
from langchain_community.embeddings import HuggingFaceEmbeddings

# CrewAI imports (only imported when needed)
try:
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai_tools import tool, BaseTool
    import wikipediaapi
    from langchain_community.tools import DuckDuckGoSearchResults
    from llmsherpa.readers import LayoutPDFReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    #from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS #from langchain.vectorstores import FAISS
    from langchain.docstore.document import Document
    from contextlib import redirect_stdout
    from contextlib import redirect_stderr
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# RAG workflow imports (only imported when needed)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import FAISS as RAG_FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_community.chat_models import ChatOllama
    from langchain_community.llms import Ollama
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.documents import Document as RAG_Document
    from langchain_community.tools import DuckDuckGoSearchRun
    from langgraph.graph import END, StateGraph
    from typing_extensions import TypedDict
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


os.environ["USER_AGENT"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

# Global variables for RAG
global vectorstore, retriever, question

def generate_stream_json_response(prompt, model="llama3.1:8b"):
    data = json.dumps({"model": model, "prompt": prompt})
    process = subprocess.Popen(["curl", "-X", "POST", "-d", data, "http://localhost:11434/api/generate"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    full_response = ""
    with jsonstreams.Stream(jsonstreams.Type.array, filename='./response_log.txt') as output:
        while True:
            line, _ = process.communicate()
            if not line:
                break
            try:
                record = line.decode("utf-8").split("\n")
                for i in range(len(record)-1):
                    data = json.loads(record[i].replace('\0', ''))
                    if "response" in data:
                        full_response += data["response"]
                        with output.subobject() as output_e:
                            output_e.write('response', data["response"])
                    else:
                        return full_response.replace('\0', '')
                if len(record)==1:
                    data = json.loads(record[0].replace('\0', ''))
                    if "error" in data:
                        full_response += data["error"]
                        with output.subobject() as output_e:
                            output_e.write('error', data["error"])
                return full_response.replace('\0', '')
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error)
    return full_response.replace('\0', '')

# RAG workflow functions (only available if imports successful)
if RAG_AVAILABLE:
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def faiss_get_num_items(vect):
        return vect.index.ntotal

    def faiss_search_id_by_contex(vect, context):
        for _id, doc in vect.docstore._dict.items():
            if(context in doc.page_content):
                return _id
        return 0
    def faiss_delete_by_contex(vect, context):
        for _id, doc in vect.docstore._dict.items():
            if(context in doc.page_content):
                id_id = []
                id_id.append(_id)
                vect.delete(id_id[0:1])
        return 0
    def faiss_clear(vect):
        for k in range(vect.index.ntotal):
            copy_safe = vect.docstore._dict.items()
            for _id, doc in copy_safe:
                id_id = []
                id_id.append(_id)
                vect.delete(id_id[0:1])
                break
        return 0
    def faiss_delete_by_id(vect, id):
        id_id = []
        id_id.append(id)
        vect.delete(id_id[0:1])
    def faiss_dump(vect):
        k=0
        for _id, doc in vect.docstore._dict.items():
            print(k, _id, doc)
            k += 1

    def rag_vectorstore_search(query: str, k: int = 4) -> str:
        """Search the RAG vector store for relevant content with similarity scores"""
        global vectorstore
        try:
            # Initialize vector store if not exists
            if 'vectorstore' not in globals() or vectorstore is None:
                initialize_rag_vectorstore()

            if vectorstore is None:
                return "No vector store exists. Please create one first."

            results = vectorstore.similarity_search_with_score(query=query, k=k)

            if not results:
                return "No results found."

            print("______________________")
            print(f"Query: {query}")
            print("______________________")

            formatted_results = []
            for doc, score in results:
                source = doc.metadata.get("source", "unknown source")
                content = doc.page_content
                print(f"* [SIM={score:.3f}] {content[:200]}... [{source}]")
                formatted_results.append(f"[Similarity: {score:.3f}]\nFrom {source}:\n{content}\n")

            ret = "\n---\n".join(formatted_results)
            return ret

        except Exception as e:
            error_msg = f"Error searching vector store: {str(e)}"
            print(error_msg)
            return error_msg

    def initialize_rag_vectorstore(local_llm='llama3.1:8b'):
        """Initialize the RAG vector store with documents"""
        global vectorstore, retriever

        # Check if local vectorstore exists
        vectorstore_dir = os.path.expanduser("~/vectorstore_aia/vectorstore.db")
        vectorstore_index = os.path.join(vectorstore_dir, "index.faiss")
        if os.path.exists(vectorstore_index):
            print(f"Loading existing RAG vectorstore from {vectorstore_dir}")
            try:
                vectorstore = RAG_FAISS.load_local(
                    vectorstore_dir,
                    HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'),
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading existing vectorstore: {e}")
                print("Creating new vectorstore from sources...")
                vectorstore = create_new_vectorstore(create_from_web=False)

        else:
            print("Creating new RAG vectorstore from sources...")
            vectorstore = create_new_vectorstore(create_from_web=False)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 4}
        )
        return vectorstore, retriever

    def create_new_vectorstore(create_from_web: bool = False):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=250,
                chunk_overlap=0,
                separators=[".", ",", "?", "\n"]
            )
            if create_from_web:
                """Create a new vectorstore from web sources"""
                urls = [
                    "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
                    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
                ]

                docs = [WebBaseLoader(url).load() for url in urls]
                docs_list = [item for sublist in docs for item in sublist]
                doc_splits = text_splitter.split_documents(docs_list)
                vectorstore = RAG_FAISS.from_documents(
                    doc_splits,
                    HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
                )
            else:
                todo_file = os.path.expanduser("/mnt/c/Users/bierschneider_e1/Desktop/todo.txt")
                if not os.path.exists(todo_file):
                    print(f"Warning: Source file not found: {todo_file}")
                    print("Creating vectorstore with sample data instead...")
                    # Create a simple sample document
                    sample_text = "This is a sample vectorstore. Add your own documents later."
                    chunked_text_documents = text_splitter.create_documents([sample_text])
                else:
                    with open(todo_file, encoding="utf8") as f:
                        read_text_to_be_split = f.read()
                    chunked_text = text_splitter.split_text(read_text_to_be_split)
                    chunked_text_documents = text_splitter.create_documents(chunked_text)

                vectorstore = RAG_FAISS.from_documents(
                    chunked_text_documents,
                    HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
                )

            # Save vectorstore
            print("Saving RAG vectorstore locally to ~/vectorstore_aia/vectorstore.db")
            vectorstore_dir = os.path.expanduser("~/vectorstore_aia")
            os.makedirs(vectorstore_dir, exist_ok=True)
            vectorstore.save_local(os.path.join(vectorstore_dir, "vectorstore.db"))
            print("Vectorstore created and saved successfully!")

            return vectorstore
        except Exception as e:
            print(f"Error creating vectorstore: {e}")
            import traceback
            traceback.print_exc()
            raise

    def stream_rag_response(user_prompt, system_prompt="You are a helpful assistant", local_llm='myllama3'):
        """Stream response from local LLM for RAG"""
        template = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {system_prompt}
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {user_prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """

        prompt = PromptTemplate(
            input_variables=["system_prompt", "user_prompt"],
            template=template
        )

        llm = Ollama(model=local_llm, stop=["<|eot_id|>"])
        response = llm(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt))
        return response

    # RAG State definition
    class RAGGraphState(TypedDict):
        """
        Represents the state of our RAG graph.

        Attributes:
            question: question
            generation: LLM generation
            code_tool: whether to add search
            documents: list of documents
        """
        question: str
        generation: str
        code_tool: str
        documents: List[str]

    # RAG Python code execution tool
    class RAGExecutePythonCode(BaseTool):
        name: str = Field(default="rag_code_tool")
        description: str = Field(default="Only use this code tool if you have to write python code and need execute it. Only if need to execute python code!")
        local_llm: str = Field(default='myllama3')
        def __init__(self, local_llm='myllama3'):
            super().__init__()
            self.local_llm = local_llm

        async def _arun(self, response: str):
            try:
                prompt = PromptTemplate(
                    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert in writing python code to solve complex tasks. Do answer with python code only. Include all neccessary python libraries at the beginning of the code. Question related to coding and code execution to solve: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                    input_variables=["question"],
                )
                llm = ChatOllama(model=self.local_llm, temperature=0)
                rag_chain_local = prompt | llm | StrOutputParser()
                response = rag_chain_local.invoke({"question": question})

                python_code_match = re.findall(r"```python\s+(.*?)\s+```", response, re.DOTALL)
                if python_code_match:
                    python_code = python_code_match[0].replace("\\n", "\n")
                    print("PYTHON CODE FOUND")
                    print(python_code)

                    # Auto-execute without user confirmation for CLI usage
                    my_stdout = StringIO()
                    try:
                        with redirect_stdout(my_stdout):
                            exec(python_code)
                        code_exec_output = my_stdout.getvalue()
                        print(code_exec_output)
                        return code_exec_output
                    except Exception as error:
                        error_msg = f"An exception occurred: {error}"
                        print(error_msg)
                        return error_msg
                else:
                    return "No Python code block found in response"
            except Exception as e:
                return f"Error in code execution: {str(e)}"

        def _run(self, response: str):
            # Synchronous version that calls async version
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(response))
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def create_rag_workflow(local_llm='myllama3'):
        """Create the RAG workflow with all components"""
        global vectorstore, retriever, question

        # Initialize vectorstore if not already done
        if 'vectorstore' not in globals() or vectorstore is None:
            initialize_rag_vectorstore(local_llm)

        # Initialize tools
        web_search_tool = DuckDuckGoSearchRun()
        execute_python_code_tool = RAGExecutePythonCode(local_llm)

        # Create graders and generators
        retrieval_grader = create_rag_retrieval_grader(local_llm)
        rag_chain = create_rag_generation_chain(local_llm)
        hallucination_grader = create_rag_hallucination_grader(local_llm)
        answer_grader = create_rag_answer_grader(local_llm)
        question_router = create_rag_question_router(local_llm)

        # RAG Graph nodes
        def retrieve(state):
            """Retrieve documents from vectorstore"""
            print("---RETRIEVE---")
            question = state["question"]
            documents = retriever.invoke(question)
            return {"documents": documents, "question": question}

        def generate(state):
            """Generate answer using RAG on retrieved documents"""
            print("---GENERATE---")
            question = state["question"]
            documents = state["documents"]

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
            return {"documents": documents, "question": question, "generation": generation}

        def grade_documents(state):
            """Determines whether the retrieved documents are relevant to the question"""
            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
            question = state["question"]
            documents = state["documents"]

            filtered_docs = []
            code_tool = "No"
            for d in documents:
                score = retrieval_grader.invoke({"question": question, "document": d.page_content})
                grade = score['score']
                if grade.lower() == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    code_tool = "Yes"
                    continue
            return {"documents": filtered_docs, "question": question, "code_tool": code_tool}

        async def web_duck_search(question):
            return web_search_tool.run(question)

        def web_search(state):
            """Web search based on the question"""
            print("---WEB SEARCH---")
            question = state["question"]
            documents = state["documents"]

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                web_results = loop.run_until_complete(web_duck_search(question))
                print(web_results)
            finally:
                asyncio.set_event_loop(None)
                loop.close()

            web_results = RAG_Document(page_content=web_results)
            if documents is not None:
                documents.append(web_results)
            else:
                documents = [web_results]
            return {"documents": documents, "question": question}

        def code_tool(state):
            """Write python code based on the question"""
            print("---CODE TOOL---")
            question = state["question"]
            documents = state["documents"]

            code_results = execute_python_code_tool.run(question)

            if code_results is not None:
                code_results = RAG_Document(page_content=str(code_results))
                if documents is not None:
                    documents.append(code_results)
                else:
                    documents = [code_results]
            return {"documents": documents, "question": question}

        # Conditional edges
        def route_question(state):
            """Route question to code tool or RAG."""
            print("---ROUTE QUESTION---")
            question = state["question"]
            source = question_router.invoke({"question": question})
            print(f"Routing to: {source['datasource']}")

            if source['datasource'] == 'code_tool':
                print("---ROUTE QUESTION TO CODE TOOL---")
                return "codetool"
            elif source['datasource'] == 'vectorstore':
                print("---ROUTE QUESTION TO RAG---")
                return "vectorstore"

        def decide_to_generate(state):
            """Determines whether to generate an answer, or add code tool"""
            print("---ASSESS GRADED DOCUMENTS---")
            code_tool = state["code_tool"]

            if code_tool == "Yes":
                print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE CODE TOOL---")
                return "codetool"
            else:
                print("---DECISION: GENERATE---")
                return "generate"

        def grade_generation_v_documents_and_question(state):
            """Determines whether the generation is grounded in the document and answers question."""
            print("---CHECK HALLUCINATIONS---")
            question = state["question"]
            documents = state["documents"]
            generation = state["generation"]

            score = hallucination_grader.invoke({"documents": documents, "generation": generation})
            grade = score.get('score', 'yes')

            if grade == "yes":
                print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                print("---GRADE GENERATION vs QUESTION---")
                score = answer_grader.invoke({"question": question, "generation": generation})
                grade = score['score']
                if grade == "yes":
                    print("---DECISION: GENERATION ADDRESSES QUESTION---")
                    return "useful"
                else:
                    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                    return "not useful"
            else:
                print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                return "not supported"

        # Create workflow
        workflow = StateGraph(RAGGraphState)

        # Define the nodes
        workflow.add_node("codetool", code_tool)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)

        # Build graph
        workflow.set_conditional_entry_point(
            route_question,
            {
                "codetool": "codetool",
                "vectorstore": "retrieve",
            },
        )

        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "codetool": "codetool",
                "generate": "generate",
            },
        )
        workflow.add_edge("codetool", "generate")
        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "codetool",
            },
        )

        return workflow.compile()

    def create_rag_retrieval_grader(local_llm):
        """Create document relevance grader"""
        llm = ChatOllama(model=local_llm, format="json", temperature=0)
        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
            of a retrieved document to a user question. If the document contains keywords related to the user question,
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide only only binary score as a JSON with a single key 'score' and no premable or explaination.
            Always return JSON with a single key 'score' that includes the binary value 'yes' or 'no.
             <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["question", "document"],
        )
        return prompt | llm | JsonOutputParser()

    def create_rag_generation_chain(local_llm):
        """Create RAG generation chain"""
        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {question}
            Context: {context}
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question", "context"],
        )
        llm = ChatOllama(model=local_llm, temperature=0)
        return prompt | llm | StrOutputParser()

    def create_rag_hallucination_grader(local_llm):
        """Create hallucination grader"""
        llm = ChatOllama(model=local_llm, format="json", temperature=0)
        prompt = PromptTemplate(
            template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
            an answer is grounded in or supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate
            whether the answer is grounded in or supported by a set of facts. Provide only a binary score as a JSON with only one
            single key 'score' and no preamble or explanation or any other key.
            Always return a JSON with only one single key 'score' that includes the binary value 'yes' or 'no.'<|eot_id|><|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ------- \n
            {documents}
            \n ------- \n
            Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "documents"],
        )
        return prompt | llm | JsonOutputParser()

    def create_rag_answer_grader(local_llm):
        """Create answer grader"""
        llm = ChatOllama(model=local_llm, format="json", temperature=0)
        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
            useful to resolve a question. Provide only the binary score as a JSON with a single key 'score' and no preamble or explanation and nothing else.
            Always return a JSON with only one single key 'score' that includes the binary value 'yes' or 'no.
             <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
            \n ------- \n
            {generation}
            \n ------- \n
            Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "question"],
        )
        return prompt | llm | JsonOutputParser()

    def create_rag_question_router(local_llm):
        """Create question router"""
        llm = ChatOllama(model=local_llm, format="json", temperature=0)
        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a
            user question to a vectorstore or code tool. Use the vectorstore for questions on LLM agents,
            prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords
            in the question related to these topics. Otherwise, use code_tool. Give a binary choice 'code_tool'
            or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and
            no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question"],
        )
        return prompt | llm | JsonOutputParser()

    def run_rag_workflow(prompt, model='myllama3'):
        """Run the RAG workflow with the given prompt"""
        global question

        question = prompt

        # Create and run workflow
        app = create_rag_workflow(model)
        inputs = {"question": prompt}

        final_result = None
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Finished running: {key}")
            final_result = value

        if final_result and "generation" in final_result:
            return final_result["generation"]
        else:
            return "No generation found in workflow output"

# CrewAI functions (only available if imports successful)
if CREWAI_AVAILABLE:
    def get_wiki_content(key_word):
        try: # Wikipedia API ready
            wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
            page = wiki_wiki.page(key_word)
            if page.exists(): # Page - Exists: True
                print("Page - Title:", page.title)
                print("Page - Summary:", page.summary)
                return page.summary
            else:
                print("Page not found.")
            return page.summary
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error)
        return ""

    class ExecutePythonCode(BaseTool):
        name: str = "ExecutePythonCode"
        description: str = "Only use this tool if you have to write python code and execute it, but make sure you have meaningful \
                            debug output and always provide the complete code with includes not only the function definition. Only if need to execute python code!"

        def _run(self, response: str):
            try:
                python_code_match = re.findall(r"```python\s+(.*?)\s+```", response, re.DOTALL)
                if len(python_code_match) > 0:
                    print("PYTHON CODE FOUND")
                    python_code_match = python_code_match[0].replace("\\n", "\n")
                else:
                    python_code_match = re.findall(r"```(.*?)```", response, re.DOTALL)
                    if len(python_code_match) > 0:
                        print("CODE FOUND")
                        python_code_match = python_code_match[0].replace("\\n", "\n")
                    else:
                        pattern = r"^((?:(?:import|from) .*\n)+.*)"
                        match = re.match(pattern, response, re.DOTALL)
                        if match:
                            python_code_match = match.group(1)
                            print("IMPORT BLOCK FOUND")
                        else:
                            print("Error: No valid python code")
                            return ("Error: No valid python code")

                if input(f"<>PYTHON CODE FOUND, EXECUTE? ").upper() == 'Y':
                    code_to_exec = python_code_match.replace("\\'", "'").replace('\\"', '"')
                    code_to_exec = python_code_match.replace("\\n", "\n").replace("\\t", "\t")
                    code_to_exec = python_code_match.replace("\\\\n", "\n")
                    while True:
                        try:
                            if "rm -r" in code_to_exec or "rm -f" in code_to_exec or "rm " in code_to_exec:
                                raise ValueError("Dangerous command detected in python code.")
                            buffer = StringIO()
                            with redirect_stdout(buffer), redirect_stderr(buffer):
                                exec(code_to_exec, globals())
                            # Retrieve everything printed or errored
                            code_exec_output = buffer.getvalue()
                            # Reset buffer to loop again
                            buffer.truncate(0)
                            buffer.seek(0)
                            print(code_exec_output)  # Optional: still show in terminal
                            return(code_exec_output)
                        except Exception as error:
                            print("An exception occurred:", error)
                            if "No module named" in str(error):
                                module_match = re.search(r"No module named '([^']+)'", str(error))
                                if module_match:
                                    module_name = module_match.group(1)
                                    print(f"Module name parsed: {module_name}")
                                    print(f"Trying to install: {module_name}")
                                    try:
                                        result = subprocess.run(
                                            ["pip", "install", module_name],
                                            capture_output=True,
                                            text=True,
                                            check=True
                                        )
                                        print(result.stdout)
                                    except subprocess.CalledProcessError as error:
                                        print("An exception occurred:", error)
                                        print("Error output:", error.stderr)
                                        return f"An exception occurred: {error}"
                            else:
                                print("ExecutePythonCode exception occurred:", error)
                                return (f"ExecutePythonCode exception occurred: {error}")
            except Exception as e:
                print(f"This is not a valid python code search syntax. Try a different string based syntax. {e}")
                return "This is not a valid python code search syntax. Try a different string based syntax."

            def _arun(self, radius: int):
                raise NotImplementedError("This tool does not support async")

    execute_python_code_tool = ExecutePythonCode()


    class ReadFile(BaseTool):
        name: str = Field(default="ReadFile")
        description: str = Field(default="Only use this tool if you want to read a file content. Only execute this tool if you want to read a file!")

        def _run(self, filename: str):
            try:
                if len(filename) != 0:
                    if filename != "":
                        print(filename)
                    if filename.startswith("http"):
                        print("URL found")
                        try:
                            # Use wget to fetch the content
                            command = f"wget -qO - {filename}"
                            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
                            print(result.stdout)
                            return result.stdout
                        except subprocess.CalledProcessError as e:
                            print(f"Error running command: {str(e)}")
                            print("Standard Error Output:", e.stderr)
                    elif filename.endswith("pdf"):
                        print("PDF found")
                        try:
                            # Use pdftotext for pdf to text conversion and pipe it to cat
                            command = f"pdftotext {filename} -"
                            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
                            print(result.stdout)
                            return result.stdout
                        except subprocess.CalledProcessError as e:
                            print(f"Error running command: {str(e)}")
                            print("Standard Error Output:", e.stderr)
                    else:
                        print("Filename found")
                        try:
                            # Using subprocess to execute 'cat' command
                            result = subprocess.run(['cat', filename],  capture_output=True,  text=True,  check=True)
                            print(f"Reading file: {filename}")
                            print(result.stdout)
                            return result.stdout
                        except subprocess.CalledProcessError as e:
                            error_msg = f"Error reading file with cat: {str(e)}"
                            print(error_msg)
                            return error_msg
                        except Exception as error:
                            error_msg = f"Error: {str(error)}"
                            print(error_msg)
                            return error_msg
            except Exception:
                return "This is not a valid python code search syntax. Try a different string based syntax."

            def _arun(self, radius: int):
                raise NotImplementedError("This tool does not support async")

    read_file_tool = ReadFile()

    class ExecuteLinuxShellCommand(BaseTool):
        name: str = Field(default="ExecuteLinuxShellCommand")
        description: str = Field(default="Only use this tool if you want to run a Linux or UNIX program on this computer. Only execute this tool if run a shell command!")

        def _run(self, command: str):
            try:
                if len(command) != 0:
                    if command != "":
                        print(command)
                    print("Command found")
                    try:
                        # Using subprocess to execute command
                        result = subprocess.run([command], shell=True, capture_output=True, text=True, check=True)
                        print(f"Executing: {command}")
                        print(result.stdout)
                        return result.stdout
                    except subprocess.CalledProcessError as e:
                        error_msg = f"Error {e.stderr}: {str(e)}"
                        print(error_msg)
                        return error_msg
                    except Exception as error:
                        error_msg = f"Error: {str(error)}"
                        print(error_msg)
                        return error_msg
            except Exception:
                return "This is not a valid python code search syntax. Try a different string based syntax."

            def _arun(self, radius: int):
                raise NotImplementedError("This tool does not support async")

    executed_linux_shell_command_tool = ExecuteLinuxShellCommand()

    class PDFVectorStoreTool(BaseTool):
        name: str = "pdf_vector_store"
        description: str = "Use this tool to read cybersecurity related PDFs to gain a deep penetration test knowledge and understanding."
        # Define class attributes properly with Field
        llmsherpa_api_url: str = Field(default="http://localhost:5001/api/parseDocument?renderFormat=all")
        pdf_reader: Any = Field(default=None)
        text_splitter: Any = Field(default=None)
        embeddings: Any = Field(default=None)
        vector_store: Any = Field(default=None)
        dir: str = Field(default=None)

        def __init__(self, **data):
            super().__init__(**data)
            try:
                self.pdf_reader = LayoutPDFReader(self.llmsherpa_api_url)
                self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # About 1500-2000 tokens, safe for 8k context
                chunk_overlap=200,
                length_function=len,
                    separators=[
                    "\n\n\n",  # Triple line breaks (paragraphs)
                    "\n\n",    # Double line breaks (paragraphs)
                    "\n",      # Single line breaks
                ],
                keep_separator=True,
                is_separator_regex=False
                )
                # Initialize HuggingFace embeddings with a good model for semantic search
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={'device': 'cpu'}, # Use CPU by default
                    encode_kwargs={'normalize_embeddings': True} # Normalize embeddings for better similarity search
                )
                self.vector_store = None
                self.dir = os.path.expanduser("~/")
                self.create_vector_store(self.dir)
            except Exception as e:
                print(f"Warning: PDF Vector Store initialization failed: {e}")
                self.vector_store = None

        def vectorstore_get_num_items(self):
            if self.vector_store:
                return self.vector_store.index.ntotal
            return 0

        def vectorstore_search_id_by_contex(self, context):
            if not self.vector_store:
                return 0
            for _id, doc in self.vector_store.docstore._dict.items():
                if(context in doc.page_content):
                    return _id
            return 0

        def vectorstore_delete_by_contex(self, context):
            if not self.vector_store:
                return 0
            for _id, doc in self.vector_store.docstore._dict.items():
                if(context in doc.page_content):
                    id_id = []
                    id_id.append(_id)
                    self.vector_store.delete(id_id[0:1])
            return 0

        def vectorstore_clear(self):
            if not self.vector_store:
                return 0
            for k in range(self.vector_store.index.ntotal):
                copy_safe = self.vector_store.docstore._dict.items()
                for _id, doc in copy_safe:
                    id_id = []
                    id_id.append(_id)
                    self.vector_store.delete(id_id[0:1])
                    break
            return 0

        def vectorstore_delete_by_id(self, id):
            if not self.vector_store:
                return
            id_id = []
            id_id.append(id)
            self.vector_store.delete(id_id[0:1])

        def vectorstore_dump(self):
            if not self.vector_store:
                print("No vector store available")
                return
            k=0
            for _id, doc in self.vector_store.docstore._dict.items():
                print(k, _id, doc)
                k += 1

        def save_vector_store(self):
            if self.vector_store:
                self.vector_store.save_local(self.dir + "/vectorstore_aia/vectorstore.db")

        def load_vector_store(self):
            try:
                self.vector_store = FAISS.load_local(self.dir + "/vectorstore_aia/vectorstore.db", self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print("Error: Loading vectorstore failed:", e)

        def search_vectorstore(self, input_query: str):
            if not self.vector_store:
                print("No vector store available")
                return
            print("______________________")
            print(input_query)
            print("______________________")
            results = self.vector_store.similarity_search_with_score(query=input_query,k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]" + "\n---\n")

        def vectorstore_search(self, query: str, k: int = 2) -> str:
            """Search the vector store for relevant content"""
            try:
                # Try to load vector store if not exists
                if not self.vector_store:
                    return "No vector store exists. Please create one first."
                results = self.vector_store.similarity_search(query, k=k)
                formatted_results = []
                for doc in results:
                    source = doc.metadata.get("source", "unknown source")
                    content = doc.page_content
                    formatted_results.append(f"From {source}:\n{content}\n")
                ret = "\n---\n".join(formatted_results)
                print(ret)
                return ret

            except Exception as e:
                return f"Error searching vector store: {str(e)}"

        def create_vector_store(self, directory_path: str) -> str:
            """Create a FAISS vector store from all PDFs in a directory"""
            try:
                # Save vector store
                if not os.path.exists(self.dir + "vectorstore_aia"):
                    # Verify input directory exists
                    if not os.path.exists(directory_path):
                        print(f"Warning: Directory not found at {directory_path}")
                        return f"Error: Directory not found at {directory_path}"
                    # Get all PDF files in directory
                    pdf_files = glob(os.path.join(directory_path, "*.pdf"))
                    if not pdf_files:
                        print(f"No PDF files found in {directory_path}")
                        self.load_vector_store()
                        return "No PDF files found in the directory"
                    print(f"Processing {len(pdf_files)} PDF files from {directory_path}")
                    all_documents = []
                    for pdf_path in pdf_files:
                        try:
                            print(f"Processing {pdf_path}")
                            # Read PDF
                            doc = self.pdf_reader.read_pdf(pdf_path)
                            # Extract text from all pages
                            text_content = []
                            for chunk in doc.chunks():
                                text_content.append(chunk.to_context_text())
                            # Create documents
                            full_content = "\n".join(text_content)
                            texts = self.text_splitter.split_text(full_content)
                            documents = [Document(page_content=t, metadata={"source": pdf_path}) for t in texts]
                            all_documents.extend(documents)
                        except Exception as e:
                            print(f"Error processing {pdf_path}: {str(e)}")
                            continue
                    if not all_documents:
                        return "No documents were successfully processed"
                    # Create vector store
                    self.vector_store = FAISS.from_documents(all_documents, self.embeddings)
                    print("Directory not found")
                    os.makedirs(self.dir + "vectorstore_aia", exist_ok=True)
                    self.save_vector_store()
                else:
                    self.load_vector_store()
                #print(f"Vector store created successfully from PDFs and saved.")

            except Exception as e:
                print(f"Error creating vector store: {str(e)}")

        def _run(self, command: str) -> str:
            """Run the tool with query command"""
            try:
                return self.vectorstore_search(command)
            except Exception as e:
                return f"Error processing command: {str(e)}"

        def _arun(self, command: str) -> str:
            """Async implementation"""
            raise NotImplementedError("This tool does not support async")

    try:
        pdf_vector_store_tool = PDFVectorStoreTool()
    except Exception as e:
        print(f"Warning: Could not initialize PDF vector store: {e}")
        pdf_vector_store_tool = None

    @tool("pdf vector search tool.")
    def pdf_search_tool(argument: str) -> str:
        """Use this tool to search information on cybersecurity and pentesting, if you do need information."""
        if pdf_vector_store_tool:
            return pdf_vector_store_tool.run(argument)
        return "PDF vector store not available"

    @tool ("duckduck go search tool")
    def duck_duck_go_search_tool(argument: str) -> str:
        """Use this tool to search new information by crawling the web, if you do need additional information but do not use json as input!"""
        print("Duck: ", argument)
        duckduckgo_tool = DuckDuckGoSearchResults()
        response = duckduckgo_tool.invoke(argument)
        return response

    @tool("wikipedia search tool")
    def wikipedia_search_tool(argument: str) -> str:
        """Use this tool to search a single keyword on wikipedia, if you do not know the answer to a question."""
        return get_wiki_content(argument)

    def run_crew_with_task(task_description, model="ollama/openhermes"):
        """Run CrewAI with a custom task description"""

        # Convert model format from original to ollama format if needed
        if not model.startswith("ollama/"):
            model = f"ollama/{model.replace('llama3.1:8b', 'openhermes')}"

        manager_agent = Agent(
            role="Manager",
            goal="Coordinate and oversee the completion of all tasks and provide the Final Answer, but only if 100 percent sure that final solution has been found",
            backstory="An experienced project manager who coordinates team efforts",
            verbose=True,
            allow_delegation=True,
            llm=LLM(
            model=model,
            base_url="http://localhost:11434"),
        )

        general_researcher = Agent(
            role='General Researcher and Analyst',
            goal=f'Research and analyze information to solve: {task_description}',
            backstory="""You are a versatile researcher who can analyze various topics using available tools including web search, file reading, code execution, and knowledge bases""",
            verbose=True,
            allow_delegation=True,
            tools=[duck_duck_go_search_tool, pdf_search_tool, read_file_tool, wikipedia_search_tool, execute_python_code_tool, executed_linux_shell_command_tool],
            llm= LLM(
            model=model,
            base_url="http://localhost:11434")
        )

        coder = Agent(
            role='Python Coder',
            goal=f'Write Python code to help solve: {task_description}',
            backstory="""You are a Python developer who writes code to analyze data, solve computational problems, and assist with technical tasks""",
            verbose=True,
            allow_delegation=False,
            tools=[duck_duck_go_search_tool, pdf_search_tool, read_file_tool, execute_python_code_tool],
            llm= LLM(
            model=model,
            base_url="http://localhost:11434")
        )

        scripter = Agent(
            role='System Administrator',
            goal=f'Execute shell commands and scripts to help solve: {task_description}',
            backstory="""You are a system administrator who can execute shell commands and scripts to gather system information, manage files, and perform system tasks""",
            verbose=True,
            allow_delegation=False,
            tools=[duck_duck_go_search_tool, pdf_search_tool, read_file_tool, executed_linux_shell_command_tool],
            llm= LLM(
            model=model,
            base_url="http://localhost:11434")
        )

        # Create tasks based on the user input
        main_task = Task(
            description=task_description,
            expected_output='A comprehensive solution or answer to the given task',
            agent=general_researcher
        )

        coding_task = Task(
            description=f'If needed, write Python code to help with: {task_description}',
            expected_output='Python code solution if programming is required',
            agent=coder
        )

        system_task = Task(
            description=f'If needed, execute system commands to help with: {task_description}',
            expected_output='System command output if system operations are required',
            agent=scripter
        )

        crew = Crew(
            agents=[coder, scripter, general_researcher],
            tasks=[coding_task, system_task, main_task],
            verbose=True,
            manager_agent=manager_agent,
            process=Process.hierarchical
        )

        result = crew.kickoff()
        return str(result)


def help():
    logo = r"""
       _____   ___   _____   
      /  _  \ |   | /  _  \  
     /  /_\  \|   |/  /_\  \ 
    /    |    \   /    |    \
    \____|__  /___\____|__  /
            \/            \/ 
    """

    # Choose highlight colors (adjust if you want a different vibe)
    FG1 = "\033[38;2;120;200;255m"   # light cyan
    FG2 = "\033[38;2;80;150;255m"    # medium blue
    FG3 = "\033[38;2;40;100;200m"    # darker blue
    RESET = "\033[0m"

    colored = []
    for i, line in enumerate(logo.splitlines()):
        if i <= 1:
            colored.append(FG1 + line + RESET)
        elif i <= 3:
            colored.append(FG2 + line + RESET)
        else:
            colored.append(FG3 + line + RESET)

    print("\n".join(colored))

def main():
    parser = argparse.ArgumentParser(description='CLI for AiA - AI Assistant with multiple AI systems', add_help=False) # Disable default -h/--help
    parser.add_argument('--model', '-m', type=str, default='llama3.1:8b', help='Model to use (default: llama3.1:8b)')
    parser.add_argument('--agent', '-a', type=str, default=None, help='Use CrewAI agents with custom task description')
    parser.add_argument('--advanced-agent', '-aa', action='store_true', help='Use advanced RAG workflow system')
    parser.add_argument("--dump", "-d", action="store_true", help="Dump the FAISS vectorstore contents")
    parser.add_argument('--search', '-s', type=str, default=None, help='Search the RAG vectorstore with a query')
    parser.add_argument("--unit_test", "-u", action="store_true", help="Run unit tests")
    parser.add_argument("--pr_review", "-r", action="store_true", help="Run pull request review")
    parser.add_argument('--print', '-p', action='store_true', help='Print the result to stdout')
    parser.add_argument('--output-format', type=str, choices=['text', 'json'], default='text', help='Output format: text (default) or json')
    parser.add_argument('-h', '--help', action='store_true', help='Show help and extra info')

    parser.add_argument('prompt', nargs='?', default=None, help='Prompt to send (if not provided, read from stdin)')
    args = parser.parse_args()

    if args.help:
        parser.print_help()
        help()
        return

    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print('No prompt provided.', file=sys.stderr)
        sys.exit(1)

    if args.unit_test:
        unit_test_prompt = "You are a Python testing expert specializing in edge case testing. Your task is to write Python test code to thoroughly test all functions in the provided file. The test code must include: 1. Happy path scenarios for each function. 2. Edge cases, including unexpected inputs such as None, -1, 0, 1, empty lists ([]), and invalid data types. 3. Error path scenarios, ensuring the functions handle exceptions gracefully. 4. Comprehensive assertions to validate the correctness of the function outputs. 5. Tests for asynchronous functions, if applicable. The test code must be written in Python using the pytest framework. Ensure that the test cases are modular, well-documented, and include setup and teardown methods if necessary. Do not include any explanations or comments outside the test code itself. Only output the Python test code. Always test your test code using the code tool to execute the python code. Do not do a code review. Just write python test code with pytest. This is the code you should write pytest code for:"
        prompt = unit_test_prompt + "\n" + prompt

    if args.pr_review:
        pr_review_prompt = "You are a senior software engineer and code reviewer specializing in Python code quality, security, and correctness. Your task is to perform a detailed Pull Request review of the provided code changes. You must analyze the diff as if it were a GitHub PR and output only the structured review text. Your review must include: 1. Identification of logical or functional errors, including incorrect assumptions, missing edge case handling, and potential runtime failures. 2. Style, structure, and maintainability issues following PEP 8 and Pythonic best practices. 3. Security concerns such as unsafe input handling, data exposure, or weak cryptographic use. 4. Performance issues, such as unnecessary computations, unbounded loops, or inefficient data structures. 5. Missing or insufficient unit tests, especially for edge cases (e.g., None, -1, 0, 1, empty lists ([]), invalid inputs). 6. Documentation gaps — missing docstrings, unclear naming, or lack of comments explaining nontrivial logic. 7. Suggestions for improvement that are concise, actionable, and specific. Formatting requirements:  Provide your output as a structured PR review, not code. - Use Markdown formatting with sections: - Summary - Strengths - Issues Found - Suggestions for Improvement  Do not rewrite the code or include any Python source unless you are showing a short snippet to illustrate a fix. - Focus on correctness, maintainability, and developer clarity. - Only output the PR review content — no additional commentary, explanations, or metadata. This is the Pull Request diff you should review:"
        prompt = pr_review_prompt + "\n" + prompt

    # Check if the --dump option is provided
    if args.dump:
        if 'vectorstore' in globals() and vectorstore is not None:
            print("Dumping FAISS vectorstore contents:")
            faiss_dump(vectorstore)
        else:
            initialize_rag_vectorstore()
            print("Dumping FAISS vectorstore contents:")
            faiss_dump(vectorstore)

    # Check if the --search option is provided
    if args.search:
        if not RAG_AVAILABLE:
            print('RAG dependencies not available. Please install required packages.', file=sys.stderr)
            sys.exit(1)
        result = rag_vectorstore_search(args.search, k=4)
        if args.output_format == 'json':
            output = json.dumps({'result': result})
            print(output)
        else:
            print(result)
        sys.exit(0)

    # Determine which system to use
    if args.advanced_agent:
        if not RAG_AVAILABLE:
            print('RAG dependencies not available. Please install required packages.', file=sys.stderr)
            sys.exit(1)
        # Use RAG workflow system
        result = run_rag_workflow(prompt, model=args.model)
    elif args.agent:
        if not CREWAI_AVAILABLE:
            print('CrewAI dependencies not available. Please install required packages.', file=sys.stderr)
            sys.exit(1)
        # Use CrewAI with the agent task description
        result = run_crew_with_task(args.agent, model=args.model)
    else:
        # Use original stream JSON response
        result = generate_stream_json_response(prompt, model=args.model)

    if args.output_format == 'json':
        output = json.dumps({'result': result})
    else:
        output = result

    if args.print:
        print(output)
    else:
        print(output)

if __name__ == '__main__':
    main()
